# bank_demo_app.py
# Standalone Streamlit app focused on a real-world BANKING example
# Demonstrates: Process, Proxy, Skeleton, Identical Proxy/Skeleton, IPC vs Local,
# Forwarding Pointers, Shortcutting (clients learn new location)
# Run:  streamlit run bank_demo_app.py

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Set

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏦 Banking Demo — Proxy • Skeleton • Forwarding", layout="wide")

# -------------------------------
# Data model
# -------------------------------
@dataclass
class Process:
    name: str
    label: str
    proxies: List[str] = field(default_factory=list)
    skeletons: List[str] = field(default_factory=list)
    holds_object: bool = False

@dataclass
class ProxyRef:
    id: str
    owner_process: str
    known_skeleton_process: str  # where the client thinks the skeleton is

@dataclass
class SkeletonRef:
    id: str
    host_process: str
    forwards_to: Optional[str] = None  # forwarding pointer target process

@dataclass
class World:
    processes: Dict[str, Process]
    proxies: Dict[str, ProxyRef]
    skeletons: Dict[str, SkeletonRef]
    object_process: str
    log: List[str] = field(default_factory=list)

    def log_event(self, msg: str):
        self.log.append(msg)

# -------------------------------
# World setup & helpers
# -------------------------------

def init_banking_world() -> World:
    # P1=ATM, P2=MobileApp, P3=API Gateway (skeleton host), P4=Account Service (object)
    processes = {
        "P1": Process(name="P1", label="ATM (Client)"),
        "P2": Process(name="P2", label="MobileApp (Client)"),
        "P3": Process(name="P3", label="API Gateway (Skeleton Host)"),
        "P4": Process(name="P4", label="Account Service (Object)", holds_object=True),
    }

    # Gateway skeleton pointing to Account Service initially
    skel = SkeletonRef(id="S@P3", host_process="P3", forwards_to="P4")
    processes["P3"].skeletons.append(skel.id)

    # Identical proxies at ATM and Mobile App
    proxies = {
        "px_atm": ProxyRef(id="px_atm", owner_process="P1", known_skeleton_process="P3"),
        "px_mob": ProxyRef(id="px_mob", owner_process="P2", known_skeleton_process="P3"),
    }
    processes["P1"].proxies.append("px_atm")
    processes["P2"].proxies.append("px_mob")

    w = World(processes=processes, proxies=proxies, skeletons={skel.id: skel}, object_process="P4")
    w.log_event("Initialized: ATM & MobileApp have identical proxies → API Gateway skeleton → Account Service object.")
    return w


def move_object(w: World, dst_process: str, create_new_skeleton: bool, keep_forwarding: bool):
    if dst_process == w.object_process:
        w.log_event(f"Object already at {dst_process} — no move.")
        return
    src = w.object_process
    w.processes[src].holds_object = False
    w.processes[dst_process].holds_object = True
    w.object_process = dst_process

    # Optional: create new skeleton at destination
    if create_new_skeleton:
        sid = f"S@{dst_process}"
        if sid not in w.skeletons:
            w.skeletons[sid] = SkeletonRef(id=sid, host_process=dst_process)
            w.processes[dst_process].skeletons.append(sid)
        w.log_event(f"Object moved {src} → {dst_process}; created new skeleton at {dst_process}.")

    # Manage forwarding pointers from older skeletons
    if keep_forwarding:
        for s in w.skeletons.values():
            if s.host_process != dst_process:
                s.forwards_to = dst_process
        w.log_event(f"Forwarding pointers: old skeletons → {dst_process}.")
    else:
        for s in w.skeletons.values():
            if s.host_process != dst_process:
                s.forwards_to = None
        w.log_event("No forwarding kept; clients must learn new location via failure or discovery.")


def invoke(w: World, proxy_id: str, do_shortcut: bool = True) -> Tuple[bool, str, List[Tuple[str, str]]]:
    if proxy_id not in w.proxies:
        return False, f"Proxy {proxy_id} not found", []

    pr = w.proxies[proxy_id]
    route_text: List[str] = []
    edges: List[Tuple[str, str]] = []

    # Client → known skeleton (IPC unless same process)
    edges.append((pr.owner_process, pr.known_skeleton_process))
    hop_type = "LOCAL" if pr.owner_process == pr.known_skeleton_process else "IPC"
    route_text.append(f"{pr.owner_process} → {pr.known_skeleton_process} ({hop_type})")

    # Find skeleton
    target = None
    for s in w.skeletons.values():
        if s.host_process == pr.known_skeleton_process:
            target = s
            break
    if not target:
        w.log_event(f"{proxy_id}: no skeleton at {pr.known_skeleton_process} (stale reference)")
        return False, " → ".join(route_text + ["✖ no skeleton"]), edges

    # Follow forwarding pointers (short chain)
    visited: Set[str] = set()
    current = target
    k = 0
    while current.forwards_to and current.forwards_to not in visited and k < 8:
        visited.add(current.host_process)
        edges.append((current.host_process, current.forwards_to))
        route_text.append(f"{current.host_process} → {current.forwards_to} (forward)")
        # step to skeleton at forwards_to if exists
        nxt = None
        for s in w.skeletons.values():
            if s.host_process == current.forwards_to:
                nxt = s
                break
        if nxt is None:
            break
        current = nxt
        k += 1

    # Final service hop to where object lives (conceptual)
    if current.host_process != w.object_process:
        edges.append((current.host_process, w.object_process))
        route_text.append(f"{current.host_process} → {w.object_process} (serve)")

    # Shortcut after first success: update proxy to known skeleton at object (if any), else current
    if do_shortcut:
        new_known = w.object_process if f"S@{w.object_process}" in w.skeletons else current.host_process
        pr.known_skeleton_process = new_known
        w.log_event(f"{proxy_id} learned skeleton location: {new_known} (shortcut).")

    return True, " → ".join(route_text + [f"→ {w.object_process} (object)"]), edges


def gc_skeletons(w: World):
    # Remove skeletons not at object host, not forwarding, and not targeted by any proxy
    to_rm = []
    for sid, s in w.skeletons.items():
        if s.host_process == w.object_process:
            continue
        if s.forwards_to is not None:
            continue
        if any(p.known_skeleton_process == s.host_process for p in w.proxies.values()):
            continue
        to_rm.append(sid)
    for sid in to_rm:
        host = w.skeletons[sid].host_process
        if sid in w.processes[host].skeletons:
            w.processes[host].skeletons.remove(sid)
        del w.skeletons[sid]
        w.log_event(f"GC: removed skeleton at {host}.")

# -------------------------------
# Visualization
# -------------------------------

def draw_world(w: World, highlight_edges: Optional[List[Tuple[str, str]]] = None, highlight_nodes: Optional[Set[str]] = None, title: str = ""):
    highlight_edges = highlight_edges or []
    highlight_nodes = highlight_nodes or set()

    G = nx.DiGraph()
    for pname, proc in w.processes.items():
        label = f"{pname}\n{proc.label}"
        if proc.holds_object:
            label += "\n[OBJECT]"
        if proc.proxies:
            label += "\nProxies: " + ", ".join(proc.proxies)
        if proc.skeletons:
            label += f"\nSkeletons: {len(proc.skeletons)}"
        G.add_node(pname, label=label)

    # Proxy edges (client → skeleton host)
    for pid, pr in w.proxies.items():
        G.add_edge(pr.owner_process, pr.known_skeleton_process, kind="proxy", label=pid)

    # Forwarding pointer edges (skeleton host → next)
    for s in w.skeletons.values():
        if s.forwards_to:
            G.add_edge(s.host_process, s.forwards_to, kind="forward", label="forward")

    # Service edges (skeleton host → object)
    for s in w.skeletons.values():
        G.add_edge(s.host_process, w.object_process, kind="serve", label="serve")

    pos = nx.spring_layout(G, seed=42, k=1.1)

    fig = plt.figure(figsize=(9.5, 6.8))

    # Nodes
    node_colors = ["#ffd166" if n in highlight_nodes else "#a8dadc" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=2100, node_color=node_colors, edgecolors="#1d3557", linewidths=1.4)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes})

    # Helper to draw each edge type
    def draw_kind(kind: str, style: str, width: float, alpha: float):
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == kind]
        widths = [3.0 if (u, v) in (highlight_edges or []) else width for (u, v) in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True, style=style, width=widths, alpha=alpha)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G.edges[u, v]['label'] for (u, v) in edges})

    draw_kind("proxy", style="solid", width=2.0, alpha=0.95)
    draw_kind("forward", style="dashed", width=2.0, alpha=0.95)
    draw_kind("serve", style="dotted", width=1.6, alpha=0.75)

    plt.title(title)
    plt.axis('off')
    st.pyplot(fig, width='stretch')

# -------------------------------
# UI — Two tabs: Guided Steps • Interactive Playground
# -------------------------------
GUIDE_STEPS = [
    (1, "Actors"),
    (2, "Login (Identical Proxies)"),
    (3, "Balance (Skeleton Dispatch)"),
    (4, "Withdraw (IPC Path)"),
    (5, "Scale-out (Identical Skeletons)"),
    (6, "Migration (Forwarding Pointer)"),
    (7, "Shortcutting & Cleanup"),
]

if 'world_bank' not in st.session_state:
    st.session_state.world_bank = init_banking_world()
if 'guide_step_idx' not in st.session_state:
    st.session_state.guide_step_idx = 1
if 'guide_is_playing' not in st.session_state:
    st.session_state.guide_is_playing = False

_tabs = st.tabs(["🎓 Guided Steps", "🧪 Playground"])

# -------------------------------
# Tab 1 — Guided Steps
# -------------------------------
with _tabs[0]:
    left, right = st.columns([3, 2])

    with right:
        st.subheader("Step Controls")
        st.markdown("**Steps**: " + " → ".join([f"{i}. {name}" for i, name in GUIDE_STEPS]))
        _g_val = st.slider("Step", 1, len(GUIDE_STEPS), value=st.session_state.guide_step_idx, key="guide_step_ui")
        if not st.session_state.guide_is_playing:
            st.session_state.guide_step_idx = _g_val

        spd = st.select_slider("Animation speed", options=["slow", "normal", "fast"], value="normal", key="anim_speed_bank_guide")
        delay = {"slow": 1.2, "normal": 0.8, "fast": 0.4}[spd]

        col = st.columns(3)
        with col[0]:
            if st.button("◀ Prev", key="guide_prev"):
                st.session_state.guide_step_idx = max(1, st.session_state.guide_step_idx - 1)
        with col[1]:
            if st.button("▶ Play" if not st.session_state.guide_is_playing else "⏸ Pause", key="guide_play"):
                st.session_state.guide_is_playing = not st.session_state.guide_is_playing
        with col[2]:
            if st.button("Next ▶", key="guide_next"):
                st.session_state.guide_step_idx = min(len(GUIDE_STEPS), st.session_state.guide_step_idx + 1)

    # Build highlighted path for current step
    w = init_banking_world()  # fresh world for clean narration
    hi_edges: List[Tuple[str, str]] = []
    caption = ""

    if st.session_state.guide_step_idx == 1:
        caption = "**Actors:** P1=ATM, P2=MobileApp, P3=API Gateway (skeleton host), P4=Account Service (object)."
    elif st.session_state.guide_step_idx == 2:
        caption = "**Login (Identical Proxies):** ATM & MobileApp hold identical proxies to the same Gateway skeleton."
        hi_edges = [("P1", "P3"), ("P2", "P3")]
    elif st.session_state.guide_step_idx == 3:
        caption = "**Balance:** Gateway's skeleton unmarshals request and invokes Account Service (object) at P4."
        hi_edges = [("P3", "P4")]
    elif st.session_state.guide_step_idx == 4:
        caption = "**Withdraw (IPC Path):** Full path ATM→Gateway→Service."
        hi_edges = [("P1", "P3"), ("P3", "P4")]
    elif st.session_state.guide_step_idx == 5:
        # Add a second identical skeleton at P3
        sid2 = "S2@P3"
        if sid2 not in w.skeletons:
            w.skeletons[sid2] = SkeletonRef(id=sid2, host_process="P3", forwards_to="P4")
            w.processes["P3"].skeletons.append(sid2)
        caption = "**Scale-out:** multiple identical skeletons at the Gateway for throughput & resilience."
        hi_edges = [("P1", "P3"), ("P2", "P3")]
    elif st.session_state.guide_step_idx == 6:
        # Migrate account service to P2 (simulate new cluster node) and keep forwarding
        move_object(w, dst_process="P2", create_new_skeleton=True, keep_forwarding=True)
        caption = "**Migration:** Account Service moves to P2; old Gateway skeleton forwards to new location (forwarding pointer)."
        hi_edges = [("P3", "P2")]
    elif st.session_state.guide_step_idx == 7:
        # After first contact, clients learn the new location (shortcut)
        for pid in list(w.proxies.keys()):
            invoke(w, pid, do_shortcut=True)
        caption = "**Shortcutting & Cleanup:** Proxies update to new skeleton; old skeletons can be GC'ed when unused."
        hi_edges = [("P1", "P2"), ("P2", "P2")]

    with left:
        draw_world(w, highlight_edges=hi_edges, title=f"Step {st.session_state.guide_step_idx}: {GUIDE_STEPS[st.session_state.guide_step_idx-1][1]}")

    with right:
        st.subheader("Explanation")
        st.markdown(caption)
        st.caption("Edges highlight the current flow (proxy IPC, skeleton forwarding, or service call).")

    if st.session_state.guide_is_playing:
        time.sleep(delay)
        st.session_state.guide_step_idx = 1 + (st.session_state.guide_step_idx % len(GUIDE_STEPS))
        st.experimental_rerun()

# -------------------------------
# Tab 2 — Interactive Playground
# -------------------------------
with _tabs[1]:
    if 'play_world' not in st.session_state:
        st.session_state.play_world = init_banking_world()

    pw: World = st.session_state.play_world

    with st.sidebar:
        st.title("Playground Controls")
        st.markdown("**Concepts**\n- **Proxy**: client stub at ATM/MobileApp\n- **Skeleton**: API Gateway dispatcher\n- **Object**: Account Service implementation\n- **IPC**: cross-process network hop\n- **Forwarding**: old skeleton forwards after migration\n- **Shortcut**: client learns new location after first contact")

        st.markdown("---")
        if st.button("Reset world", key="play_reset"):
            st.session_state.play_world = init_banking_world()
            pw = st.session_state.play_world

        st.subheader("Move Account Service")
        dst = st.selectbox("Move to process:", ["P1", "P2", "P3", "P4"], index=1, key="play_dst")
        create_new_skel = st.checkbox("Create NEW skeleton at destination", value=True, key="play_new_skel")
        keep_fwd = st.checkbox("Keep forwarding pointers", value=True, key="play_keep_fwd")
        if st.button("Move object", key="play_move_obj"):
            move_object(pw, dst, create_new_skeleton=create_new_skel, keep_forwarding=keep_fwd)

        st.subheader("Invoke via Client")
        client = st.selectbox("Choose client proxy:", ["px_atm", "px_mob"], index=0, key="play_client")
        do_short = st.checkbox("Shortcut after success", value=True, key="play_short")
        if st.button("Invoke", key="play_invoke"):
            ok, route, edges = invoke(pw, client, do_shortcut=do_short)
            pw.log_event(("SUCCESS" if ok else "FAIL") + f" — route: {route}")

        if st.button("Garbage-collect skeletons", key="play_gc"):
            gc_skeletons(pw)

    left2, right2 = st.columns([3, 2])
    with left2:
        draw_world(pw, title="Playground State")
    with right2:
        st.subheader("Event Log")
        if pw.log:
            for line in pw.log[-30:][::-1]:
                st.markdown(f"• {line}")
        else:
            st.info("No events yet. Try moving the object or invoking via a proxy.")
