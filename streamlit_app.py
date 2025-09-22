# streamlit_app.py
# Educational simulator + animated tutorial for Forwarding Pointers, Proxy, Skeleton,
# Identical Proxy, Interprocess Communication (IPC), and Local Invocation.
# Run locally:  streamlit run streamlit_app.py

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Set

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Forwarding Pointers • Proxy • Skeleton — Animated Tutorial & Simulator",
    layout="wide",
)

# -------------------------------
# Data Model
# -------------------------------
@dataclass
class Process:
    name: str
    proxies: List[str] = field(default_factory=list)   # proxy ids living in this process
    skeletons: List[str] = field(default_factory=list) # skeleton ids living in this process
    holds_object: bool = False                        # whether the real object is here

@dataclass
class ProxyRef:
    id: str
    owner_process: str
    # The proxy believes the object's skeleton is at this process
    known_skeleton_process: str

@dataclass
class SkeletonRef:
    id: str
    host_process: str
    forwards_to: Optional[str] = None  # process name of next skeleton (forwarding pointer)

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
# World Constructors
# -------------------------------

def fresh_processes(n: int = 4) -> Dict[str, Process]:
    return {f"P{i}": Process(name=f"P{i}") for i in range(1, n + 1)}


def init_default_world(identical_proxy_count: int = 2) -> World:
    processes = fresh_processes()

    # Place real object at P4
    processes["P4"].holds_object = True

    # Place a skeleton at P3 that serves the object in P4
    skel_main = SkeletonRef(id="S@P3", host_process="P3", forwards_to="P4")
    processes["P3"].skeletons.append(skel_main.id)

    proxies: Dict[str, ProxyRef] = {}
    # Create N identical proxies (one per client process starting from P1)
    for i in range(identical_proxy_count):
        proc_name = f"P{i+1}"
        pid = f"p{i+1}"
        proxies[pid] = ProxyRef(id=pid, owner_process=proc_name, known_skeleton_process="P3")
        processes[proc_name].proxies.append(pid)

    world = World(
        processes=processes,
        proxies=proxies,
        skeletons={skel_main.id: skel_main},
        object_process="P4",
    )
    world.log_event("Initialized world: object at P4, skeleton at P3 (forward→P4), identical proxies at P1 and P2.")
    return world

# -------------------------------
# Simulator Actions
# -------------------------------

def move_object(world: World, dst_process: str, create_new_skeleton: bool, keep_forwarding: bool):
    if dst_process == world.object_process:
        world.log_event(f"Object already at {dst_process} — no move performed.")
        return

    src = world.object_process
    world.processes[src].holds_object = False
    world.object_process = dst_process
    world.processes[dst_process].holds_object = True

    # Optionally create a new skeleton at the destination
    new_skel_id = f"S@{dst_process}"
    if create_new_skeleton and new_skel_id not in world.skeletons:
        world.skeletons[new_skel_id] = SkeletonRef(id=new_skel_id, host_process=dst_process)
        world.processes[dst_process].skeletons.append(new_skel_id)
        world.log_event(f"Object moved {src} → {dst_process}. New skeleton created at {dst_process}.")

    # Manage forwarding pointers
    if keep_forwarding:
        for s in world.skeletons.values():
            if s.host_process != dst_process:
                s.forwards_to = dst_process
        world.log_event(f"Established forwarding pointers from old skeletons → {dst_process}.")
    else:
        for s in world.skeletons.values():
            if s.host_process != dst_process:
                s.forwards_to = None
        world.log_event("No forwarding pointers kept; old skeletons do not forward.")


def invoke(world: World, proxy_id: str, shortcut_after_first: bool = True) -> Tuple[bool, str, List[Tuple[str, str]]]:
    """Simulate an invocation from a proxy.
    Returns (success, route_str, edge_route) and updates the proxy's known location if shortcutting.
    """
    if proxy_id not in world.proxies:
        return False, f"Proxy {proxy_id} not found", []

    pr = world.proxies[proxy_id]
    route_labels: List[str] = []
    edge_route: List[Tuple[str, str]] = []

    # Client → known skeleton host (IPC unless same process)
    edge_route.append((pr.owner_process, pr.known_skeleton_process))
    link_label = "LOCAL" if pr.owner_process == pr.known_skeleton_process else "IPC"
    route_labels.append(f"{pr.owner_process} → {pr.known_skeleton_process} ({link_label})")

    # Find an existing skeleton at the known process
    target_skel = None
    for s in world.skeletons.values():
        if s.host_process == pr.known_skeleton_process:
            target_skel = s
            break

    if target_skel is None:
        msg = f"Proxy {proxy_id} tried {pr.known_skeleton_process} — no skeleton there (stale reference)."
        world.log_event(msg)
        return False, " → ".join(route_labels + ["✖ no skeleton"]), edge_route

    # Follow forwarding pointers (if any)
    visited: Set[str] = set()
    current = target_skel
    hops = 0
    while current.forwards_to and current.forwards_to not in visited and hops < 12:
        visited.add(current.host_process)
        edge_route.append((current.host_process, current.forwards_to))
        route_labels.append(f"{current.host_process} → {current.forwards_to} (forward)")
        # Move to skeleton at forwards_to, if any
        next_skel = None
        for s in world.skeletons.values():
            if s.host_process == current.forwards_to:
                next_skel = s
                break
        if next_skel is None:
            # No explicit skeleton object; break and assume service reaches the object there
            break
        current = next_skel
        hops += 1

    # Final dispatch to the object process (conceptual)
    if current.host_process != world.object_process:
        edge_route.append((current.host_process, world.object_process))
        route_labels.append(f"{current.host_process} → {world.object_process} (serve)")

    # Success; optionally apply shortcutting
    if shortcut_after_first:
        # Prefer a real skeleton at the object's process if present; else keep current
        new_known = world.object_process if f"S@{world.object_process}" in world.skeletons else current.host_process
        pr.known_skeleton_process = new_known
        world.log_event(f"{proxy_id} learned skeleton location: {new_known} (shortcut applied).")

    return True, " → ".join(route_labels + [f"→ {world.object_process} (object)"]), edge_route


def garbage_collect_old_skeletons(world: World):
    to_remove = []
    for sid, s in world.skeletons.items():
        still_needed = (s.host_process == world.object_process) or (s.forwards_to is not None)
        # Also keep if any proxy still points to this skeleton host
        if not still_needed and not any(p.known_skeleton_process == s.host_process for p in world.proxies.values()):
            to_remove.append(sid)
    for sid in to_remove:
        host = world.skeletons[sid].host_process
        if sid in world.processes[host].skeletons:
            world.processes[host].skeletons.remove(sid)
        del world.skeletons[sid]
        world.log_event(f"Garbage collected skeleton at {host}.")

# -------------------------------
# Visualization
# -------------------------------

def draw_world(world: World, highlight_nodes: Optional[Set[str]] = None, highlight_edges: Optional[List[Tuple[str, str]]] = None, title: Optional[str] = None):
    highlight_nodes = highlight_nodes or set()
    highlight_edges = highlight_edges or []

    G = nx.DiGraph()

    # Add process nodes
    for pname, proc in world.processes.items():
        label = pname
        if proc.holds_object:
            label += "[OBJECT]"
        if proc.proxies:
            label += "Proxies: " + ", ".join(proc.proxies)
        if proc.skeletons:
            # Show only count to keep nodes compact; details are in log/legend
            label += f"Skeletons: {len(proc.skeletons)}"
        G.add_node(pname, label=label)

    # Edges: proxies to their known skeleton process
    for pid, pr in world.proxies.items():
        G.add_edge(pr.owner_process, pr.known_skeleton_process, kind="proxy", label=pid)

    # Forwarding pointers
    for s in world.skeletons.values():
        if s.forwards_to:
            G.add_edge(s.host_process, s.forwards_to, kind="forward", label="forward")

    # Service edges (skeleton host → object process)
    for s in world.skeletons.values():
        G.add_edge(s.host_process, world.object_process, kind="serve", label="serve")

    pos = nx.spring_layout(G, seed=7, k=1.2)

    fig = plt.figure(figsize=(9.5, 6.8))

    # Nodes
    node_colors = ["#ffd166" if n in highlight_nodes else "#a8dadc" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, edgecolors="#1d3557", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes})

    # Edges by kind
    def draw_edges(kind: str, style: str, width: float, alpha: float):
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == kind]
        # Thicken if highlighted
        widths = [3.0 if (u, v) in highlight_edges else width for (u, v) in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True, style=style, width=widths, alpha=alpha)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G.edges[u, v]['label'] for (u, v) in edges})

    draw_edges("proxy", style="solid", width=2.0, alpha=0.9)
    draw_edges("forward", style="dashed", width=2.0, alpha=0.9)
    draw_edges("serve", style="dotted", width=1.6, alpha=0.7)

    plt.title(title or "")
    plt.axis('off')
    st.pyplot(fig, use_container_width=True)

# -------------------------------
# Tutorial (Animated)
# -------------------------------

def make_world_for_step(step: int) -> Tuple[World, Set[str], List[Tuple[str, str]], str]:
    """Return (world, highlight_nodes, highlight_edges, caption) for the tutorial step."""
    P = fresh_processes()
    proxies: Dict[str, ProxyRef] = {}
    skeletons: Dict[str, SkeletonRef] = {}
    obj_at = ""
    caption = ""
    hl_nodes: Set[str] = set()
    hl_edges: List[Tuple[str, str]] = []

    def W() -> World:
        return World(processes=P, proxies=proxies, skeletons=skeletons, object_process=obj_at or "P1")

    # Steps definition
    if step == 1:
        obj_at = "P3"  # just to initialize world
        w = W()
        caption = "**Process** = an executing program with its own memory space. Here we show four processes P1..P4."
        hl_nodes = {"P1", "P2", "P3", "P4"}
        return w, hl_nodes, hl_edges, caption

    if step == 2:
        obj_at = "P3"
        P[obj_at].holds_object = True
        skeletons["S@P3"] = SkeletonRef(id="S@P3", host_process="P3")
        P["P3"].skeletons.append("S@P3")
        w = W()
        caption = "**Object** lives in P3. **Skeleton** in P3 exposes its methods to the network."
        hl_nodes = {"P3"}
        return w, hl_nodes, hl_edges, caption

    if step == 3:
        obj_at = "P3"
        P[obj_at].holds_object = True
        skeletons["S@P3"] = SkeletonRef(id="S@P3", host_process="P3")
        P["P3"].skeletons.append("S@P3")
        proxies["p1"] = ProxyRef(id="p1", owner_process="P1", known_skeleton_process="P3")
        P["P1"].proxies.append("p1")
        w = W()
        caption = "**Proxy** (client stub) at P1 looks like the object but forwards calls to P3's skeleton."
        hl_nodes = {"P1", "P3"}
        hl_edges = [("P1", "P3")]
        return w, hl_nodes, hl_edges, caption

    if step == 4:
        obj_at = "P3"
        P[obj_at].holds_object = True
        skeletons["S@P3"] = SkeletonRef(id="S@P3", host_process="P3")
        P["P3"].skeletons.append("S@P3")
        proxies["p1"] = ProxyRef(id="p1", owner_process="P1", known_skeleton_process="P3")
        P["P1"].proxies.append("p1")
        w = W()
        caption = "**Interprocess Communication (IPC)**: P1 → P3 across processes to reach the skeleton."
        hl_edges = [("P1", "P3")]
        return w, set(), hl_edges, caption

    if step == 5:
        obj_at = "P3"
        P[obj_at].holds_object = True
        skeletons["S@P3"] = SkeletonRef(id="S@P3", host_process="P3")
        P["P3"].skeletons.append("S@P3")
        # Local proxy in same process P3
        proxies["p_local"] = ProxyRef(id="p_local", owner_process="P3", known_skeleton_process="P3")
        P["P3"].proxies.append("p_local")
        w = W()
        caption = "**Local Invocation**: proxy and skeleton in the same process (P3) — no network hop."
        hl_edges = [("P3", "P3")]
        return w, {"P3"}, hl_edges, caption

    if step == 6:
        obj_at = "P3"
        P[obj_at].holds_object = True
        skeletons["S@P3"] = SkeletonRef(id="S@P3", host_process="P3")
        P["P3"].skeletons.append("S@P3")
        # Identical proxies at P1 and P2
        proxies["p1"] = ProxyRef(id="p1", owner_process="P1", known_skeleton_process="P3")
        proxies["p2"] = ProxyRef(id="p2", owner_process="P2", known_skeleton_process="P3")
        P["P1"].proxies.append("p1")
        P["P2"].proxies.append("p2")
        w = W()
        caption = "**Identical Proxies**: P1:p1 and P2:p2 behave the same and talk to the same skeleton."
        hl_edges = [("P1", "P3"), ("P2", "P3")]
        return w, set(), hl_edges, caption

    if step == 7:
        obj_at = "P3"
        P[obj_at].holds_object = True
        # Two identical skeletons (e.g., replicated dispatcher) in P3
        skeletons["S@P3"] = SkeletonRef(id="S@P3", host_process="P3")
        skeletons["S2@P3"] = SkeletonRef(id="S2@P3", host_process="P3")
        P["P3"].skeletons.extend(["S@P3", "S2@P3"])
        # One proxy at P1
        proxies["p1"] = ProxyRef(id="p1", owner_process="P1", known_skeleton_process="P3")
        P["P1"].proxies.append("p1")
        w = W()
        caption = "**Identical Skeletons**: multiple server-side dispatchers expose the same object API."
        hl_nodes = {"P3"}
        hl_edges = [("P1", "P3")]
        return w, hl_nodes, hl_edges, caption

    if step == 8:
        # Object moves P3 → P4; keep old skeleton with forward pointer
        obj_at = "P4"
        P[obj_at].holds_object = True
        skeletons["S@P3"] = SkeletonRef(id="S@P3", host_process="P3", forwards_to="P4")
        P["P3"].skeletons.append("S@P3")
        # New skeleton at P4
        skeletons["S@P4"] = SkeletonRef(id="S@P4", host_process="P4")
        P["P4"].skeletons.append("S@P4")
        # Proxy still thinks skeleton is at P3
        proxies["p1"] = ProxyRef(id="p1", owner_process="P1", known_skeleton_process="P3")
        P["P1"].proxies.append("p1")
        w = W()
        caption = "**Forwarding Pointer**: object moved to P4. Old skeleton at P3 forwards to P4."
        hl_edges = [("P3", "P4")]
        return w, {"P3", "P4"}, hl_edges, caption

    if step == 9:
        # Same as step 8 but show an invocation path through forward pointer
        obj_at = "P4"
        P[obj_at].holds_object = True
        skeletons["S@P3"] = SkeletonRef(id="S@P3", host_process="P3", forwards_to="P4")
        P["P3"].skeletons.append("S@P3")
        skeletons["S@P4"] = SkeletonRef(id="S@P4", host_process="P4")
        P["P4"].skeletons.append("S@P4")
        proxies["p1"] = ProxyRef(id="p1", owner_process="P1", known_skeleton_process="P3")
        P["P1"].proxies.append("p1")
        w = W()
        # Compute route for highlight
        _, _, edge_route = invoke_preview(w, "p1")
        caption = "**Invocation via Forwarding**: P1→P3 (IPC), then forward P3→P4, then serve to object."
        return w, set(), edge_route, caption

    if step == 10:
        # After one call, proxy learns the new skeleton location (shortcut)
        obj_at = "P4"
        P[obj_at].holds_object = True
        skeletons["S@P4"] = SkeletonRef(id="S@P4", host_process="P4")
        P["P4"].skeletons.append("S@P4")
        proxies["p1"] = ProxyRef(id="p1", owner_process="P1", known_skeleton_process="P4")
        P["P1"].proxies.append("p1")
        w = W()
        caption = "**Shortcutting**: proxy updated → direct P1→P4 (IPC). Old skeletons can be garbage-collected."
        hl_edges = [("P1", "P4")]
        return w, set(), hl_edges, caption

    # Fallback — show empty world
    obj_at = "P1"
    return W(), set(), [], ""


def invoke_preview(world: World, proxy_id: str) -> Tuple[bool, str, List[Tuple[str, str]]]:
    # helper used by the tutorial to compute a path without mutating proxy state
    pr = world.proxies[proxy_id]
    tmp = ProxyRef(id=pr.id, owner_process=pr.owner_process, known_skeleton_process=pr.known_skeleton_process)
    world.proxies[proxy_id] = tmp
    ok, route, edge_route = invoke(world, proxy_id, shortcut_after_first=False)
    return ok, route, edge_route

# -------------------------------
# Sidebar — Shared Legend
# -------------------------------
with st.sidebar:
    st.title("Concepts Cheat‑Sheet")
    st.markdown(
        """
        - **Process**: running program with its own memory.
        - **Object**: the thing we invoke methods on (lives inside a process).
        - **Proxy** (client stub): local stand‑in that marshals calls to a remote skeleton.
        - **Skeleton** (server stub): receives requests, unmarshals, invokes the object.
        - **Interprocess Communication (IPC)**: network hop between processes.
        - **Local Invocation**: proxy and skeleton in same process — no network.
        - **Identical Proxies**: multiple proxies for the same object in different clients.
        - **Identical Skeletons**: replicated server dispatchers exposing the same API.
        - **Forwarding Pointer**: old skeleton forwards to the object's new location.
        - **Shortcutting**: after first call via forwarding, proxy learns the new location.
        """
    )

# -------------------------------
# Banking Example Helpers
# -------------------------------

def init_banking_world() -> World:
    """Map P1..P4 to banking roles for narration.
    P1=ATM, P2=MobileApp, P3=API Gateway (skeleton), P4=Account Service (object).
    """
    w = init_default_world(identical_proxy_count=2)
    # Ensure skeleton at P3 forwards to object at P4 initially
    for s in w.skeletons.values():
        if s.host_process == "P3":
            s.forwards_to = "P4"
    # Overwrite event log with role labels
    w.log = []
    w.log_event("Banking world: ATM(P1), MobileApp(P2), API Gateway(P3), Account Service(P4).")
    return w


def banking_step(step: int) -> Tuple[World, str, List[Tuple[str, str]]]:
    """Return (world, caption, highlight_edges) for the banking tutorial step."""
    w = init_banking_world()
    hilite: List[Tuple[str, str]] = []

    if step == 1:
        cap = "**Actors**: P1=ATM, P2=MobileApp, P3=API Gateway (skeleton host), P4=Account Service (object)."
        return w, cap, hilite

    if step == 2:
        cap = "**Login**: ATM & MobileApp hold **identical proxies** to call the API Gateway at P3."
        hilite = [("P1", "P3"), ("P2", "P3")]
        return w, cap, hilite

    if step == 3:
        cap = "**Balance**: Gateway's **skeleton** unmarshals request and invokes **Account Service** at P4."
        hilite = [("P3", "P4")]
        return w, cap, hilite

    if step == 4:
        cap = "**Withdraw**: Full IPC path ATM→Gateway→Service."
        hilite = [("P1", "P3"), ("P3", "P4")]
        return w, cap, hilite

    if step == 5:
        # add an identical skeleton (replica) at P3 logically
        if "S2@P3" not in w.skeletons:
            w.skeletons["S2@P3"] = SkeletonRef(id="S2@P3", host_process="P3", forwards_to="P4")
            w.processes["P3"].skeletons.append("S2@P3")
        cap = "**Scale‑out**: multiple **identical skeletons** at Gateway for throughput & resilience."
        hilite = [("P1", "P3"), ("P2", "P3")]
        return w, cap, hilite

    if step == 6:
        # migrate service to P2 (simulate cluster move)
        move_object(w, dst_process="P2", create_new_skeleton=True, keep_forwarding=True)
        cap = "**Migration**: Account Service moves to P2; old skeletons at P3 install **forwarding pointers** → P2."
        # show forwarding edge
        hilite = [("P3", "P2")]
        return w, cap, hilite

    if step == 7:
        # after first call, proxies learn new location (shortcut)
        for pid in list(w.proxies.keys()):
            invoke(w, pid, shortcut_after_first=True)
        cap = "**Shortcutting**: Clients learn new location; future ATM/MobileApp calls go **direct to P2** via updated proxies."
        hilite = [("P1", "P2"), ("P2", "P2")]
        return w, cap, hilite

    return w, "", hilite

# -------------------------------
# Tabs: Tutorial (Animated) • Simulator (Playground) • Banking Example
# -------------------------------


def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

TUTORIAL_STEPS = [
    (1, "Processes"),
    (2, "Object & Skeleton"),
    (3, "Proxy"),
    (4, "IPC (Remote Call)"),
    (5, "Local Invocation"),
    (6, "Identical Proxies"),
    (7, "Identical Skeletons"),
    (8, "Object Migration & Forwarding"),
    (9, "Invocation via Forwarding"),
    (10, "Shortcutting & Cleanup"),
]

# Banking example steps (ATM/MobileApp → API Gateway → Account Service)
BANKING_STEPS = [
    (1, "Actors"),
    (2, "Login (Proxies)"),
    (3, "Balance (Skeleton Dispatch)"),
    (4, "Withdraw (IPC Path)"),
    (5, "Scale-out (Identical Skeletons)"),
    (6, "Migration (Forwarding Pointer)"),
    (7, "Shortcutting to New Node"),
]

if 'world' not in st.session_state:
    st.session_state.world = init_default_world(identical_proxy_count=2)
if 't_step' not in st.session_state:
    st.session_state.t_step = 1
if 't_play' not in st.session_state:
    st.session_state.t_play = False

st.title("Forwarding Pointers • Proxy • Skeleton — Animated Tutorial & Simulator")

_tabs = st.tabs(["🎓 Tutorial (Animated)", "🧪 Playground Simulator", "🏦 Banking Example"])

# -------------------------------
# Tab 1 — Animated Tutorial
# -------------------------------
with _tabs[0]:
    left, right = st.columns([3, 2])
    with right:
        st.subheader("Step Controls")
        st.write("Follow each step to see definitions and a small animation.")
        st.markdown("**Steps**: " + " → ".join([f"{i}. {name}" for i, name in TUTORIAL_STEPS]))
        st.slider("Tutorial Step", min_value=1, max_value=len(TUTORIAL_STEPS), key="t_step")
        speed = st.select_slider("Animation speed", options=["slow", "normal", "fast"], value="normal")
        delay = {"slow": 1.2, "normal": 0.8, "fast": 0.4}[speed]

        colp = st.columns(3)
        with colp[0]:
            if st.button("◀ Prev"):
                st.session_state.t_step = max(1, st.session_state.t_step - 1)
        with colp[1]:
            if st.button("▶ Play" if not st.session_state.t_play else "⏸ Pause"):
                st.session_state.t_play = not st.session_state.t_play
        with colp[2]:
            if st.button("Next ▶"):
                st.session_state.t_step = min(len(TUTORIAL_STEPS), st.session_state.t_step + 1)

    # Build world for this step
    world_t, hl_nodes, hl_edges, caption = make_world_for_step(st.session_state.t_step)

    with left:
        draw_world(world_t, highlight_nodes=hl_nodes, highlight_edges=hl_edges, title=f"Step {st.session_state.t_step}: {TUTORIAL_STEPS[st.session_state.t_step-1][1]}")

    with right:
        st.subheader("Explanation")
        st.markdown(caption)
        st.caption("Highlighted nodes/edges correspond to the current concept or invocation path.")

    # Auto-advance (animation)
    if st.session_state.t_play:
        time.sleep(delay)
        st.session_state.t_step = 1 + (st.session_state.t_step % len(TUTORIAL_STEPS))
        safe_rerun()

# -------------------------------
# Tab 2 — Interactive Playground Simulator
# -------------------------------
with _tabs[1]:
    world: World = st.session_state.world

    with st.sidebar:
        st.markdown("---")
        st.subheader("Playground Controls")

        n = st.slider("Number of identical proxies (clients)", 1, 4, 2)
        if st.button("Reset world"):
            st.session_state.world = init_default_world(identical_proxy_count=n)
            world = st.session_state.world

        st.markdown("---")
        st.subheader("Move Object (Migration)")
        dst = st.selectbox("Move object to:", options=list(world.processes.keys()), index=3)
        create_new_skel = st.checkbox("Create NEW skeleton at destination", value=True)
        keep_fwd = st.checkbox("Keep forwarding pointers from old skeletons", value=True)
        if st.button("Move object"):
            move_object(world, dst, create_new_skeleton=create_new_skel, keep_forwarding=keep_fwd)

        st.markdown("---")
        st.subheader("Invoke")
        colA, colB = st.columns(2)
        with colA:
            choose_proxy = st.selectbox("Choose proxy (client)", options=list(world.proxies.keys()))
        with colB:
            shortcut = st.checkbox("Shortcut after first invocation (proxy learns new location)", value=True)
        if st.button("Invoke via selected proxy"):
            ok, route, edge_route = invoke(world, choose_proxy, shortcut_after_first=shortcut)
            world.log_event(("SUCCESS" if ok else "FAIL") + f" — route: {route}")

        if st.button("Garbage-collect unused skeletons"):
            garbage_collect_old_skeletons(world)

    left2, right2 = st.columns([3, 2])
    with left2:
        draw_world(world, title="Playground State")
    with right2:
        st.subheader("Event Log")
        if world.log:
            for line in world.log[-30:][::-1]:
                st.markdown(f"• {line}")
        else:
            st.info("No events yet. Use the controls to move the object or invoke via a proxy.")

# -------------------------------
# Tab 3 — Banking Example (Real App Mapping)
# -------------------------------
with _tabs[2]:
    st.subheader("Banking — Real App Walkthrough")
    st.caption("P1=ATM, P2=MobileApp, P3=API Gateway (skeleton host), P4=Account Service (object)")

    if 'bank_step' not in st.session_state:
        st.session_state.bank_step = 1
    if 'bank_play' not in st.session_state:
        st.session_state.bank_play = False

    leftB, rightB = st.columns([3, 2])
    with rightB:
        st.markdown("**Steps**: " + " → ".join([f"{i}. {name}" for i, name in BANKING_STEPS]))
        st.slider("Banking Step", 1, len(BANKING_STEPS), key="bank_step")
        speedB = st.select_slider("Animation speed", options=["slow", "normal", "fast"], value="normal")
        delayB = {"slow": 1.2, "normal": 0.8, "fast": 0.4}[speedB]
        colbp = st.columns(3)
        with colbp[0]:
            if st.button("◀ Prev "):
                st.session_state.bank_step = max(1, st.session_state.bank_step - 1)
        with colbp[1]:
            if st.button("▶ Play " if not st.session_state.bank_play else "⏸ Pause"):
                st.session_state.bank_play = not st.session_state.bank_play
        with colbp[2]:
            if st.button("Next ▶ "):
                st.session_state.bank_step = min(len(BANKING_STEPS), st.session_state.bank_step + 1)

    # Build world & highlights for this banking step
    wbank, capB, edgesB = banking_step(st.session_state.bank_step)

    with leftB:
        draw_world(wbank, highlight_edges=edgesB, title=f"Banking Step {st.session_state.bank_step}: {BANKING_STEPS[st.session_state.bank_step-1][1]}")

    with rightB:
        st.subheader("Explanation")
        st.markdown(capB)
        st.caption("Edges highlight the current flow (proxy IPC, skeleton forwarding, or service call).")

    if st.session_state.bank_play:
        time.sleep(delayB)
        st.session_state.bank_step = 1 + (st.session_state.bank_step % len(BANKING_STEPS))
        safe_rerun()

st.markdown("""warding, shortcutting)
- *Forwarding Pointers II–IV (proxies & skeletons)* → Steps 2–7
- *IPC vs Local* → Steps 4–5
""")
