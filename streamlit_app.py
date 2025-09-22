# banking_real_flows_app.py
# Standalone Streamlit app for a real-world BANKING demo with flows
# Demonstrates: Process, Proxy, Skeleton, Identical Proxy/Skeleton, IPC vs Local,
# Forwarding Pointers, Shortcutting, and real flows: Login, Balance, Deposit, Withdraw
# Run:  streamlit run banking_real_flows_app.py

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Set

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏦 Banking Demo — Real Flows • Proxy • Skeleton • Forwarding", layout="wide")

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
    # Banking: simple ledger + users & sessions
    accounts: Dict[str, int] = field(default_factory=lambda: {"A-100": 500, "B-200": 800})
    users: Dict[str, Dict[str, object]] = field(default_factory=lambda: {
        "alice": {"pin": "1234", "accounts": ["A-100"], "require_2fa": False},
        "bob":   {"pin": "4321", "accounts": ["B-200"], "require_2fa": True},
    })
    sessions: Dict[str, Dict[str, object]] = field(default_factory=dict)  # key: client proxy id → {user, 2fa}
    daily_withdraw_limit: int = 500
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
        w.log_event("No forwarding kept; clients must learn new location via discovery.")

# -------------------------------
# Invocation & networking
# -------------------------------
def invoke_route(w: World, proxy_id: str) -> Tuple[bool, str, List[Tuple[str, str]], str]:
    """Compute route from client proxy to object via skeleton/forwarding.
    Returns (success, route_str, edge_list, current_skeleton_host)."""
    if proxy_id not in w.proxies:
        return False, f"Proxy {proxy_id} not found", [], ""

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
        return False, " → ".join(route_text + ["✖ no skeleton"]), edges, pr.known_skeleton_process

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

    return True, " → ".join(route_text + [f"→ {w.object_process} (object)"]), edges, current.host_process

def invoke_and_shortcut(w: World, proxy_id: str) -> Tuple[bool, str, List[Tuple[str, str]]]:
    ok, route, edges, current_skel = invoke_route(w, proxy_id)
    if not ok:
        w.log_event(f"{proxy_id}: route failed — {route}")
        return ok, route, edges
    # Shortcut: update proxy to known skeleton at object (if any), else current
    new_known = w.object_process if f"S@{w.object_process}" in w.skeletons else current_skel
    w.proxies[proxy_id].known_skeleton_process = new_known
    w.log_event(f"{proxy_id} learned skeleton location: {new_known} (shortcut).")
    return ok, route, edges

# -------------------------------
# Banking operations & auth
# -------------------------------
def is_authed(w: World, client_proxy: str) -> bool:
    return client_proxy in w.sessions and w.sessions[client_proxy].get("2fa", False) is True

def login(w: World, client_proxy: str, username: str, pin: str, twofa_passed: bool = False) -> bool:
    u = w.users.get(username)
    if not u:
        w.log_event(f"LOGIN FAIL ({client_proxy}): unknown user '{username}'.")
        return False
    if pin != u["pin"]:
        w.log_event(f"LOGIN FAIL ({client_proxy}): wrong PIN for '{username}'.")
        return False
    if u.get("require_2fa", False) and not twofa_passed:
        w.log_event(f"LOGIN PENDING 2FA ({client_proxy}): '{username}' needs 2FA.")
        w.sessions[client_proxy] = {"user": username, "2fa": False}
        return False
    # success
    w.sessions[client_proxy] = {"user": username, "2fa": True}
    w.log_event(f"LOGIN OK ({client_proxy}): '{username}'.")
    return True

def balance(w: World, username: str) -> List[Tuple[str, int]]:
    accs = w.users.get(username, {}).get("accounts", [])
    return [(a, w.accounts.get(a, 0)) for a in accs]

def deposit(w: World, username: str, account: str, amount: int) -> bool:
    if account not in w.users.get(username, {}).get("accounts", []):
        w.log_event(f"DEPOSIT DENIED: {username} not owner of {account}")
        return False
    w.accounts[account] = w.accounts.get(account, 0) + amount
    w.log_event(f"DEPOSIT: {account} += {amount} → {w.accounts[account]}")
    return True

def withdraw(w: World, username: str, account: str, amount: int) -> bool:
    if account not in w.users.get(username, {}).get("accounts", []):
        w.log_event(f"WITHDRAW DENIED: {username} not owner of {account}")
        return False
    if amount > w.daily_withdraw_limit:
        w.log_event(f"WITHDRAW DENIED: exceeds daily limit {w.daily_withdraw_limit}")
        return False
    bal = w.accounts.get(account, 0)
    if amount > bal:
        w.log_event(f"WITHDRAW FAIL: {account} has {bal}, need {amount}")
        return False
    w.accounts[account] = bal - amount
    w.log_event(f"WITHDRAW: {account} -= {amount} → {w.accounts[account]}")
    return True

# -------------------------------
# Visualization (bigger arrows + legend)
# -------------------------------
def draw_world_big(w: World,
                   highlight_edges: Optional[List[Tuple[str, str]]] = None,
                   highlight_nodes: Optional[Set[str]] = None,
                   title: str = ""):
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
        G.add_edge(pr.owner_process, pr.known_skeleton_process, kind="proxy", label=f"proxy {pid}")

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
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, arrows=True, arrowsize=28,
            connectionstyle="arc3,rad=0.08", style=style, width=widths, alpha=alpha
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G.edges[u, v]['label'] for (u, v) in edges})

    draw_kind("proxy",   style="solid",  width=2.6, alpha=0.95)
    draw_kind("forward", style="dashed", width=2.6, alpha=0.95)
    draw_kind("serve",   style="dotted", width=2.0, alpha=0.75)

    plt.title(title)
    plt.axis('off')
    st.pyplot(fig, width='stretch')

    # Legend / Direction Guide
    st.markdown(
        "> **Arrow Key**  \\\n"
        "> — **solid →** proxy call (client → skeleton)  \\\n"
        "> — **dashed →** forwarding pointer (old skeleton → new location)  \\\n"
        "> — **dotted →** skeleton dispatch to object (skeleton → object)"
    )

# -------------------------------
# UI — Three tabs: Guided • Playground • Flows
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

if 'guide_step_idx_rf' not in st.session_state:
    st.session_state.guide_step_idx_rf = 1
if 'guide_is_playing_rf' not in st.session_state:
    st.session_state.guide_is_playing_rf = False

# World instances per tab
if 'play_world_rf' not in st.session_state:
    st.session_state.play_world_rf = init_banking_world()
if 'flow_world_rf' not in st.session_state:
    st.session_state.flow_world_rf = init_banking_world()

_tabs = st.tabs(["🎓 Guided", "🧪 Playground", "🔁 Flows (Real)"])

# -------------------------------
# Tab 1 — Guided
# -------------------------------
with _tabs[0]:
    left, right = st.columns([3, 2])

    with right:
        st.subheader("Step Controls")
        st.markdown("**Steps**: " + " → ".join([f"{i}. {name}" for i, name in GUIDE_STEPS]))
        _g_val = st.slider("Step", 1, len(GUIDE_STEPS),
                           value=st.session_state.guide_step_idx_rf, key="guide_step_ui_rf")
        if not st.session_state.guide_is_playing_rf:
            st.session_state.guide_step_idx_rf = _g_val

        spd = st.select_slider("Animation speed",
                               options=["slow", "normal", "fast"],
                               value="normal", key="anim_speed_bank_guide_rf")
        delay = {"slow": 1.2, "normal": 0.8, "fast": 0.4}[spd]

        col = st.columns(3)
        with col[0]:
            if st.button("◀ Prev", key="guide_prev_rf"):
                st.session_state.guide_step_idx_rf = max(1, st.session_state.guide_step_idx_rf - 1)
        with col[1]:
            if st.button("▶ Play" if not st.session_state.guide_is_playing_rf else "⏸ Pause", key="guide_play_rf"):
                st.session_state.guide_is_playing_rf = not st.session_state.guide_is_playing_rf
        with col[2]:
            if st.button("Next ▶", key="guide_next_rf"):
                st.session_state.guide_step_idx_rf = min(len(GUIDE_STEPS), st.session_state.guide_step_idx_rf + 1)

    # Build highlighted path for current step
    w_guided = init_banking_world()  # fresh world for clean narration
    hi_edges: List[Tuple[str, str]] = []
    caption = ""

    step = st.session_state.guide_step_idx_rf
    if step == 1:
        caption = "**Actors:** P1=ATM, P2=MobileApp, P3=API Gateway (skeleton host), P4=Account Service (object)."
    elif step == 2:
        caption = "**Login (Identical Proxies):** ATM & MobileApp hold identical proxies to the same Gateway skeleton."
        hi_edges = [("P1", "P3"), ("P2", "P3")]
    elif step == 3:
        caption = "**Balance:** Gateway's skeleton unmarshals request and invokes Account Service (object) at P4."
        hi_edges = [("P3", "P4")]
    elif step == 4:
        caption = "**Withdraw (IPC Path):** Full path ATM→Gateway→Service."
        hi_edges = [("P1", "P3"), ("P3", "P4")]
    elif step == 5:
        # Add a second identical skeleton at P3
        sid2 = "S2@P3"
        if sid2 not in w_guided.skeletons:
            w_guided.skeletons[sid2] = SkeletonRef(id=sid2, host_process="P3", forwards_to="P4")
            w_guided.processes["P3"].skeletons.append(sid2)
        caption = "**Scale-out:** multiple identical skeletons at the Gateway for throughput & resilience."
        hi_edges = [("P1", "P3"), ("P2", "P3")]
    elif step == 6:
        # Migrate account service to P2 and keep forwarding
        move_object(w_guided, dst_process="P2", create_new_skeleton=True, keep_forwarding=True)
        caption = "**Migration:** Account Service moves to P2; old Gateway skeleton forwards to new location (forwarding pointer)."
        hi_edges = [("P3", "P2")]
    elif step == 7:
        # After first contact, clients learn the new location (shortcut)
        for pid in list(w_guided.proxies.keys()):
            invoke_and_shortcut(w_guided, pid)
        caption = "**Shortcutting & Cleanup:** Proxies update to new skeleton; old skeletons can be GC'ed when unused."
        hi_edges = [("P1", "P2"), ("P2", "P2")]

    with left:
        draw_world_big(w_guided, highlight_edges=hi_edges,
                       title=f"Step {step}: {GUIDE_STEPS[step-1][1]}")

    with right:
        st.subheader("Explanation")
        st.markdown(caption)
        st.caption("Edges highlight the current flow (proxy IPC, skeleton forwarding, or service call).")
        st.markdown("---")
        st.subheader("Process Details")
        st.markdown(
            """
            **Proxy process (client side)**
            - Looks like the object locally.
            - **Marshals** parameters and performs a **remote call** to the skeleton.
            - Updates its cached location after the first successful call (**shortcutting**).

            **Skeleton process (server side)**
            - Listens for network requests from proxies.
            - **Unmarshals** parameters and invokes the real object in the object process.
            - If the object moved, uses a **forwarding pointer** to send the request to the new host.

            **Local invocation process**
            - When proxy and skeleton live in the **same process**, the call is local (no network hop).
            - You can simulate this by having a proxy whose `owner_process == known_skeleton_process`.
            """
        )

    if st.session_state.guide_is_playing_rf:
        time.sleep(delay)
        st.session_state.guide_step_idx_rf = 1 + (st.session_state.guide_step_idx_rf % len(GUIDE_STEPS))
        st.rerun()

# -------------------------------
# Tab 2 — Playground
# -------------------------------
with _tabs[1]:
    pw: World = st.session_state.play_world_rf

    with st.sidebar:
        st.title("Playground Controls")
        st.markdown("**Concepts**\n- **Proxy**: client stub at ATM/MobileApp\n- **Skeleton**: API Gateway dispatcher\n- **Object**: Account Service implementation\n- **IPC**: cross-process network hop\n- **Forwarding**: old skeleton forwards after migration\n- **Shortcut**: client learns new location after first contact")

        st.markdown("---")
        if st.button("Reset world", key="play_reset_rf"):
            st.session_state.play_world_rf = init_banking_world()
            pw = st.session_state.play_world_rf

        st.subheader("Move Account Service")
        dst = st.selectbox("Move to process:", ["P1", "P2", "P3", "P4"], index=1, key="play_dst_rf")
        create_new_skel = st.checkbox("Create NEW skeleton at destination", value=True, key="play_new_skel_rf")
        keep_fwd = st.checkbox("Keep forwarding pointers", value=True, key="play_keep_fwd_rf")
        if st.button("Move object", key="play_move_obj_rf"):
            move_object(pw, dst, create_new_skeleton=create_new_skel, keep_forwarding=keep_fwd)

        st.subheader("Invoke via Client")
        client = st.selectbox("Choose client proxy:", ["px_atm", "px_mob"], index=0, key="play_client_rf")
        do_short = st.checkbox("Shortcut after success", value=True, key="play_short_rf")
        if st.button("Invoke", key="play_invoke_rf"):
            if do_short:
                ok, route, edges = invoke_and_shortcut(pw, client)
            else:
                ok, route, edges, _ = invoke_route(pw, client)
            pw.log_event(("SUCCESS" if ok else "FAIL") + f" — route: {route}")

        if st.button("Garbage-collect skeletons", key="play_gc_rf"):
            # Remove skeletons not at object host, not forwarding, and not targeted by any proxy
            to_rm = []
            for sid, s in pw.skeletons.items():
                if s.host_process == pw.object_process:   # keep
                    continue
                if s.forwards_to is not None:             # keep forwarding nodes
                    continue
                if any(p.known_skeleton_process == s.host_process for p in pw.proxies.values()):
                    continue
                to_rm.append(sid)
            for sid in to_rm:
                host = pw.skeletons[sid].host_process
                if sid in pw.processes[host].skeletons:
                    pw.processes[host].skeletons.remove(sid)
                del pw.skeletons[sid]
                pw.log_event(f"GC: removed skeleton at {host}.")

    left2, right2 = st.columns([3, 2])
    with left2:
        ok2, route2, edges2, _ = invoke_route(pw, client)  # for live visualization
        draw_world_big(pw, highlight_edges=edges2, title=f"Playground Route: {route2 if ok2 else 'No route'}")
    with right2:
        st.subheader("Event Log & Accounts")
        if pw.log:
            for line in pw.log[-30:][::-1]:
                st.markdown(f"• {line}")
        else:
            st.info("No events yet. Try moving the object or invoking via a proxy.")
        st.markdown("---")
        st.markdown("**Ledger**")
        for acc, bal in pw.accounts.items():
            st.markdown(f"- {acc}: **{bal}**")

# -------------------------------
# Tab 3 — Flows (Real)
# -------------------------------
with _tabs[2]:
    fw: World = st.session_state.flow_world_rf

    st.subheader("Real Banking Flows (with Auth & Limits)")
    st.caption("Each run traverses proxy → skeleton → (forwarding) → object, then executes the bank op.")

    cols = st.columns(3)
    with cols[0]:
        client = st.selectbox("Client", ["px_atm", "px_mob"], index=0, key="flow_client_rf")
        username = st.selectbox("User", list(fw.users.keys()), index=0, key="flow_user_rf")
    with cols[1]:
        pin = st.text_input("PIN", type="password", value="1234", key="flow_pin_rf")
        twofa = st.checkbox("2FA passed (for users requiring 2FA)", value=False, key="flow_2fa_rf")
    with cols[2]:
        op = st.selectbox("Operation", ["login", "balance", "deposit", "withdraw"], index=0, key="flow_op_rf")
        account = st.selectbox("Account", list(fw.accounts.keys()), index=0, key="flow_acct_rf")
        amount = st.number_input("Amount (for deposit/withdraw)", min_value=1, max_value=10000, value=100, key="flow_amt_rf")

    run_cols = st.columns(2)
    with run_cols[0]:
        if st.button("Run Flow", key="flow_run_rf"):
            # Always compute route & shortcut first to mimic real call path
            ok, route, edges = invoke_and_shortcut(fw, client)
            fw.log_event(("ROUTE OK" if ok else "ROUTE FAIL") + f" — {route}")
            if ok:
                # Execute operation if routed successfully
                if op == "login":
                    login(fw, client, username, pin, twofa_passed=twofa)
                elif op == "balance":
                    if not is_authed(fw, client):
                        fw.log_event("DENIED: not authenticated.")
                    else:
                        rows = balance(fw, fw.sessions[client]["user"])  # user from session
                        for a, b in rows:
                            fw.log_event(f"BALANCE: {a} = {b}")
                elif op == "deposit":
                    if not is_authed(fw, client):
                        fw.log_event("DENIED: not authenticated.")
                    else:
                        deposit(fw, fw.sessions[client]["user"], account, int(amount))
                elif op == "withdraw":
                    if not is_authed(fw, client):
                        fw.log_event("DENIED: not authenticated.")
                    else:
                        withdraw(fw, fw.sessions[client]["user"], account, int(amount))
            else:
                st.warning("Routing failed (no skeleton). Fix topology in Playground and try again.")

    with run_cols[1]:
        st.markdown("**Routing Visualization**")
        ok2, route2, edges2, _ = invoke_route(fw, client)
        draw_world_big(fw, highlight_edges=edges2, title=f"Flow Route: {route2 if ok2 else 'No route'}")

    st.markdown("---")
    st.subheader("Event Log & Ledger")
    if fw.log:
        for line in fw.log[-30:][::-1]:
            st.markdown(f"• {line}")
    else:
        st.info("No events yet. Run a flow above.")
    st.markdown("**Ledger**")
    for acc, bal in fw.accounts.items():
        st.markdown(f"- {acc}: **{bal}**")
