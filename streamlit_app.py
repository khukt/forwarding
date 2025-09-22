# banking_interactive_graph_app.py
# Interactive banking demo where the GRAPH explains the whole story.
# - Hover nodes: role + details (proxy/skeleton/object, sessions, ledger)
# - Hover edges: step explanations (proxy call / forwarding / serve)
# - Run real flows (login/balance/deposit/withdraw): route is highlighted with steps ① ② ③
# - Toggle "Local Invocation": see proxy+skeleton co-located (no network hop)
#
# Run:
#   pip install streamlit pyvis networkx
#   streamlit run banking_interactive_graph_app.py

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Set

import streamlit as st
import networkx as nx
from pyvis.network import Network
from streamlit.components.v1 import html

st.set_page_config(page_title="🏦 Banking — Interactive Graph (Proxy • Skeleton • Forwarding)", layout="wide")

# -------------------------------
# Data model
# -------------------------------
@dataclass
class Process:
    name: str
    label: str
    proxies: List[str] = field(default_factory=list)
    skeletons: List[str] = field(default_factory=list)
    holds_object: bool = False  # whether the Account Service (object) is here

@dataclass
class ProxyRef:
    id: str
    owner_process: str               # which process owns this proxy
    known_skeleton_process: str      # where the client believes the skeleton is

@dataclass
class SkeletonRef:
    id: str
    host_process: str
    forwards_to: Optional[str] = None  # forwarding pointer target (another process)

@dataclass
class World:
    processes: Dict[str, Process]
    proxies: Dict[str, ProxyRef]
    skeletons: Dict[str, SkeletonRef]
    object_process: str
    # Bank state (lives with the object/service)
    accounts: Dict[str, int] = field(default_factory=lambda: {"A-100": 500, "B-200": 800})
    users: Dict[str, Dict[str, object]] = field(default_factory=lambda: {
        "alice": {"pin": "1234", "accounts": ["A-100"], "require_2fa": False},
        "bob":   {"pin": "4321", "accounts": ["B-200"], "require_2fa": True},
    })
    sessions: Dict[str, Dict[str, object]] = field(default_factory=dict)  # key: client proxy id → {"user":..., "2fa":bool}

# -------------------------------
# World setup & helpers
# -------------------------------
def init_world() -> World:
    # P1=ATM, P2=MobileApp, P3=API Gateway (Skeleton Host), P4=Account Service (Object)
    processes = {
        "P1": Process("P1", "ATM (Client)"),
        "P2": Process("P2", "MobileApp (Client)"),
        "P3": Process("P3", "API Gateway (Skeleton Host)"),
        "P4": Process("P4", "Account Service (Object)", holds_object=True),
    }
    skel = SkeletonRef(id="S@P3", host_process="P3", forwards_to="P4")
    processes["P3"].skeletons.append(skel.id)

    proxies = {
        "px_atm": ProxyRef("px_atm", owner_process="P1", known_skeleton_process="P3"),
        "px_mob": ProxyRef("px_mob", owner_process="P2", known_skeleton_process="P3"),
    }
    processes["P1"].proxies.append("px_atm")
    processes["P2"].proxies.append("px_mob")

    return World(processes=processes, proxies=proxies, skeletons={skel.id: skel}, object_process="P4")

def ensure_skeleton(world: World, process: str):
    sid = f"S@{process}"
    if sid not in world.skeletons:
        world.skeletons[sid] = SkeletonRef(id=sid, host_process=process)
        if sid not in world.processes[process].skeletons:
            world.processes[process].skeletons.append(sid)

def move_object(world: World, dst: str, create_new_skeleton=True, keep_forwarding=True):
    src = world.object_process
    if src == dst:
        return
    world.processes[src].holds_object = False
    world.processes[dst].holds_object = True
    world.object_process = dst
    if create_new_skeleton:
        ensure_skeleton(world, dst)
    # forwarding pointers on all other skeletons
    for s in world.skeletons.values():
        if s.host_process != dst:
            s.forwards_to = dst if keep_forwarding else None

# -------------------------------
# Routing (proxy → skeleton → [forwarding]* → object)
# -------------------------------
def compute_route(world: World, proxy_id: str) -> Tuple[bool, List[Tuple[str, str, str]]]:
    """
    Returns (ok, steps) where steps = list of (src, dst, kind)
      kind ∈ {"proxy","forward","serve"}
    """
    if proxy_id not in world.proxies:
        return False, []
    pr = world.proxies[proxy_id]
    steps: List[Tuple[str, str, str]] = []
    # proxy hop
    steps.append((pr.owner_process, pr.known_skeleton_process, "proxy"))

    # identify skeleton at known location
    target = None
    for s in world.skeletons.values():
        if s.host_process == pr.known_skeleton_process:
            target = s
            break
    if not target:
        return False, steps

    # follow forwarding chain
    current = target
    seen: Set[str] = set()
    while current.forwards_to and current.forwards_to not in seen:
        seen.add(current.host_process)
        steps.append((current.host_process, current.forwards_to, "forward"))
        nxt = None
        for s in world.skeletons.values():
            if s.host_process == current.forwards_to:
                nxt = s
                break
        if not nxt:
            break
        current = nxt

    # final service hop to where object actually lives (conceptual)
    if current.host_process != world.object_process:
        steps.append((current.host_process, world.object_process, "serve"))
    else:
        # still record a serve hop to make the object target explicit
        steps.append((current.host_process, world.object_process, "serve"))
    return True, steps

def shortcut_after_success(world: World, proxy_id: str):
    """After a successful call, make the proxy point to the skeleton near the object (if exists)."""
    pr = world.proxies[proxy_id]
    obj_host = world.object_process
    skel_id = f"S@{obj_host}"
    pr.known_skeleton_process = obj_host if skel_id in world.skeletons else pr.known_skeleton_process

# -------------------------------
# Banking ops (executed by the object at world.object_process)
# -------------------------------
def op_login(world: World, client_proxy: str, username: str, pin: str, twofa_passed: bool) -> str:
    u = world.users.get(username)
    if not u:
        return "❌ LOGIN FAIL — unknown user"
    if pin != u["pin"]:
        return "❌ LOGIN FAIL — wrong PIN"
    if u.get("require_2fa", False) and not twofa_passed:
        world.sessions[client_proxy] = {"user": username, "2fa": False}
        return "⚠️ LOGIN needs 2FA"
    world.sessions[client_proxy] = {"user": username, "2fa": True}
    return f"✅ LOGIN OK — {username}"

def _is_authed(world: World, client_proxy: str) -> bool:
    return client_proxy in world.sessions and world.sessions[client_proxy].get("2fa") is True

def op_balance(world: World, client_proxy: str) -> str:
    if not _is_authed(world, client_proxy):
        return "⛔ balance denied — not authenticated"
    user = world.sessions[client_proxy]["user"]
    rows = [(a, world.accounts.get(a, 0)) for a in world.users[user]["accounts"]]
    return "💰 " + ", ".join([f"{a}={b}" for a, b in rows])

def op_deposit(world: World, client_proxy: str, account: str, amount: int) -> str:
    if not _is_authed(world, client_proxy):
        return "⛔ deposit denied — not authenticated"
    user = world.sessions[client_proxy]["user"]
    if account not in world.users[user]["accounts"]:
        return "⛔ deposit denied — not owner"
    world.accounts[account] = world.accounts.get(account, 0) + amount
    return f"✅ deposit {amount} → {account} = {world.accounts[account]}"

def op_withdraw(world: World, client_proxy: str, account: str, amount: int, daily_limit: int = 500) -> str:
    if not _is_authed(world, client_proxy):
        return "⛔ withdraw denied — not authenticated"
    if amount > daily_limit:
        return f"⛔ withdraw denied — > daily limit {daily_limit}"
    user = world.sessions[client_proxy]["user"]
    if account not in world.users[user]["accounts"]:
        return "⛔ withdraw denied — not owner"
    bal = world.accounts.get(account, 0)
    if amount > bal:
        return f"⛔ withdraw denied — insufficient funds ({bal})"
    world.accounts[account] = bal - amount
    return f"✅ withdraw {amount} → {account} = {world.accounts[account]}"

# -------------------------------
# Interactive graph (PyVis)
# -------------------------------
COLORS = {
    "proxy":   "#1f77b4",  # blue
    "forward": "#ff7f0e",  # orange
    "serve":   "#7f7f7f",  # gray
    "highlight": "#d62728" # red
}

def _legend_edges(net: Network):
    # Tiny legend subgraph (fixed positions for clarity)
    net.add_node("L", label="Legend", x=-600, y=-260, fixed=True, physics=False, color="#eeeeee", shape="box")
    net.add_node("L1", label="proxy", x=-750, y=-200, fixed=True, physics=False, color="#ddeeff")
    net.add_node("L2", label="forward", x=-750, y=-140, fixed=True, physics=False, color="#ffe4cc")
    net.add_node("L3", label="serve", x=-750, y=-80,  fixed=True, physics=False, color="#eeeeee")
    net.add_node("L4", label="highlighted step", x=-750, y=-20, fixed=True, physics=False, color="#ffdada")

    net.add_edge("L1", "L", color=COLORS["proxy"],   width=3, arrows="to", smooth=True, title="Proxy call (client → skeleton)")
    net.add_edge("L2", "L", color=COLORS["forward"], width=3, dashes=True, arrows="to", smooth=True, title="Forwarding pointer (old skeleton → new location)")
    net.add_edge("L3", "L", color=COLORS["serve"],   width=2, arrows="to", smooth=True, title="Skeleton dispatch to object")
    net.add_edge("L4", "L", color=COLORS["highlight"], width=5, arrows="to", smooth=True, title="Route step highlight (① ② ③ …)")

def build_graph_html(world: World,
                     route: List[Tuple[str, str, str]],
                     explain_steps: bool = True,
                     local_invocation_hint: Optional[str] = None) -> str:
    """
    Build interactive HTML for Streamlit via PyVis.
    - route: list of (src, dst, kind)
    - explain_steps: add numeric labels ①, ②, ③ to edges in the route with rich tooltips
    - local_invocation_hint: optional note shown in the ATM or MobileApp node tooltip when co-located
    """
    net = Network(height="680px", width="100%", directed=True, cdn_resources="in_line", bgcolor="#ffffff")
    net.set_options("""{
      "physics": {"stabilization": true, "barnesHut": {"gravitationalConstant": -2000, "springLength": 180}},
      "edges": {"smooth": {"type": "dynamic"}}
    }""")

    # Pre-calc node tooltips
    def node_title(pname: str) -> str:
        proc = world.processes[pname]
        parts = [f"<b>{pname}</b>: {proc.label}"]
        if proc.proxies:
            parts.append(f"🧩 Proxies here: {', '.join(proc.proxies)}")
        if proc.skeletons:
            parts.append(f"🧱 Skeletons here: {len(proc.skeletons)}")
        if proc.holds_object:
            # Show compact ledger in the object node
            ledger = ", ".join([f"{a}={b}" for a, b in world.accounts.items()])
            parts.append(f"🗄️ <b>Object host</b> (Account Service)<br/>Ledger: {ledger}")
        if local_invocation_hint and pname in ("P1", "P2"):
            parts.append(local_invocation_hint)
        return "<br/>".join(parts)

    # Add nodes
    for pname, proc in world.processes.items():
        color = "#fff6d5" if proc.holds_object else "#e8f4fa"
        shape = "box"
        net.add_node(pname, label=f"{pname}\n{proc.label}", title=node_title(pname),
                     color=color, shape=shape)

    # Build full edge set: proxy, forward, serve
    all_edges: List[Tuple[str, str, str]] = []

    # Proxy edges (from every proxy's owner → known skeleton process)
    for pid, pr in world.proxies.items():
        title = (f"① Proxy call (client → skeleton)<br/>"
                 f"<b>Proxy:</b> {pid}<br/><b>Owner:</b> {pr.owner_process}<br/><b>Thinks skeleton at:</b> {pr.known_skeleton_process}")
        all_edges.append((pr.owner_process, pr.known_skeleton_process, "proxy"))
        net.add_edge(pr.owner_process, pr.known_skeleton_process,
                     color=COLORS["proxy"], width=3, arrows="to", title=title)

    # Forwarding pointer edges (skeleton host → forwards_to)
    for s in world.skeletons.values():
        if s.forwards_to:
            title = ("Forwarding pointer (old skeleton → new location)<br/>"
                     f"<b>From:</b> {s.host_process} <b>to:</b> {s.forwards_to}")
            all_edges.append((s.host_process, s.forwards_to, "forward"))
            net.add_edge(s.host_process, s.forwards_to,
                         color=COLORS["forward"], width=3, arrows="to", dashes=True, title=title)

    # Serve edges (skeleton host → object)
    for s in world.skeletons.values():
        title = ("Skeleton dispatch to Object<br/>"
                 f"<b>Skeleton at:</b> {s.host_process} <b>→ Object at:</b> {world.object_process}")
        all_edges.append((s.host_process, world.object_process, "serve"))
        net.add_edge(s.host_process, world.object_process,
                     color=COLORS["serve"], width=2, arrows="to", title=title)

    # Highlight the actual route with step numbers
    if route and explain_steps:
        circ_nums = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨"]
        for idx, (src, dst, kind) in enumerate(route):
            label = circ_nums[idx] if idx < len(circ_nums) else str(idx+1)
            kind_text = {"proxy":"Proxy call (client → skeleton)",
                         "forward":"Forwarding pointer (old skeleton → new location)",
                         "serve":"Skeleton dispatch to Object"}[kind]
            # Add a duplicate "highlight" edge on top (thicker + label)
            net.add_edge(src, dst, color=COLORS["highlight"], width=5, arrows="to",
                         title=f"<b>{label}</b> {kind_text}", label=label, font={"align":"top"})

    # Legend
    _legend_edges(net)

    return net.generate_html()

# -------------------------------
# UI (graph-only explanations; controls are in the sidebar)
# -------------------------------
if "world" not in st.session_state:
    st.session_state.world = init_world()

world = st.session_state.world

with st.sidebar:
    st.header("Controls")

    st.markdown("**Client & Operation**")
    client = st.selectbox("Client proxy", ["px_atm", "px_mob"], index=0)
    op = st.selectbox("Operation", ["login", "balance", "deposit", "withdraw"], index=0)

    st.markdown("**Auth**")
    user = st.selectbox("User", list(world.users.keys()), index=0)
    pin = st.text_input("PIN", value="1234", type="password")
    twofa = st.checkbox("2FA passed (if required)", value=False)

    st.markdown("**Account / Amount**")
    account = st.selectbox("Account", list(world.accounts.keys()), index=0)
    amount = st.number_input("Amount (for deposit/withdraw)", min_value=1, max_value=10000, value=100)

    st.markdown("---")
    st.markdown("**Topology**")
    local_demo = st.checkbox("Co-locate skeleton at ATM (Local Invocation demo)", value=False)
    migrate_to = st.selectbox("Move Object to:", ["(no move)","P1","P2","P3","P4"], index=0)
    keep_forwarding = st.checkbox("Keep forwarding pointers after move", value=True)

    colb1, colb2 = st.columns(2)
    with colb1:
        run_btn = st.button("▶ Run")
    with colb2:
        reset_btn = st.button("⟲ Reset")

# Apply topology changes (before computing route)
if reset_btn:
    st.session_state.world = init_world()
    world = st.session_state.world

# Local invocation: co-locate a skeleton at ATM and point ATM proxy there
if local_demo:
    ensure_skeleton(world, "P1")
    world.proxies["px_atm"].known_skeleton_process = "P1"

# Optional migration
if migrate_to != "(no move)":
    move_object(world, migrate_to, create_new_skeleton=True, keep_forwarding=keep_forwarding)

# Execute selected operation through routing, and update tooltips only (no separate text panel)
route_steps: List[Tuple[str, str, str]] = []
op_result = ""

ok, route_steps = compute_route(world, client)
if run_btn and ok:
    # Perform the operation at the object (after routing)
    if op == "login":
        op_result = op_login(world, client, user, pin, twofa_passed=twofa)
    elif op == "balance":
        op_result = op_balance(world, client)
    elif op == "deposit":
        op_result = op_deposit(world, client, account, int(amount))
    elif op == "withdraw":
        op_result = op_withdraw(world, client, account, int(amount))
    # Shortcut the proxy to the object's skeleton if present
    shortcut_after_success(world, client)
elif run_btn and not ok:
    op_result = "❌ Routing failed — no skeleton at known location."

# Hint for local invocation (adds to node tooltip)
local_hint = None
if local_demo:
    local_hint = "🟢 <b>Local invocation</b>: proxy and skeleton are co-located here → no network hop."

# After operation, rebuild graph with live state embedded in tooltips and highlighted route
graph_html = build_graph_html(world, route_steps, explain_steps=True, local_invocation_hint=local_hint)

# Add an operation badge into the graph by putting it in the title of the object node (already done),
# and also show a tiny status chip above the graph (non-intrusive)
if run_btn and op_result:
    st.caption(f"Status: {op_result}")

# Render graph (all explanations live in hover tooltips & edge labels)
html(graph_html, height=720)
