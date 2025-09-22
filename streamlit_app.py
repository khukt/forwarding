# banking_omnet_style_app.py
# OMNeT++-style, traceable animation of proxy → skeleton → (forwarding)* → object
# - Animated frames with Plotly (play/pause + scrub slider)
# - Everything is explained on the GRAPH (hover tooltips + step annotations)
# - Real flows: login / balance / deposit / withdraw
# - Local invocation demo, migration + forwarding pointers, shortcutting after success
#
# Run:
#   pip install streamlit plotly networkx pandas
#   streamlit run banking_omnet_style_app.py

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Set
import math
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(
    page_title="🏦 OMNeT++-style Banking — Proxy • Skeleton • Forwarding",
    layout="wide"
)

# -------------------------------
# Data model
# -------------------------------
@dataclass
class Process:
    name: str
    label: str
    proxies: List[str] = field(default_factory=list)
    skeletons: List[str] = field(default_factory=list)
    holds_object: bool = False  # Account Service here?

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
    sessions: Dict[str, Dict[str, object]] = field(default_factory=dict)  # client proxy id → {"user":..., "2fa":bool}
    log: List[Dict[str, object]] = field(default_factory=list)  # event trace

# -------------------------------
# Colors
# -------------------------------
C = {
    "node_obj": "#fff6d5",
    "node_proc": "#e8f4fa",
    "edge_proxy": "#1f77b4",    # blue
    "edge_forward": "#ff7f0e",  # orange
    "edge_serve": "#7f7f7f",    # gray
    "edge_hl": "#d62728",       # red
    "token": "#d62728",         # red
}

# -------------------------------
# World helpers
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

def node_title(world: World, pname: str, local_hint: Optional[str] = None) -> str:
    proc = world.processes[pname]
    parts = [f"<b>{pname}</b>: {proc.label}"]
    if proc.proxies:
        parts.append(f"🧩 Proxies: {', '.join(proc.proxies)}")
    if proc.skeletons:
        parts.append(f"🧱 Skeletons: {len(proc.skeletons)}")
    if proc.holds_object:
        ledger = ", ".join([f"{a}={b}" for a, b in world.accounts.items()])
        parts.append(f"🗄️ <b>Object host</b> (Account Service)<br/>Ledger: {ledger}")
    if local_hint and pname in ("P1", "P2"):
        parts.append(local_hint)
    return "<br/>".join(parts)

# -------------------------------
# Routing (proxy → skeleton → [forwarding]* → object)
# -------------------------------
def compute_route(world: World, proxy_id: str) -> Tuple[bool, List[Tuple[str, str, str]], List[str]]:
    """
    Returns (ok, steps, step_labels)
      steps = list of (src, dst, kind), kind ∈ {"proxy","forward","serve"}
      step_labels = human-readable labels for animation (e.g., "Proxy call (px_atm): P1→P3 (IPC)")
    """
    if proxy_id not in world.proxies:
        return False, [], []
    pr = world.proxies[proxy_id]
    steps: List[Tuple[str, str, str]] = []
    labels: List[str] = []

    # proxy hop
    hop_type = "LOCAL" if pr.owner_process == pr.known_skeleton_process else "IPC"
    steps.append((pr.owner_process, pr.known_skeleton_process, "proxy"))
    labels.append(f"Proxy call ({pr.id}): {pr.owner_process}→{pr.known_skeleton_process} ({hop_type})")

    # identify skeleton at known location
    target = None
    for s in world.skeletons.values():
        if s.host_process == pr.known_skeleton_process:
            target = s
            break
    if not target:
        return False, steps, labels

    # follow forwarding chain
    current = target
    seen: Set[str] = set()
    while current.forwards_to and current.forwards_to not in seen:
        seen.add(current.host_process)
        steps.append((current.host_process, current.forwards_to, "forward"))
        labels.append(f"Forwarding pointer: {current.host_process}→{current.forwards_to}")
        nxt = None
        for s in world.skeletons.values():
            if s.host_process == current.forwards_to:
                nxt = s
                break
        if not nxt:
            break
        current = nxt

    # final service hop to where object actually lives (conceptual)
    steps.append((current.host_process, world.object_process, "serve"))
    labels.append(f"Skeleton dispatch: {current.host_process}→{world.object_process} (object)")

    return True, steps, labels

def shortcut_after_success(world: World, proxy_id: str):
    pr = world.proxies[proxy_id]
    obj_host = world.object_process
    skel_id = f"S@{obj_host}"
    if skel_id in world.skeletons:
        pr.known_skeleton_process = obj_host

# -------------------------------
# Banking operations (executed by the object)
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
# Layout + drawing helpers (Plotly)
# -------------------------------
def fixed_layout_pos() -> Dict[str, Tuple[float, float]]:
    # Hand-tuned layout for clarity (stable across runs)
    # You can tweak coordinates to taste.
    return {
        "P1": (-1.2,  0.6),   # ATM
        "P2": (-1.2, -0.6),   # Mobile
        "P3": ( 0.2,  0.0),   # Gateway
        "P4": ( 1.6,  0.0),   # Account Service
        "L":  (-2.0, -1.6),   # Legend anchor (off to the side)
    }

def interpolate(p1, p2, t: float) -> Tuple[float, float]:
    return (p1[0] * (1-t) + p2[0] * t, p1[1] * (1-t) + p2[1] * t)

def edge_line(x1,y1,x2,y2, color, width, dash=None, name=None, hover=None):
    return go.Scatter(
        x=[x1, x2], y=[y1, y2],
        mode="lines",
        line=dict(color=color, width=width, dash=dash if dash else "solid"),
        hoverinfo="text", text=hover, name=name if name else "",
        showlegend=False
    )

def arrow_annot(x1,y1,x2,y2, text:str, color:str):
    # Place arrow head near destination; ax/ay are tail offsets
    ax = (x1 - x2) * 60
    ay = (y1 - y2) * 60
    return dict(
        x=x2, y=y2, ax=x2+ax, ay=y2+ay,
        xref="x", yref="y", axref="x", ayref="y",
        text=text, showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2, arrowcolor=color,
        font=dict(color=color, size=14), bgcolor="rgba(255,255,255,0.6)"
    )

def build_static_traces(world: World, pos: Dict[str, Tuple[float, float]]):
    # Nodes
    x_nodes, y_nodes, text_nodes, colors = [], [], [], []
    for pname, proc in world.processes.items():
        x, y = pos[pname]
        x_nodes.append(x); y_nodes.append(y)
        label = f"{pname}<br>{proc.label}"
        if proc.holds_object:
            label += "<br>[OBJECT]"
        if proc.proxies:
            label += "<br>Proxies: " + ", ".join(proc.proxies)
        if proc.skeletons:
            label += f"<br>Skeletons: {len(proc.skeletons)}"
        # ledger & sessions on object node
        if proc.holds_object:
            ledger = ", ".join([f"{a}={b}" for a, b in world.accounts.items()])
            label += f"<br><b>Ledger:</b> {ledger}"
        text_nodes.append(label)
        colors.append(C["node_obj"] if proc.holds_object else C["node_proc"])
    nodes = go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers+text",
        text=[p for p in world.processes.keys()],
        textposition="bottom center",
        hovertext=text_nodes, hoverinfo="text",
        marker=dict(size=36, color=colors, line=dict(color="#1d3557", width=2)),
        showlegend=False
    )

    # All base edges (thin, semantic hover)
    base_edges = []
    # proxy edges
    for pid, pr in world.proxies.items():
        x1,y1 = pos[pr.owner_process]; x2,y2 = pos[pr.known_skeleton_process]
        base_edges.append(edge_line(x1,y1,x2,y2, C["edge_proxy"], 2,
                                    name="proxy", hover=f"Proxy call ({pid}): {pr.owner_process}→{pr.known_skeleton_process}"))

    # forwarding edges
    for s in world.skeletons.values():
        if s.forwards_to:
            x1,y1 = pos[s.host_process]; x2,y2 = pos[s.forwards_to]
            base_edges.append(edge_line(x1,y1,x2,y2, C["edge_forward"], 2, dash="dash",
                                        name="forward", hover=f"Forwarding: {s.host_process}→{s.forwards_to}"))

    # serve edges
    for s in world.skeletons.values():
        x1,y1 = pos[s.host_process]; x2,y2 = pos[world.object_process]
        base_edges.append(edge_line(x1,y1,x2,y2, C["edge_serve"], 1.5, dash="dot",
                                    name="serve", hover=f"Serve: {s.host_process}→{world.object_process} (object)"))

    return nodes, base_edges

def build_animation(world: World,
                    route: List[Tuple[str,str,str]],
                    labels: List[str],
                    pos: Dict[str, Tuple[float,float]],
                    frames_per_hop: int = 12):
    """Return (fig, frames, slider_steps) Plotly objects."""
    nodes, base_edges = build_static_traces(world, pos)

    # Base figure
    fig = go.Figure(data=[*base_edges, nodes])
    fig.update_layout(
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        dragmode="pan", hovermode="closest",
        title="Animated Route — Hover nodes/edges for details; use Play ▶ or slider to scrub"
    )

    # If no route (not run yet), return static fig
    if not route:
        return fig, [], []

    # Frames: moving token + highlighted edge + annotation with step text
    frames = []
    slider_steps = []
    circ = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨"]

    # We'll add token as the LAST trace per frame so it hovers nicely
    for h, (src, dst, kind) in enumerate(route):
        x1,y1 = pos[src]; x2,y2 = pos[dst]
        label = circ[h] if h < len(circ) else str(h+1)
        desc = labels[h] if h < len(labels) else f"Step {h+1}: {src}→{dst}"

        for k in range(frames_per_hop):
            t = (k+1)/frames_per_hop
            xt, yt = interpolate((x1,y1), (x2,y2), t)
            # highlighted edge
            hl_edge = edge_line(x1,y1,x2,y2, C["edge_hl"], 5)
            # token
            token = go.Scatter(
                x=[xt], y=[yt], mode="markers+text",
                marker=dict(size=18, color=C["token"], symbol="diamond"),
                text=[label], textposition="top center",
                hoverinfo="text", textfont=dict(size=14, color=C["edge_hl"]),
                showlegend=False
            )
            # annotation arrow with explanation
            ann = [arrow_annot(x1,y1,x2,y2, f"{label} {desc}", C["edge_hl"])]

            frames.append(go.Frame(
                name=f"f{h}_{k}",
                data=[*base_edges, nodes, hl_edge, token],
                layout=go.Layout(annotations=ann)
            ))

        slider_steps.append(dict(
            label=label, method="animate",
            args=[[f"f{h}_0"], {"frame": {"duration": 1, "redraw": True}, "mode": "immediate"}]
        ))

    fig.update(frames=frames)

    # Animation controls
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0.0, "y": 1.12,
            "pad": {"r": 8, "t": 8},
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True} ]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "y": 1.06,
            "pad": {"t": 30},
            "steps": [
                # create a step for every frame, not just hop start, for fine scrubbing
                *[{
                    "label": f if i % frames_per_hop else (circ[i//frames_per_hop] if i//frames_per_hop < len(circ) else str(i//frames_per_hop+1)),
                    "method": "animate",
                    "args": [[f], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                } for i, f in enumerate([fr.name for fr in frames])]
            ]
        }]
    )
    return fig, frames, slider_steps

# -------------------------------
# Trace recording
# -------------------------------
def record_trace(world: World, route: List[Tuple[str,str,str]], labels: List[str], op_result: str):
    # Simple time = frame index (discrete). One row per hop (end of hop).
    rows = []
    t = 0
    for i, (src, dst, kind) in enumerate(route):
        rows.append({"t": t, "step": i+1, "src": src, "dst": dst, "kind": kind, "desc": labels[i]})
        t += 1
    if op_result:
        rows.append({"t": t, "step": "result", "src": "-", "dst": "-", "kind": "result", "desc": op_result})
    world.log.extend(rows)

def trace_df(world: World) -> pd.DataFrame:
    if not world.log:
        return pd.DataFrame(columns=["t","step","src","dst","kind","desc"])
    return pd.DataFrame(world.log)

# -------------------------------
# Streamlit state
# -------------------------------
if "world" not in st.session_state:
    st.session_state.world = init_world()
world = st.session_state.world

# -------------------------------
# Sidebar controls (like a simulator config)
# -------------------------------
with st.sidebar:
    st.header("Simulation Controls")

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
    local_demo = st.checkbox("Local Invocation demo (co-locate skeleton at ATM)", value=False)
    migrate_to = st.selectbox("Move Object to:", ["(no move)","P1","P2","P3","P4"], index=0)
    keep_forwarding = st.checkbox("Keep forwarding pointers after move", value=True)

    st.markdown("**Animation**")
    frames_per_hop = st.slider("Frames per hop (smoothness)", 4, 30, 12)

    colb1, colb2, colb3 = st.columns(3)
    with colb1:
        run_btn = st.button("▶ Run")
    with colb2:
        reset_btn = st.button("⟲ Reset world")
    with colb3:
        clear_trace_btn = st.button("🧹 Clear trace")

# Apply topology changes
if reset_btn:
    st.session_state.world = init_world()
    world = st.session_state.world

if clear_trace_btn:
    world.log.clear()

# Local invocation: co-locate a skeleton at ATM and point ATM proxy there
if local_demo:
    ensure_skeleton(world, "P1")
    world.proxies["px_atm"].known_skeleton_process = "P1"

# Optional migration
if migrate_to != "(no move)":
    move_object(world, migrate_to, create_new_skeleton=True, keep_forwarding=keep_forwarding)

# -------------------------------
# Compute route & execute operation (if Run)
# -------------------------------
ok, route, labels = compute_route(world, client)
op_result = ""
if run_btn and ok:
    if op == "login":
        op_result = op_login(world, client, user, pin, twofa_passed=twofa)
    elif op == "balance":
        op_result = op_balance(world, client)
    elif op == "deposit":
        op_result = op_deposit(world, client, account, int(amount))
    elif op == "withdraw":
        op_result = op_withdraw(world, client, account, int(amount))
    # OMNeT++-style: record the trace of this transaction
    record_trace(world, route, labels, op_result)
    # Shortcut after success (proxy learns object's skeleton)
    shortcut_after_success(world, client)
elif run_btn and not ok:
    op_result = "❌ Routing failed — no skeleton at known location."
    record_trace(world, route, labels, op_result)

# -------------------------------
# Build animated figure (graph explains itself)
# -------------------------------
pos = fixed_layout_pos()
fig, frames, _ = build_animation(world, route if ok else [], labels if ok else [], pos, frames_per_hop=frames_per_hop)

left, right = st.columns([3, 2], vertical_alignment="top")
with left:
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Trace (like OMNeT++)")
    df = trace_df(world)
    if df.empty:
        st.info("No events yet. Press ▶ Run to execute a flow and record its route.")
    else:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download trace CSV", data=csv, file_name="bank_trace.csv", mime="text/csv")
    if op_result:
        st.caption(f"Status: {op_result}")

    st.markdown("---")
    st.markdown("**Legend**  \n"
                f"• **Proxy** edge = {C['edge_proxy']}  \n"
                f"• **Forwarding** edge = {C['edge_forward']}  \n"
                f"• **Serve** edge = {C['edge_serve']}  \n"
                f"• **Highlight** = {C['edge_hl']}  \n"
                f"• Token = moving red diamond (step label ① ② ③ …)")

