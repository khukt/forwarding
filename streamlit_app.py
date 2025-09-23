# app_fixed32_pro.py
# Chord DHT — Fixed ring 0..31 with polished, compact UI:
# - Above-the-fold layout: chart left, details in tabs right
# - Control bar on top, status metrics, preset scenarios
# - Disabled placeholders for non-active IDs, finger reveal, step-by-step lookup

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Page / Style ----------
st.set_page_config(
    page_title="Chord 0..31 • Visual Tutor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Tighten default padding to fit above the fold
st.markdown(
    """
    <style>
    .block-container { padding-top: 0.75rem; padding-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.25rem 0.75rem; }
    .stMetric { background: #fafafa; border-radius: 12px; padding: 0.5rem 0.75rem; }
    .small-note { color: #666; font-size: 0.85rem; }
    .tight { margin-top: -0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Constants ----------
M = 5
SPACE = 2 ** M
ALL_POSITIONS = list(range(SPACE))

# ---------- Math helpers ----------
def sha1_mod(s: str, space: int) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h, 16) % space

def mod_interval_contains(a: int, b: int, x: int, m: int, inclusive_right: bool = False) -> bool:
    if a == b:
        return inclusive_right
    if a < b:
        return (a < x <= b) if inclusive_right else (a < x < b)
    return ((a < x <= m - 1) or (0 <= x <= b)) if inclusive_right else ((a < x < m) or (0 <= x < b))

def successor_of(x: int, nodes_sorted: List[int]) -> int:
    for n in nodes_sorted:
        if n >= x:
            return n
    return nodes_sorted[0]

@dataclass
class FingerEntry:
    i: int
    start: int
    node: int

def build_finger_table(n: int, nodes_sorted: List[int], m: int) -> List[FingerEntry]:
    entries: List[FingerEntry] = []
    if not nodes_sorted:
        return entries
    space = 2 ** m
    for i in range(1, m + 1):
        start = (n + 2 ** (i - 1)) % space
        succ = successor_of(start, nodes_sorted)
        entries.append(FingerEntry(i=i, start=start, node=succ))
    return entries

def closest_preceding_finger(n: int, fingers: List[int], target: int, m: int) -> int:
    for f in reversed(fingers):
        if f != n and mod_interval_contains(n, target, f, 2 ** m, inclusive_right=False):
            return f
    return n

def chord_lookup_full(start_node: int, key: int, nodes_sorted: List[int], m: int, max_steps: int = 64):
    """Return (path, reasons) for iterative Chord lookup using only active nodes."""
    if not nodes_sorted:
        return [start_node], [r"\text{No active nodes.}"]
    path = [start_node]
    reasons: List[str] = []
    succ_k = successor_of(key, nodes_sorted)
    finger_map: Dict[int, List[int]] = {n: [fe.node for fe in build_finger_table(n, nodes_sorted, m)]
                                        for n in nodes_sorted}
    while len(path) < max_steps:
        curr = path[-1]
        if curr == succ_k:
            reasons.append(rf"\mathbf{{Stop:}}\ \text{{current}}={curr}=\operatorname{{succ}}({key})")
            break

        curr_idx = nodes_sorted.index(curr)
        curr_succ = nodes_sorted[(curr_idx + 1) % len(nodes_sorted)]

        if mod_interval_contains(curr, curr_succ, key, 2 ** m, inclusive_right=True):
            reasons.append(
                rf"\text{{Since }} {key}\in({curr},{curr_succ}] \Rightarrow "
                rf"\text{{next}}=\operatorname{{succ}}({curr})={curr_succ}"
            )
            path.append(curr_succ)
            if curr_succ == succ_k:
                reasons.append(rf"\mathbf{{Arrived}}\ \text{{at}}\ \operatorname{{succ}}({key})={succ_k}")
                break
            continue

        cpf = closest_preceding_finger(curr, finger_map[curr], key, m)
        if cpf == curr:
            reasons.append(
                rf"\text{{No finger in }}({curr},{key}) \Rightarrow "
                rf"\text{{fallback to }} \operatorname{{succ}}({curr})={curr_succ}"
            )
            path.append(curr_succ)
        else:
            reasons.append(
                rf"\text{{Choose closest preceding finger of }}{curr}\ \text{{toward }}{key}: "
                rf"{cpf}\in({curr},{key})"
            )
            path.append(cpf)

        if path[-1] == succ_k:
            reasons.append(rf"\mathbf{{Arrived}}\ \text{{at}}\ \operatorname{{succ}}({key})={succ_k}")
            break

    return path, reasons

# ---------- Plot helpers ----------
def node_xy(id_val: int, space: int, radius: float = 1.0) -> Tuple[float, float]:
    theta = 2 * math.pi * (id_val / space)
    return radius * math.cos(theta), radius * math.sin(theta)

def ring_figure(
    active_nodes: List[int],
    selected: Optional[int] = None,
    fingers: Optional[List[FingerEntry]] = None,
    highlight_start: Optional[int] = None,
    show_radial: bool = False,
    pin_selected_ft: bool = False,
    route_path: Optional[List[int]] = None,
    route_hops_to_show: int = 0,
    key: Optional[int] = None,
    width: int = 720,
    height: int = 720,
) -> go.Figure:
    R = 1.0
    circle_angles = np.linspace(0, 2*np.pi, 361)
    fig = go.Figure()

    # Base ring
    fig.add_trace(go.Scatter(x=np.cos(circle_angles), y=np.sin(circle_angles),
                             mode="lines", name="Ring", hoverinfo="skip"))

    # Disabled placeholders (0..31 not in active)
    active_set = set(active_nodes)
    disabled_positions = [i for i in ALL_POSITIONS if i not in active_set]
    if disabled_positions:
        xs, ys = [], []
        for nid in disabled_positions:
            x, y = node_xy(nid, SPACE, R)
            xs.append(x); ys.append(y)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=[str(i) for i in disabled_positions],
            textposition="top center",
            marker=dict(size=9, symbol="circle-open", color="lightgray",
                        line=dict(width=1, color="lightgray")),
            name="Disabled", opacity=0.45, hoverinfo="skip"
        ))

    # Active nodes (selected red, succ(key) orange)
    succ_k = successor_of(key, active_nodes) if (key is not None and active_nodes) else None
    xs, ys, sizes, colors, labels = [], [], [], [], []
    for nid in active_nodes:
        x, y = node_xy(nid, SPACE, R)
        xs.append(x); ys.append(y)
        if selected == nid:
            sizes.append(16); colors.append("crimson"); labels.append(f"{nid} (selected)")
        elif succ_k is not None and succ_k == nid:
            sizes.append(14); colors.append("orange"); labels.append(f"{nid} (succ(key))")
        else:
            sizes.append(11); colors.append("royalblue"); labels.append(str(nid))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[str(n) for n in active_nodes], textposition="top center",
        hovertext=labels, hoverinfo="text",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
        name="Active nodes"
    ))

    # Finger chords from selected
    if selected is not None and fingers:
        sx, sy = node_xy(selected, SPACE, R)
        for fe in fingers:
            tx, ty = node_xy(fe.node, SPACE, R)
            fig.add_trace(go.Scatter(
                x=[sx, tx], y=[sy, ty], mode="lines",
                line=dict(width=2, dash="dot"),
                name=f"finger[{fe.i}]→{fe.node}",
                hovertext=f"start={fe.start} → succ={fe.node}",
                hoverinfo="text", showlegend=False
            ))

    # Highlight start[i]
    if highlight_start is not None and selected is not None:
        hx, hy = node_xy(highlight_start, SPACE, R)
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy], mode="markers+text",
            text=[f"start={highlight_start}"], textposition="bottom center",
            marker=dict(size=13, symbol="diamond", line=dict(width=1, color="black")),
            name="start[i]", hoverinfo="text"
        ))
        if show_radial:
            sx, sy = node_xy(selected, SPACE, R)
            fig.add_trace(go.Scatter(
                x=[sx, hx], y=[sy, hy], mode="lines",
                line=dict(width=1, dash="dash"), name="n→start", hoverinfo="skip", showlegend=False
            ))

    # Route arrows for revealed hops
    if route_path and route_hops_to_show > 0:
        for i in range(min(route_hops_to_show, len(route_path) - 1)):
            a = route_path[i]; b = route_path[i + 1]
            ax, ay = node_xy(a, SPACE, R)
            bx, by = node_xy(b, SPACE, R)
            fig.add_trace(go.Scatter(
                x=[ax, bx], y=[ay, by], mode="lines+markers",
                line=dict(width=3), marker=dict(size=6),
                name=f"hop {i+1}", hoverinfo="skip", showlegend=False
            ))
            fig.add_annotation(
                x=bx, y=by, ax=ax, ay=ay,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2
            )

    # Pin selected node's finger table card (tiny)
    if selected is not None and fingers:
        x, y = node_xy(selected, SPACE, R * 1.10)
        table = "i  start  succ<br>" + "<br>".join(f"{fe.i}  {fe.start:<5}  {fe.node}" for fe in fingers)
        fig.add_annotation(
            x=x, y=y, xanchor="left", yanchor="middle",
            text=f"<b>Node {selected}</b><br><span style='font-family:monospace; font-size:12px'>{table}</span>",
            showarrow=False, bordercolor="black", borderwidth=1, bgcolor="white", opacity=0.95
        )

    fig.update_layout(
        width=width, height=height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=4, r=4, t=28, b=4),
        plot_bgcolor="white",
        title="Chord • Ring 0..31 • Fingers • Lookup",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# ---------- Session state ----------
def init_state():
    if "k" not in st.session_state:
        st.session_state.k = 0  # fingers revealed
    if "route_path" not in st.session_state:
        st.session_state.route_path: List[int] = []
    if "route_reasons" not in st.session_state:
        st.session_state.route_reasons: List[str] = []
    if "route_idx" not in st.session_state:
        st.session_state.route_idx = 0  # hops revealed
    if "active_nodes" not in st.session_state:
        st.session_state.active_nodes = [1, 4, 9, 11, 14, 18, 20, 21, 28]
    if "selected" not in st.session_state:
        st.session_state.selected = st.session_state.active_nodes[0]
    if "key_id" not in st.session_state:
        st.session_state.key_id = 26

init_state()

# ---------- Title Row ----------
tcol1, tcol2 = st.columns([0.72, 0.28])
with tcol1:
    st.subheader("🔗 Chord DHT — Visual Tutor (Fixed 0..31)")
    st.caption("All positions 0..31 are drawn. Non-listed IDs appear as grey placeholders. Routing uses only active nodes.")
with tcol2:
    # Presets for instant demos
    st.write("Presets")
    c1, c2 = st.columns(2)
    if c1.button("Demo A: k=12 from 28"):
        st.session_state.active_nodes = [1, 4, 9, 11, 14, 18, 20, 21, 28]
        st.session_state.selected = 28
        st.session_state.key_id = 12
        st.session_state.k = 5
        path, reasons = chord_lookup_full(28, 12, st.session_state.active_nodes, M)
        st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_idx = path, reasons, len(path)-1
    if c2.button("Demo B: k=26 from 1"):
        st.session_state.active_nodes = [1, 4, 9, 11, 14, 18, 20, 21, 28]
        st.session_state.selected = 1
        st.session_state.key_id = 26
        st.session_state.k = 5
        path, reasons = chord_lookup_full(1, 26, st.session_state.active_nodes, M)
        st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_idx = path, reasons, len(path)-1

# ---------- Top Control Bar ----------
bar = st.container()
with bar:
    b1, b2, b3, b4, b5, b6 = st.columns([2.5, 1.2, 1.2, 1, 1, 1.2])

    with b1:
        ids_text = st.text_input(
            "Active nodes (0..31)",
            value=", ".join(str(n) for n in st.session_state.active_nodes),
            help="Comma/space separated. Others will show as disabled placeholders."
        )
        raw = [t.strip() for t in ids_text.replace(",", " ").split()]
        try:
            active_nodes = sorted(set(int(x) % SPACE for x in raw if x != ""))
        except ValueError:
            active_nodes = st.session_state.active_nodes
        st.session_state.active_nodes = active_nodes

    with b2:
        if not st.session_state.active_nodes:
            st.session_state.active_nodes = [1]
        st.session_state.selected = st.selectbox(
            "Selected node n",
            options=st.session_state.active_nodes,
            index=min(len(st.session_state.active_nodes)-1,
                      st.session_state.active_nodes.index(st.session_state.selected)
                      if st.session_state.selected in st.session_state.active_nodes else 0)
        )

    with b3:
        st.session_state.key_id = st.number_input("Key k", 0, 31, st.session_state.key_id, 1)

    with b4:
        if st.button("Reset fingers", help="Hide all finger chords"):
            st.session_state.k = 0
    with b5:
        if st.button("Next finger", help="Reveal next finger entry"):
            st.session_state.k = min(M, st.session_state.k + 1)
    with b6:
        if st.button("Reset route", help="Recompute from current start node & key"):
            path, reasons = chord_lookup_full(st.session_state.selected, st.session_state.key_id, st.session_state.active_nodes, M)
            st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_idx = path, reasons, 0

# Route next-hop button in a small row
b7, _ = st.columns([0.15, 0.85])
with b7:
    if st.button("Next hop"):
        if st.session_state.route_path:
            st.session_state.route_idx = min(
                len(st.session_state.route_path) - 1,
                st.session_state.route_idx + 1
            )
        else:
            path, reasons = chord_lookup_full(st.session_state.selected, st.session_state.key_id, st.session_state.active_nodes, M)
            st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_idx = path, reasons, 0

# ---------- Compute current state ----------
selected = st.session_state.selected
key_id = st.session_state.key_id
active_nodes = st.session_state.active_nodes

fingers_all = build_finger_table(selected, active_nodes, M)
k = st.session_state.k
fingers_shown = fingers_all[:k]
current_start = fingers_shown[-1].start if k > 0 else None

if not st.session_state.route_path:
    path, reasons = chord_lookup_full(selected, key_id, active_nodes, M)
    st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_idx = path, reasons, 0

route_path = st.session_state.route_path
route_reasons = st.session_state.route_reasons
route_hops_to_show = st.session_state.route_idx
succ_k = successor_of(key_id, active_nodes) if active_nodes else None

# ---------- Status Metrics ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active nodes", len(active_nodes))
m2.metric("Selected n", selected)
m3.metric("Key k", key_id)
m4.metric("succ(k)", succ_k if succ_k is not None else "-")

# ---------- Main: Chart + Tabs ----------
left, right = st.columns([0.58, 0.42])

with left:
    fig = ring_figure(
        active_nodes=active_nodes,
        selected=selected,
        fingers=fingers_shown,
        highlight_start=current_start,
        show_radial=True,
        pin_selected_ft=True,
        route_path=route_path,
        route_hops_to_show=route_hops_to_show,
        key=key_id,
        width=720, height=720,
    )
    st.plotly_chart(fig, use_container_width=False)

with right:
    tabs = st.tabs(["Finger table", "Lookup route", "Ring info"])
    with tabs[0]:
        st.markdown("**Definition (m=5):**")
        st.latex(r"\text{start}[i] = (n + 2^{i-1}) \bmod 32")
        st.latex(r"\text{finger}[i] = \operatorname{succ}(\text{start}[i])")
        df_ft = pd.DataFrame(
            [{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in fingers_shown],
            columns=["i", "start", "successor"]
        )
        st.dataframe(df_ft, hide_index=True, height=240, use_container_width=True)
        if k > 0:
            fe = fingers_shown[-1]
            st.markdown("**Current step**")
            st.latex(rf"n = {selected}")
            st.latex(rf"\text{{start}}[{fe.i}] = ({selected} + 2^{{{fe.i-1}}}) \bmod 32 = {fe.start}")
            st.latex(rf"\text{{finger}}[{fe.i}] = \operatorname{{succ}}({fe.start}) = {fe.node}")
        else:
            st.markdown('<div class="small-note">Click <b>Next finger</b> to reveal the first entry.</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("**Responsibility**")
        st.latex(rf"\operatorname{{succ}}(k) = {succ_k}")
        st.markdown("**Path**")
        st.code(" → ".join(str(n) for n in route_path), language="text")
        st.markdown("**Reasoning**")
        # Show reasons that correspond to drawn hops (+1 message for final arrival when reached)
        max_to_show = min(route_hops_to_show + 1, len(route_reasons))
        for i in range(max_to_show):
            st.latex(route_reasons[i])

    with tabs[2]:
        st.markdown("**Ring**")
        st.latex(r"2^5 = 32 \Rightarrow \text{IDs } 0..31")
        st.markdown("**Active nodes**")
        st.dataframe(pd.DataFrame([{"node_id": n} for n in active_nodes]), hide_index=True, height=180)
        disabled_positions = [i for i in ALL_POSITIONS if i not in set(active_nodes)]
        st.markdown("**Disabled placeholders**")
        st.dataframe(pd.DataFrame([{"id": n} for n in disabled_positions]), hide_index=True, height=180)
