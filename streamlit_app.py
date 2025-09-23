# app_fixed32_route.py
# Chord DHT — Fixed ring 0..31, disabled placeholders, fingers, and step-by-step lookup route

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- Fixed ring ----------------
M = 5
SPACE = 2 ** M
ALL_POSITIONS = list(range(SPACE))

# ---------------- Math helpers ----------------
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

# ---------------- Chord data ----------------
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
    """Return (path: List[int], reasons: List[str]) using only nodes_sorted (active)."""
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

# ---------------- Visualization ----------------
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
    key: Optional[int] = None
) -> go.Figure:
    R = 1.0
    circle_angles = np.linspace(0, 2*np.pi, 361)
    fig = go.Figure()

    # Base ring
    fig.add_trace(go.Scatter(x=np.cos(circle_angles), y=np.sin(circle_angles),
                             mode="lines", name="Ring", hoverinfo="skip"))

    active_set = set(active_nodes)
    disabled_positions = [i for i in ALL_POSITIONS if i not in active_set]

    # Disabled placeholders (grey hollow)
    if disabled_positions:
        xs, ys = [], []
        for nid in disabled_positions:
            x, y = node_xy(nid, SPACE, R)
            xs.append(x); ys.append(y)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=[str(i) for i in disabled_positions],
            textposition="top center",
            marker=dict(size=10, symbol="circle-open", color="lightgray",
                        line=dict(width=1, color="lightgray")),
            name="Disabled (placeholder)", opacity=0.45, hoverinfo="skip"
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
            sizes.append(12); colors.append("royalblue"); labels.append(str(nid))
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

    # Highlight current start[i]
    if highlight_start is not None and selected is not None:
        hx, hy = node_xy(highlight_start, SPACE, R)
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy], mode="markers+text",
            text=[f"start={highlight_start}"], textposition="bottom center",
            marker=dict(size=14, symbol="diamond", line=dict(width=1, color="black")),
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
            ))

    # Optional pinned finger table card for selected
    if selected is not None and fingers:
        x, y = node_xy(selected, SPACE, R * 1.12)
        table = "i  start  succ<br>" + "<br>".join(f"{fe.i}  {fe.start:<5}  {fe.node}" for fe in fingers)
        fig.add_annotation(
            x=x, y=y, xanchor="left", yanchor="middle",
            text=f"<b>Node {selected}</b><br><span style='font-family:monospace'>{table}</span>",
            showarrow=False, bordercolor="black", borderwidth=1, bgcolor="white", opacity=0.95
        )

    fig.update_layout(
        width=880, height=880,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        title="Chord • Fixed Ring 0..31 • Fingers • Lookup Route",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# ---------------- Session state ----------------
def init_state():
    if "k" not in st.session_state:
        st.session_state.k = 0  # fingers revealed
    if "route_path" not in st.session_state:
        st.session_state.route_path: List[int] = []
    if "route_reasons" not in st.session_state:
        st.session_state.route_reasons: List[str] = []
    if "route_idx" not in st.session_state:
        st.session_state.route_idx = 0  # hops revealed

init_state()

# ---------------- UI: active nodes ----------------
st.set_page_config(page_title="Chord 0..31 • Fingers + Route", layout="wide")
st.title("🔗 Chord DHT — Fixed Ring 0..31 (with step-by-step lookup)")

st.markdown(
    "IDs **0..31** are always drawn. IDs you do **not** list are shown as **disabled** placeholders "
    "(grey hollow circles). Finger tables and routing use **only your active nodes**."
)

mode = st.radio("Active-node input", ["Manual IDs (0..31)", "Hash labels → IDs (mod 32)"], index=0)

if mode == "Manual IDs (0..31)":
    ids_text = st.text_area("Active node IDs (comma/space separated)",
                            value="1, 4, 9, 11, 14, 18, 20, 21, 28")
    raw = [t.strip() for t in ids_text.replace(",", " ").split()]
    try:
        active_nodes = sorted(set(int(x) % SPACE for x in raw if x != ""))
    except ValueError:
        st.error("Please enter only integers.")
        active_nodes = []
else:
    labels_text = st.text_area("Node labels (one per line)", value="nodeA\nnodeB\nnodeC\nnodeD")
    labels = [s.strip() for s in labels_text.splitlines() if s.strip()]
    node_map = {lbl: sha1_mod(lbl, SPACE) for lbl in labels}
    active_nodes = sorted(set(node_map.values()))
    st.markdown("**Hashing:**")
    st.latex(r"\text{node\_id} = \operatorname{SHA1}(\text{label}) \bmod 32")

if not active_nodes:
    st.warning("Add at least one active node.")
    st.stop()

# ---------------- Fingers (step-by-step) ----------------
st.header("Step 1 — Finger table (reveal per entry)")
selected = st.selectbox("Selected node (draw fingers from here)", options=active_nodes, index=0)
fingers_all = build_finger_table(selected, active_nodes, M)

c1, c2, c3 = st.columns([1,1,6])
with c1:
    if st.button("Reset entries"):
        st.session_state.k = 0
with c2:
    if st.button("Next entry"):
        st.session_state.k = min(M, st.session_state.k + 1)

k = st.session_state.k
fingers_shown = fingers_all[:k]
current_start = fingers_shown[-1].start if k > 0 else None

st.subheader("📇 Finger table (revealed so far)")
df_ft = pd.DataFrame([{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in fingers_shown],
                     columns=["i", "start", "successor"])
st.dataframe(df_ft, use_container_width=True, hide_index=True)

st.markdown("**Definitions (m=5):**")
st.latex(r"\text{start}[i] = (n + 2^{i-1}) \bmod 32")
st.latex(r"\text{finger}[i] = \operatorname{succ}(\text{start}[i])")

if k > 0:
    fe = fingers_shown[-1]
    st.markdown("**Current step:**")
    st.latex(rf"n = {selected}")
    st.latex(rf"\text{{start}}[{fe.i}] = ({selected} + 2^{{{fe.i-1}}}) \bmod 32 = {fe.start}")
    st.latex(rf"\text{{finger}}[{fe.i}] = \operatorname{{succ}}({fe.start}) = {fe.node}")
else:
    st.info("Click **Next entry** to compute finger[1].")

# ---------------- Lookup (search the data / find the route) ----------------
st.header("Step 2 — Search data / Find the route (step-by-step)")

colA, colB, colC = st.columns([1.5,1.5,6])
with colA:
    start_node = st.selectbox("Start node", options=active_nodes, index=0, key="start_node")
with colB:
    key_id = st.number_input("Key (data) ID", min_value=0, max_value=31, value=26, step=1, key="key_id")

succ_k = successor_of(key_id, active_nodes)
st.markdown("**Key responsibility:**")
st.latex(rf"\operatorname{{succ}}(k) = {succ_k}")

# Controls: build full route, then reveal hop-by-hop
cA, cB, cC = st.columns([1,1,6])
with cA:
    if st.button("Start lookup (reset)"):
        path, reasons = chord_lookup_full(start_node, key_id, active_nodes, M)
        st.session_state.route_path = path
        st.session_state.route_reasons = reasons
        st.session_state.route_idx = 0
with cB:
    if st.button("Next hop"):
        if st.session_state.route_path:
            st.session_state.route_idx = min(
                len(st.session_state.route_path) - 1,
                st.session_state.route_idx + 1
            )

# If route not initialized yet, compute once from defaults
if not st.session_state.route_path:
    path, reasons = chord_lookup_full(start_node, key_id, active_nodes, M)
    st.session_state.route_path = path
    st.session_state.route_reasons = reasons
    st.session_state.route_idx = 0

route_path = st.session_state.route_path
route_reasons = st.session_state.route_reasons
route_hops_to_show = st.session_state.route_idx  # number of edges to draw

# ---------------- Draw everything together ----------------
fig = ring_figure(
    active_nodes=active_nodes,
    selected=selected,
    fingers=fingers_shown,
    highlight_start=current_start,
    show_radial=True,
    pin_selected_ft=True,
    route_path=route_path,
    route_hops_to_show=route_hops_to_show,
    key=key_id
)
st.plotly_chart(fig, use_container_width=True)

# Route details
st.subheader("🧭 Route")
st.code(" → ".join(str(n) for n in route_path), language="text")

st.subheader("📐 Hop-by-hop reasoning")
# Reveal reasons matching the number of drawn hops (each hop adds one reason; final reason on arrival)
max_to_show = min(route_hops_to_show + 1, len(route_reasons))
for i in range(max_to_show):
    st.latex(route_reasons[i])
