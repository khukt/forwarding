# app_fixed32_figstyle.py
# Chord DHT — Fixed ring 0..31 (m=5). All 32 positions are drawn.
# Any ID not listed by you is shown as a "disabled" placeholder (grey hollow dot).
# Finger tables are computed using ONLY the active nodes.

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- Fixed ring 0..31 ----
M = 5
SPACE = 2 ** M
ALL_POSITIONS = list(range(SPACE))

# ---------------- Core math ----------------
def sha1_mod(s: str, space: int) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h, 16) % space

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
    space = 2 ** m
    entries = []
    if not nodes_sorted:
        return entries
    for i in range(1, m + 1):
        start = (n + 2 ** (i - 1)) % space
        succ = successor_of(start, nodes_sorted)
        entries.append(FingerEntry(i=i, start=start, node=succ))
    return entries

# -------------- Visualization --------------
def node_xy(id_val: int, space: int, radius: float = 1.0) -> Tuple[float, float]:
    theta = 2 * math.pi * (id_val / space)
    return radius * math.cos(theta), radius * math.sin(theta)

def ring_figure(
    active_nodes: List[int],
    selected: int | None,
    fingers: List[FingerEntry] | None,
    highlight_start: int | None,
    show_radial: bool,
    pin_selected_ft: bool
) -> go.Figure:
    R = 1.0
    circle_angles = np.linspace(0, 2*np.pi, 361)
    fig = go.Figure()

    # Base ring
    fig.add_trace(go.Scatter(x=np.cos(circle_angles), y=np.sin(circle_angles),
                             mode="lines", name="Ring", hoverinfo="skip"))

    active_set = set(active_nodes)
    disabled_positions = [i for i in ALL_POSITIONS if i not in active_set]

    # Disabled placeholders (grey, hollow)
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

    # Active nodes (solid)
    if active_nodes:
        xs, ys, sizes, colors, labels = [], [], [], [], []
        for nid in active_nodes:
            x, y = node_xy(nid, SPACE, R)
            xs.append(x); ys.append(y)
            if selected == nid:
                sizes.append(16); colors.append("crimson"); labels.append(f"{nid} (selected)")
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

    # Pin selected node's finger table as a small card
    if pin_selected_ft and selected is not None and fingers:
        x, y = node_xy(selected, SPACE, R*1.12)
        table = "i  start  succ<br>" + "<br>".join(f"{fe.i}  {fe.start:<5}  {fe.node}" for fe in fingers)
        fig.add_annotation(
            x=x, y=y, xanchor="left", yanchor="middle",
            text=f"<b>Node {selected}</b><br><span style='font-family:monospace'>{table}</span>",
            showarrow=False, bordercolor="black", borderwidth=1, bgcolor="white", opacity=0.95
        )

    fig.update_layout(
        width=860, height=860,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        title="Chord • Fixed Ring 0..31 • Active vs Disabled (placeholders)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# -------------- App (single page) --------------
st.set_page_config(page_title="Chord DHT — Fixed 0..31 with placeholders", layout="wide")
st.title("🔗 Chord DHT — Fixed Ring 0..31 (disabled placeholders like the figure)")

st.markdown("**Ring:** \(m=5\Rightarrow 2^5=32\) → IDs **0..31** are always drawn. "
            "IDs you do **not** list are shown as **disabled** placeholders (grey hollow circles).")

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

# Selected node, reveal control
selected = st.selectbox("Selected node (draw fingers from here)", options=active_nodes, index=0)

# Build full finger table for selected, using ONLY active nodes
fingers_all = build_finger_table(selected, active_nodes, M)

# Reveal fingers gradually (like the figure’s step-by-step)
k = st.session_state.get("k", 0)
left, mid, right = st.columns([1,1,6])
with left:
    if st.button("Reset entries"):
        k = 0
with mid:
    if st.button("Next entry"):
        k = min(M, k + 1)
st.session_state["k"] = k

fingers_shown = fingers_all[:k]
current_start = fingers_shown[-1].start if k > 0 else None

pin_card = st.checkbox("Pin selected node's finger table card on the ring", value=True)
show_radial = st.checkbox("Show radial guide n → start[i]", value=True)

# Draw
fig = ring_figure(
    active_nodes=active_nodes,
    selected=selected,
    fingers=fingers_shown,
    highlight_start=current_start,
    show_radial=show_radial,
    pin_selected_ft=pin_card
)
st.plotly_chart(fig, use_container_width=True)

# Tables + equations
st.subheader("📇 Finger table (revealed so far)")
df_ft = pd.DataFrame([{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in fingers_shown],
                     columns=["i", "start", "successor"])
st.dataframe(df_ft, use_container_width=True, hide_index=True)

st.markdown("**Definitions (m=5):**")
st.latex(r"\text{start}[i] = (n + 2^{i-1}) \bmod 32")
st.latex(r"\text{finger}[i] = \operatorname{succ}(\text{start}[i])\ \text{(over active nodes)}")

if k == 0:
    st.info("Click **Next entry** to compute finger[1].")
else:
    fe = fingers_shown[-1]
    st.subheader("🧮 Current step")
    st.latex(rf"n = {selected}")
    st.latex(rf"\text{{start}}[{fe.i}] = ({selected} + 2^{{{fe.i-1}}}) \bmod 32 = {fe.start}")
    st.latex(rf"\text{{finger}}[{fe.i}] = \operatorname{{succ}}({fe.start}) = {fe.node}")

with st.expander("Show full finger table for selected (all 5 entries)"):
    df_full = pd.DataFrame([{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in fingers_all])
    st.dataframe(df_full, use_container_width=True, hide_index=True)
