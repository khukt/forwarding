# mini_allocation.py
import hashlib
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def sha1_mod(s: str, space: int) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h, 16) % space

def successor_of(x: int, sorted_nodes: List[int]) -> int:
    for n in sorted_nodes:
        if n >= x:
            return n
    return sorted_nodes[0]

def node_xy(id_val: int, space: int, radius: float = 1.0) -> Tuple[float, float]:
    theta = 2 * math.pi * (id_val / space)
    return radius * math.cos(theta), radius * math.sin(theta)

def ring_figure(space: int, node_map: dict, key_map: dict) -> go.Figure:
    # circle
    circle_angles = np.linspace(0, 2*np.pi, 361)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(circle_angles), y=np.sin(circle_angles),
                             mode="lines", name="Ring", hoverinfo="skip"))

    # nodes
    if node_map:
        xs, ys, labels = [], [], []
        for lbl, nid in node_map.items():
            x, y = node_xy(nid, space)
            xs.append(x); ys.append(y); labels.append(f"{lbl} ({nid})")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=[f"{node_map[lbl]}" for lbl in node_map],
            textposition="top center", name="Nodes",
            marker=dict(size=12, line=dict(width=1, color="white")),
            hovertext=labels, hoverinfo="text"
        ))

    # keys
    if key_map:
        xs, ys, labels = [], [], []
        for lbl, kid in key_map.items():
            x, y = node_xy(kid, space, radius=0.85)
            xs.append(x); ys.append(y); labels.append(f"{lbl} ({kid})")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=[f"{key_map[lbl]}" for lbl in key_map],
            textposition="bottom center", name="Keys",
            marker=dict(size=10, symbol="diamond"),
            hovertext=labels, hoverinfo="text"
        ))

    fig.update_layout(
        width=640, height=640, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white",
        title="Small Chord Hash Ring: Node & Key Allocation",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

st.set_page_config(page_title="Small Hash Ring Allocation", layout="centered")
st.title("🧭 Small Hash Ring: Node Allocation & Key Responsibility")

with st.sidebar:
    m = st.slider("m (identifier bits)", min_value=3, max_value=6, value=4, help="ID space size is 2^m")
    space = 2 ** m

    st.write("### Nodes (labels; one per line)")
    nodes_text = st.text_area("Examples:\nnodeA\nnodeB\nnodeC", value="nodeA\nnodeB\nnodeC")
    st.write("### Keys (labels; one per line)")
    keys_text = st.text_area("Examples:\nfile1\nfile2", value="file1\nfile2")

# Parse labels
node_labels = [s.strip() for s in nodes_text.splitlines() if s.strip()]
key_labels = [s.strip() for s in keys_text.splitlines() if s.strip()]

# Hash → IDs
node_map = {lbl: sha1_mod(lbl, space) for lbl in node_labels}
key_map  = {lbl: sha1_mod(lbl, space) for lbl in key_labels}

# Sort node IDs for successor logic
sorted_nodes = sorted(node_map.values())
id_to_label_multi = {}
for lbl, nid in node_map.items():
    id_to_label_multi.setdefault(nid, []).append(lbl)

# Responsibility table
rows = []
for k_lbl, kid in key_map.items():
    if not sorted_nodes:
        resp = None
    else:
        resp = successor_of(kid, sorted_nodes)
    resp_labels = ", ".join(id_to_label_multi.get(resp, [])) if resp is not None else "-"
    rows.append({
        "key_label": k_lbl,
        "key_id": kid,
        "responsible_node_id": resp if resp is not None else "-",
        "responsible_node_labels": resp_labels if resp is not None else "-"
    })
df = pd.DataFrame(rows, columns=["key_label", "key_id", "responsible_node_id", "responsible_node_labels"])

# Show ring and tables
fig = ring_figure(space, node_map, key_map)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("📍 Node placements")
    if node_map:
        df_nodes = pd.DataFrame(
            [{"node_label": lbl, "node_id": nid} for lbl, nid in sorted(node_map.items(), key=lambda x: x[1])],
            columns=["node_label", "node_id"]
        )
        st.dataframe(df_nodes, use_container_width=True, hide_index=True)
    else:
        st.info("Add some node labels in the sidebar.")

with right:
    st.subheader("🔑 Key responsibility")
    if not node_map:
        st.warning("Add at least one node to compute responsibility.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
with st.expander("ℹ️ What this shows"):
    st.markdown(f"""
- **Ring size**: `2^{m} = {space}` → IDs `0..{space-1}`  
- **Node ID** = `SHA1(node_label) % {space}`  
- **Key ID** = `SHA1(key_label) % {space}`  
- A key is stored at the **first node clockwise** whose ID ≥ key ID (wraps to 0 if none).
- This app is kept *small* so students can see collisions and wrap-around clearly.
""")
