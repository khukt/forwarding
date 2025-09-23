# app_small.py
# Chord DHT — small, fixed ring (0..31) for teaching (matches the style of the 0–31 diagram)

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- ring size fixed to 0..31 ----
M = 5
SPACE = 2 ** M  # 32

# ---------- helpers ----------
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

def build_finger_table(n: int, nodes_sorted: List[int]) -> List[FingerEntry]:
    entries = []
    for i in range(1, M + 1):
        start = (n + 2 ** (i - 1)) % SPACE
        succ = successor_of(start, nodes_sorted)
        entries.append(FingerEntry(i=i, start=start, node=succ))
    return entries

def finger_equations(n: int, entries: List[FingerEntry]) -> List[str]:
    lines = []
    for fe in entries:
        i = fe.i
        lines.append(
            rf"\text{{finger}}[{i}]:\; "
            rf"\text{{start}}[{i}] = ({n} + 2^{{{i-1}}}) \bmod 2^{{{M}}} = {fe.start},\; "
            rf"\text{{node}} = \operatorname{{succ}}({fe.start}) = {fe.node}"
        )
    return lines

def closest_preceding_finger(n: int, fingers: List[int], target: int) -> int:
    for f in reversed(fingers):
        if f != n and mod_interval_contains(n, target, f, SPACE, inclusive_right=False):
            return f
    return n

def chord_lookup_with_reasons(start_node: int, key: int, nodes_sorted: List[int], max_steps: int = 64):
    visited = [start_node]
    reasons: List[str] = []
    if len(nodes_sorted) == 0:
        return visited, [r"\text{No nodes in the ring.}"]
    succ_key = successor_of(key, nodes_sorted)

    finger_map: Dict[int, List[int]] = {
        n: [fe.node for fe in build_finger_table(n, nodes_sorted)]
        for n in nodes_sorted
    }

    while len(visited) < max_steps:
        curr = visited[-1]
        if curr == succ_key:
            reasons.append(rf"\mathbf{{Stop:}}\; \text{{current}}={curr}=\operatorname{{succ}}({key})")
            break

        curr_idx = nodes_sorted.index(curr)
        curr_succ = nodes_sorted[(curr_idx + 1) % len(nodes_sorted)]

        in_interval = mod_interval_contains(curr, curr_succ, key, SPACE, inclusive_right=True)
        if in_interval:
            reasons.append(
                rf"\text{{Since }} {key}\in({curr},{curr_succ}] \Rightarrow "
                rf"\text{{next}}=\operatorname{{succ}}({curr})={curr_succ}"
            )
            visited.append(curr_succ)
            if curr_succ == succ_key:
                reasons.append(rf"\mathbf{{Arrived}}\ \text{{at}}\ \operatorname{{succ}}({key})={succ_key}")
                break
            continue

        cpf = closest_preceding_finger(curr, finger_map[curr], key)
        if cpf == curr:
            reasons.append(
                rf"\text{{No finger in }}({curr},{key}) \Rightarrow "
                rf"\text{{fallback to }} \operatorname{{succ}}({curr})={curr_succ}"
            )
            visited.append(curr_succ)
        else:
            reasons.append(
                rf"\text{{Choose closest preceding finger of }}{curr}\ \text{{toward }}{key}: "
                rf"{cpf}\in({curr},{key})"
            )
            visited.append(cpf)

        if visited[-1] == succ_key:
            reasons.append(rf"\mathbf{{Arrived}}\ \text{{at}}\ \operatorname{{succ}}({key})={succ_key}")
            break

    return visited, reasons

# ---------- viz ----------
def node_xy(id_val: int, radius: float = 1.0) -> Tuple[float, float]:
    theta = 2 * math.pi * (id_val / SPACE)
    return radius * math.cos(theta), radius * math.sin(theta)

def ring_figure(nodes_sorted: List[int],
                selected: int = None,
                key: int = None,
                finger_entries: List[FingerEntry] = None,
                lookup_path: List[int] = None) -> go.Figure:
    R = 1.0
    circle_angles = np.linspace(0, 2*np.pi, 361)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(circle_angles), y=np.sin(circle_angles),
                             mode="lines", name="Hash ring", hoverinfo="skip"))

    succ_key = successor_of(key, nodes_sorted) if (key is not None and nodes_sorted) else None

    xs, ys, labels, colors, sizes = [], [], [], [], []
    for nid in nodes_sorted:
        x, y = node_xy(nid, R)
        xs.append(x); ys.append(y)
        label = f"{nid}"
        if selected is not None and nid == selected:
            colors.append("crimson"); sizes.append(14); label += " (selected)"
        elif succ_key is not None and nid == succ_key:
            colors.append("orange"); sizes.append(12); label += " (succ(key))"
        else:
            colors.append("royalblue"); sizes.append(10)
        labels.append(label)

    if nodes_sorted:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=[str(n) for n in nodes_sorted],
            textposition="top center",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
            hovertext=labels, hoverinfo="text", name="Nodes"
        ))

    if selected is not None and finger_entries:
        sx, sy = node_xy(selected, R)
        for fe in finger_entries:
            tx, ty = node_xy(fe.node, R)
            fig.add_trace(go.Scatter(
                x=[sx, tx], y=[sy, ty], mode="lines",
                line=dict(width=2, dash="dot"),
                name=f"finger[{fe.i}]→{fe.node}",
                hovertext=f"finger[{fe.i}] start={fe.start} → {fe.node}",
                hoverinfo="text",
                showlegend=False
            ))

    if lookup_path and len(lookup_path) > 1:
        for i in range(len(lookup_path) - 1):
            ax, ay = node_xy(lookup_path[i], R)
            bx, by = node_xy(lookup_path[i+1], R)
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

    fig.update_layout(
        width=820, height=820,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        title="Chord (m=5) • Ring 0..31 • Finger Table • Lookup",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# ---------- UI ----------
st.set_page_config(page_title="Chord DHT • Small Ring (0..31)", layout="wide")
st.title("🔗 Chord DHT — Small Ring (0..31)")
st.caption("Use Manual IDs to reproduce textbook figures exactly; or Hash mode to demonstrate mapping.")

with st.sidebar:
    mode = st.radio("Node placement mode",
                    ["Manual IDs (0..31)", "Hash labels → IDs (mod 32)"],
                    index=0)
    if mode == "Manual IDs (0..31)":
        ids_text = st.text_area("Node IDs (comma/space separated)",
                                value="1, 4, 9, 11, 14, 18, 20, 21, 28")
        raw = [t.strip() for t in ids_text.replace(",", " ").split()]
        try:
            nodes_sorted = sorted(set(int(x) % SPACE for x in raw if x != ""))
        except ValueError:
            st.error("Invalid IDs. Please enter integers in 0..31.")
            st.stop()
        st.markdown("**Ring size:**")
        st.latex(rf"2^{{{M}}} = {SPACE}\ \Rightarrow\ \text{{IDs}}=0..{SPACE-1}")
    else:
        node_labels_text = st.text_area("Node labels (1 per line)", value="nodeA\nnodeB\nnodeC\nnodeD")
        node_labels = [s.strip() for s in node_labels_text.splitlines() if s.strip()]
        node_map = {lbl: sha1_mod(lbl, SPACE) for lbl in node_labels}
        nodes_sorted = sorted(set(node_map.values()))
        st.markdown("**Hashing equation:**")
        st.latex(rf"\text{{node\_id}} = \operatorname{{SHA1}}(\text{{label}}) \bmod 2^{{{M}}} = 32")

    key_id = st.number_input("Key ID (0..31)", min_value=0, max_value=31, value=26, step=1)
    show_fingers = st.checkbox("Show selected node's finger chords", value=True)
    show_lookup = st.checkbox("Show lookup path & reasons", value=True)
    show_eq = st.checkbox("Show equations/explanations", value=True)

if not nodes_sorted:
    st.warning("Add at least two nodes to proceed.")
    st.stop()

selected_node = st.selectbox("Selected node (by ID)", options=nodes_sorted, index=0)

# Compute
finger_entries = build_finger_table(selected_node, nodes_sorted)
lookup_path, hop_reasons = ([], [])
if show_lookup:
    lookup_path, hop_reasons = chord_lookup_with_reasons(selected_node, key_id, nodes_sorted)

# Layout
left, right = st.columns([0.58, 0.42], gap="large")

with left:
    fig = ring_figure(
        nodes_sorted=nodes_sorted,
        selected=selected_node,
        key=key_id,
        finger_entries=finger_entries if show_fingers else None,
        lookup_path=lookup_path if show_lookup else None
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📍 Node placements")
    if mode == "Manual IDs (0..31)":
        df_nodes = pd.DataFrame([{"node_id": n} for n in nodes_sorted])
    else:
        df_nodes = pd.DataFrame(
            [{"node_label": lbl, "node_id": nid} for lbl, nid in sorted(node_map.items(), key=lambda x: x[1])],
            columns=["node_label", "node_id"]
        )
    st.dataframe(df_nodes, use_container_width=True, hide_index=True)

    st.subheader("📇 Finger table (selected node)")
    df_ft = pd.DataFrame(
        [{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in finger_entries],
        columns=["i", "start", "successor"]
    )
    st.dataframe(df_ft, use_container_width=True, hide_index=True)

    if show_eq:
        st.markdown("**Finger definition (m=5):**")
        st.latex(rf"\text{{start}}[i] = (n + 2^{{i-1}}) \bmod 2^{{{M}}}")
        st.latex(r"\text{finger}[i] = \operatorname{succ}(\text{start}[i])")
        with st.expander("Show per-entry substitutions"):
            for line in finger_equations(selected_node, finger_entries):
                st.latex(line)

    st.subheader("🧭 Lookup")
    succ_k = successor_of(key_id, nodes_sorted)
    st.markdown("**Summary:**")
    st.latex(rf"\text{{Target key }}k={key_id}\,,\quad \operatorname{{succ}}(k)={succ_k}")
    if lookup_path:
        st.code(" → ".join(str(n) for n in lookup_path), language="text")
    if show_eq and hop_reasons:
        with st.expander("Show hop-by-hop reasoning"):
            for r in hop_reasons:
                st.latex(r)

st.divider()
with st.expander("📘 Teaching tips"):
    st.markdown("""
- Use **Manual IDs** to reproduce the 0–31 example exactly.
- Start with a few nodes (e.g., `1, 4, 9, 11, 14, 18, 20, 21, 28`) to match textbook figures.
- Ask students to compute `start[i]` by hand for one node, then reveal with the expander.
- Try lookups like **k = 12** from node **28** and **k = 26** from node **1**, mirroring the diagram.
""")
