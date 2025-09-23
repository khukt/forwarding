# app.py
# Chord DHT — teaching-first visualizer with equation updates
# ------------------------------------------------------------
# Features:
# 1) Small ID space (2^m) with SHA-1 % 2^m allocation (nodes & keys)
# 2) Finger table building with fully-expanded equations
# 3) Lookup path with per-hop reasoning & equations
# 4) Interactive ring visual using Plotly

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------
# Helpers: hashing & modulo math
# ----------------------------
def sha1_mod(s: str, space: int) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h, 16) % space

def mod_interval_contains(a: int, b: int, x: int, m: int, inclusive_right: bool = False) -> bool:
    """
    True iff x is in interval (a, b] if inclusive_right else (a, b) on modulo-m ring (clockwise).
    """
    if a == b:
        return inclusive_right  # whole ring or empty
    if a < b:
        return (a < x <= b) if inclusive_right else (a < x < b)
    # wrapped case: a > b
    return ((a < x <= m - 1) or (0 <= x <= b)) if inclusive_right else ((a < x < m) or (0 <= x < b))

def successor_of(x: int, nodes_sorted: List[int]) -> int:
    for n in nodes_sorted:
        if n >= x:
            return n
    return nodes_sorted[0]  # wrap


# ----------------------------
# Finger table
# ----------------------------
@dataclass
class FingerEntry:
    i: int
    start: int
    node: int

def build_finger_table(n: int, nodes_sorted: List[int], m: int) -> List[FingerEntry]:
    space = 2 ** m
    entries = []
    for i in range(1, m + 1):
        start = (n + 2 ** (i - 1)) % space
        succ = successor_of(start, nodes_sorted)
        entries.append(FingerEntry(i=i, start=start, node=succ))
    return entries

def finger_equations(n: int, m: int, entries: List[FingerEntry]) -> List[str]:
    """
    Pretty LaTeX for each finger: start[i] = (n + 2^{i-1}) mod 2^m = value; finger[i] = succ(start[i]) = node
    """
    space = 2 ** m
    lines = []
    for fe in entries:
        i = fe.i
        start_expr = f"({n} + 2^{{{i-1}}}) \\bmod 2^{{{m}}}"
        start_val = fe.start
        succ_val = fe.node
        lines.append(
            rf"\[\text{{finger}}[{i}]:\; "
            rf"\text{{start}}[{i}] = {start_expr} = {start_val},\; "
            rf"\text{{node}} = \operatorname{{succ}}({start_val}) = {succ_val}\]"
        )
    return lines


# ----------------------------
# Lookup (iterative Chord)
# ----------------------------
def closest_preceding_finger(n: int, fingers: List[int], target: int, m: int) -> int:
    for f in reversed(fingers):
        if f != n and mod_interval_contains(n, target, f, 2 ** m, inclusive_right=False):
            return f
    return n

def chord_lookup_with_reasons(start_node: int, key: int, nodes_sorted: List[int], m: int, max_steps: int = 64):
    """
    Return (visited_nodes, justification_lines[])
    Each hop has a short equation/interval explanation for teaching.
    """
    space = 2 ** m
    visited = [start_node]
    reasons = []
    if len(nodes_sorted) == 0:
        return visited, ["No nodes in the ring."]
    succ_key = successor_of(key, nodes_sorted)

    # Precompute finger lists (targets only) for all nodes to keep it simple
    finger_map: Dict[int, List[int]] = {n: [fe.node for fe in build_finger_table(n, nodes_sorted, m)]
                                        for n in nodes_sorted}

    while len(visited) < max_steps:
        curr = visited[-1]
        if curr == succ_key:
            reasons.append(rf"\(\textbf{{Stop:}}\; \text{{current}}={curr}=\operatorname{{succ}}({key})\).")
            break

        # immediate successor of curr on ring
        curr_idx = nodes_sorted.index(curr)
        curr_succ = nodes_sorted[(curr_idx + 1) % len(nodes_sorted)]

        # Check if key in (curr, curr_succ]
        in_interval = mod_interval_contains(curr, curr_succ, key, space, inclusive_right=True)
        if in_interval:
            reasons.append(
                rf"\(\text{{Since }} {key}\in({curr},{curr\_succ}] \Rightarrow \text{{next}}=\text{{successor}}({curr})={curr\_succ}\)."
                .replace("curr_succ", "succ("+str(curr)+")")
            )
            visited.append(curr_succ)
            if curr_succ == succ_key:
                reasons.append(rf"\(\textbf{{Arrived at }} \operatorname{{succ}}({key})={succ_key}\).")
                break
            continue

        # Otherwise choose closest preceding finger
        cpf = closest_preceding_finger(curr, finger_map[curr], key, m)
        if cpf == curr:
            # fallback to successor to ensure progress
            reasons.append(
                rf"\(\text{{No finger in }}({curr},{key}) \Rightarrow \text{{fallback to }} \operatorname{{succ}}({curr})={curr_succ}\)."
            )
            visited.append(curr_succ)
        else:
            reasons.append(
                rf"\(\text{{Choose closest preceding finger of }}{curr}\text{ toward }{key}: {cpf}\in({curr},{key})\)."
            )
            visited.append(cpf)

        if visited[-1] == succ_key:
            reasons.append(rf"\(\textbf{{Arrived at }} \operatorname{{succ}}({key})={succ_key}\).")
            break

    return visited, reasons


# ----------------------------
# Visualization
# ----------------------------
def node_xy(id_val: int, space: int, radius: float = 1.0) -> Tuple[float, float]:
    theta = 2 * math.pi * (id_val / space)
    return radius * math.cos(theta), radius * math.sin(theta)

def ring_figure(space: int,
                nodes_sorted: List[int],
                selected: int = None,
                key: int = None,
                finger_entries: List[FingerEntry] = None,
                lookup_path: List[int] = None) -> go.Figure:
    R = 1.0
    circle_angles = np.linspace(0, 2*np.pi, 361)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(circle_angles), y=np.sin(circle_angles),
                             mode="lines", name="Hash ring", hoverinfo="skip"))

    # nodes
    xs, ys, labels, colors, sizes = [], [], [], [], []
    succ_key = None
    if key is not None and nodes_sorted:
        succ_key = successor_of(key, nodes_sorted)
    for nid in nodes_sorted:
        x, y = node_xy(nid, space, R)
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

    # fingers from selected node
    if selected is not None and finger_entries:
        sx, sy = node_xy(selected, space, R)
        for fe in finger_entries:
            tx, ty = node_xy(fe.node, space, R)
            fig.add_trace(go.Scatter(
                x=[sx, tx], y=[sy, ty], mode="lines",
                line=dict(width=2, dash="dot"),
                name=f"finger[{fe.i}]→{fe.node}",
                hovertext=f"finger[{fe.i}] start={fe.start} → {fe.node}",
                hoverinfo="text",
                showlegend=False
            ))

    # lookup path
    if lookup_path and len(lookup_path) > 1:
        for i in range(len(lookup_path) - 1):
            ax, ay = node_xy(lookup_path[i], space, R)
            bx, by = node_xy(lookup_path[i+1], space, R)
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
        title="Chord Hash Ring • Allocation • Finger Table • Lookup",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Chord DHT • Complete Teaching Demo", layout="wide")
st.title("🔗 Chord DHT — Complete Teaching Demo")
st.caption("Small hash ring, explicit equations, and interactive visualization to explain the full process.")

with st.sidebar:
    st.header("⚙️ Identifier Space")
    m = st.slider("m (identifier bits)", min_value=3, max_value=10, value=4,
                  help="ID space size is 2^m (keep small for easier teaching).")
    space = 2 ** m

    st.header("🧩 Nodes & Keys")
    st.write("Enter labels (1 per line). Each label is hashed by SHA-1 and reduced modulo 2^m.")
    node_labels_text = st.text_area("Node labels", value="nodeA\nnodeB\nnodeC\nnodeD")
    key_labels_text  = st.text_area("Key labels", value="file1\nindex\nreadme")

    st.header("🎯 Focus")
    show_fingers = st.checkbox("Show selected node's finger chords", value=True)
    show_lookup  = st.checkbox("Show lookup path & reasons", value=True)
    show_eq      = st.checkbox("Show all equation steps", value=True)

# Parse labels
node_labels = [s.strip() for s in node_labels_text.splitlines() if s.strip()]
key_labels  = [s.strip() for s in key_labels_text.splitlines() if s.strip()]

# Allocation
node_map = {lbl: sha1_mod(lbl, space) for lbl in node_labels}
key_map  = {lbl: sha1_mod(lbl, space) for lbl in key_labels}
nodes_sorted = sorted(set(node_map.values()))
id_to_node_labels = {}
for lbl, nid in node_map.items():
    id_to_node_labels.setdefault(nid, []).append(lbl)

# UI: choose selected node & key (only if available)
selected_node = nodes_sorted[0] if nodes_sorted else None
if nodes_sorted:
    selected_node = st.selectbox("Selected node (by ID)", options=nodes_sorted, index=0)
key_id = None
if key_map:
    # let user pick which key drives the lookup
    key_options = [f"{k} (id={kid})" for k, kid in key_map.items()]
    chosen = st.selectbox("Lookup target key", options=key_options, index=0)
    # parse id back
    key_id = int(chosen.split("id=")[-1].rstrip(")"))

# Compute finger table & lookup
finger_entries = build_finger_table(selected_node, nodes_sorted, m) if selected_node is not None else []
lookup_path, hop_reasons = ([], [])
if show_lookup and (selected_node is not None) and (key_id is not None):
    lookup_path, hop_reasons = chord_lookup_with_reasons(selected_node, key_id, nodes_sorted, m)

# =========== Layout ===========
left, right = st.columns([0.58, 0.42], gap="large")

# Left: Figure
with left:
    fig = ring_figure(
        space=space,
        nodes_sorted=nodes_sorted,
        selected=selected_node,
        key=key_id,
        finger_entries=finger_entries if show_fingers else None,
        lookup_path=lookup_path if show_lookup else None
    )
    st.plotly_chart(fig, use_container_width=True)

# Right: Tables & Equations
with right:
    st.subheader("📍 Node Allocation (SHA-1 → ID)")
    if node_map:
        df_nodes = pd.DataFrame(
            [{"node_label": lbl, "node_id": nid} for lbl, nid in sorted(node_map.items(), key=lambda x: x[1])],
            columns=["node_label", "node_id"]
        )
        st.dataframe(df_nodes, use_container_width=True, hide_index=True)
        if show_eq:
            st.markdown("**Equation:**  \n"
                        rf"\(\text{{node\_id}} = \operatorname{{SHA1}}(\text{{label}}) \bmod 2^{{{m}}}\)")
            with st.expander("Show per-node numeric substitutions"):
                for lbl, nid in sorted(node_map.items(), key=lambda x: x[1]):
                    hhex = hashlib.sha1(lbl.encode("utf-8")).hexdigest()
                    st.latex(
                        rf"\text{{{lbl}}}:"
                        rf"\quad \operatorname{{SHA1}}({lbl}) = \texttt{{{hhex[:10]}}}\dots "
                        rf"\Rightarrow \text{{int}} \bmod 2^{{{m}}} = {nid}"
                    )
    else:
        st.info("Add some node labels in the sidebar to see their placements.")

    st.subheader("🔑 Key Allocation & Responsibility")
    if key_map and nodes_sorted:
        rows = []
        for k_lbl, kid in key_map.items():
            resp = successor_of(kid, nodes_sorted)
            rows.append({
                "key_label": k_lbl,
                "key_id": kid,
                "responsible_node_id (succ(key))": resp,
                "responsible_node_labels": ", ".join(id_to_node_labels.get(resp, [])),
            })
        df_keys = pd.DataFrame(rows)
        st.dataframe(df_keys, use_container_width=True, hide_index=True)
        if show_eq:
            st.markdown("**Responsibility rule:**  \n"
                        rf"\(\operatorname{{succ}}(k) = \min\{{ n \in \text{{nodes}} \mid n \ge k \}}\)"
                        rf"\ \text{{(wrap to 0 if none)}}.")
    elif key_map and not nodes_sorted:
        st.warning("Add nodes first to compute responsibility.")

    st.subheader("📇 Finger Table (selected node)")
    if selected_node is not None:
        df_ft = pd.DataFrame(
            [{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in finger_entries],
            columns=["i", "start", "successor"]
        )
        st.dataframe(df_ft, use_container_width=True, hide_index=True)
        if show_eq:
            st.markdown("**Definition:**  \n"
                        rf"\(\text{{start}}[i] = (n + 2^{{i-1}}) \bmod 2^{{{m}}},\quad "
                        rf"\text{{finger}}[i] = \operatorname{{succ}}(\text{{start}}[i])\)")
            with st.expander("Show per-entry substitutions"):
                for line in finger_equations(selected_node, m, finger_entries):
                    st.markdown(line)

    st.subheader("🧭 Lookup Steps & Reasons")
    if show_lookup and key_id is not None and selected_node is not None and nodes_sorted:
        succ_k = successor_of(key_id, nodes_sorted)
        st.markdown(
            rf"**Target key:** \(k={key_id}\) &nbsp;&nbsp; "
            rf"**Responsible node:** \(\operatorname{{succ}}(k) = {succ_k}\)"
        )
        if lookup_path:
            st.code(" → ".join(str(n) for n in lookup_path), language="text")
        if show_eq and hop_reasons:
            with st.expander("Show hop-by-hop reasoning"):
                for r in hop_reasons:
                    st.markdown(r)
    elif show_lookup:
        st.info("Pick at least one node and one key to run a lookup.")

st.divider()
with st.expander("📘 Teaching tips & suggested exercises"):
    st.markdown(
        f"""
- Keep **m small** (e.g., {m}) so students can see **wrap-around** and **collisions**.
- Ask students to **predict finger entries** before revealing the equations.
- Change the selected node and compare its finger table with others; discuss **coverage** and **O(log N)** routing.
- Try sparse placements (e.g., 2–3 nodes only) and walk through the **interval checks** during lookup.
- Add/remove a node label; ask which keys **change responsibility** and **why**.
        """
    )
