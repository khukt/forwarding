# app.py
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Utilities for circular math
# ----------------------------
def mod_interval_contains(a: int, b: int, x: int, m: int, inclusive_right: bool = False) -> bool:
    """
    Return True iff x is in interval (a, b] on a modulo-m ring (if inclusive_right=True)
    or (a, b) if inclusive_right=False. Intervals are clockwise from a to b.
    """
    if a == b:
        return inclusive_right  # full ring if inclusive_right else empty
    if a < b:
        if inclusive_right:
            return a < x <= b
        else:
            return a < x < b
    else:
        # wrapped case
        if inclusive_right:
            return (a < x <= m - 1) or (0 <= x <= b)
        else:
            return (a < x < m) or (0 <= x < b)


def successor_of(x: int, nodes: List[int], m: int) -> int:
    """First node >= x clockwise (wraps around)."""
    for n in nodes:
        if n >= x:
            return n
    return nodes[0]  # wrap


def closest_preceding_finger(n: int, fingers: List[int], target: int, m: int) -> int:
    """Closest finger of n that strictly precedes target in (n, target). Fallback: n."""
    for f in reversed(fingers):
        if f != n and mod_interval_contains(n, target, f, m, inclusive_right=False):
            return f
    return n


# ----------------------------
# Chord structures
# ----------------------------
@dataclass
class FingerEntry:
    i: int
    start: int
    node: int

def build_finger_table(n: int, nodes: List[int], m: int) -> List[FingerEntry]:
    fingers = []
    space = 2 ** m
    for i in range(1, m + 1):
        start = (n + 2 ** (i - 1)) % space
        succ = successor_of(start, nodes, space)
        fingers.append(FingerEntry(i=i, start=start, node=succ))
    return fingers

def chord_lookup_path(start_node: int, key: int, nodes: List[int], m: int, max_steps: int = 64) -> List[int]:
    """
    Simulate Chord's iterative lookup from start_node to key.
    Returns the sequence of visited nodes.
    """
    visited = [start_node]
    space = 2 ** m
    # Precompute finger table for each node (small networks—fine for teaching)
    finger_map = {n: [fe.node for fe in build_finger_table(n, nodes, m)] for n in nodes}
    while len(visited) < max_steps:
        curr = visited[-1]
        succ = successor_of(key, nodes, space)
        if curr == succ:
            break
        # If key in (curr, successor(curr)] then next hop is successor(curr)
        curr_idx = nodes.index(curr)
        next_idx = (curr_idx + 1) % len(nodes)
        curr_succ = nodes[next_idx]
        if mod_interval_contains(curr, curr_succ, key, space, inclusive_right=True):
            visited.append(curr_succ)
            if curr_succ == succ:
                break
            continue
        # Else pick closest preceding finger
        nxt = closest_preceding_finger(curr, finger_map[curr], key, space)
        if nxt == curr:
            # fallback to immediate successor to make progress
            visited.append(curr_succ)
        else:
            visited.append(nxt)
        if visited[-1] == succ:
            break
    return visited


# ----------------------------
# Visualization helpers
# ----------------------------
def node_xy(id_val: int, m: int, radius: float = 1.0) -> Tuple[float, float]:
    """Position node on unit circle by id."""
    space = 2 ** m
    theta = 2 * math.pi * (id_val / space)
    return radius * math.cos(theta), radius * math.sin(theta)

def arc_points(a: int, b: int, m: int, steps: int = 50, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clockwise arc from a -> b on the circle, returning arrays of x,y points.
    """
    space = 2 ** m
    if a <= b:
        ids = np.linspace(a, b, steps)
    else:
        # wrap around
        forward = np.linspace(a, space, steps // 2, endpoint=False)
        wrap = np.linspace(0, b, steps // 2 + 1)
        ids = np.concatenate([forward, wrap])
    thetas = 2 * np.pi * (ids / space)
    return np.cos(thetas), np.sin(thetas)

def ring_figure(nodes: List[int], m: int, selected: int = None, key: int = None,
                finger_table: List[FingerEntry] = None, lookup_path: List[int] = None) -> go.Figure:
    R = 1.0
    # Base ring
    circle_angles = np.linspace(0, 2 * np.pi, 361)
    ring_x = np.cos(circle_angles)
    ring_y = np.sin(circle_angles)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ring_x, y=ring_y, mode="lines", name="Hash ring", hoverinfo="skip"))

    # Nodes
    xs, ys, texts, colors, sizes = [], [], [], [], []
    for nid in nodes:
        x, y = node_xy(nid, m, R)
        xs.append(x); ys.append(y)
        label = f"{nid}"
        if nid == selected:
            colors.append("crimson"); sizes.append(14); label += " (selected)"
        elif key is not None and successor_of(key, nodes, 2 ** m) == nid:
            colors.append("orange"); sizes.append(12); label += " (succ(key))"
        else:
            colors.append("royalblue"); sizes.append(10)
        texts.append(label)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=[str(n) for n in nodes],
        textposition="top center", marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
        name="Nodes", hovertext=texts, hoverinfo="text"
    ))

    # Draw finger table chords from selected node
    if selected is not None and finger_table:
        sx, sy = node_xy(selected, m, R)
        for fe in finger_table:
            tx, ty = node_xy(fe.node, m, R)
            # straight chord
            fig.add_trace(go.Scatter(
                x=[sx, tx], y=[sy, ty], mode="lines",
                line=dict(width=2, dash="dot"),
                name=f"finger[{fe.i}]→{fe.node}",
                hovertext=f"finger[{fe.i}] start={fe.start} → {fe.node}",
                hoverinfo="text",
                showlegend=False
            ))

    # Draw lookup path as directed segments
    if lookup_path and len(lookup_path) > 1:
        for i in range(len(lookup_path) - 1):
            ax, ay = node_xy(lookup_path[i], m, R)
            bx, by = node_xy(lookup_path[i + 1], m, R)
            fig.add_trace(go.Scatter(
                x=[ax, bx], y=[ay, by], mode="lines+markers",
                line=dict(width=3),
                marker=dict(size=6),
                name=f"hop {i+1}",
                hoverinfo="skip",
                showlegend=False
            ))
            # arrow head via annotation
            fig.add_annotation(
                x=bx, y=by, ax=ax, ay=ay, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2
            )

    # Aesthetic layout
    fig.update_layout(
        width=750, height=750,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        title="Chord Hash Ring • Nodes • Finger Table • Lookup Path",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Chord DHT Visualizer", layout="wide")

st.title("🔗 Chord DHT Visualizer")
st.caption("Interactive, classroom-friendly visualization of the Chord hash ring, finger tables, and lookups.")

with st.sidebar:
    st.header("⚙️ Parameters")
    m = st.slider("m (identifier bits)", min_value=4, max_value=16, value=8, help="ID space size is 2^m.")
    space = 2 ** m

    st.write("### Nodes")
    mode = st.radio("Node set", ["Random N nodes", "Custom IDs"], index=0)
    if mode == "Random N nodes":
        N = st.slider("Number of nodes", min_value=2, max_value=min(64, space), value=min(10, space))
        seed = st.number_input("Random seed", value=42, step=1)
        random.seed(seed)
        nodes = sorted(random.sample(range(space), N))
    else:
        ids_text = st.text_area(
            "Enter node IDs (comma/space separated)", value="1, 3, 6, 8, 12",
            help=f"Each must be in [0, {space-1}]."
        )
        raw = [t.strip() for t in ids_text.replace(",", " ").split()]
        try:
            nodes = sorted(set(int(x) % space for x in raw if x != ""))
        except ValueError:
            st.error("Invalid IDs. Please enter integers.")
            st.stop()
        if len(nodes) < 2:
            st.error("Need at least 2 nodes.")
            st.stop()

    st.divider()
    st.write("### Focus & Lookup")
    selected_node = st.selectbox("Selected node", options=nodes, index=0)
    key_id = st.number_input(f"Key ID (0..{space-1})", min_value=0, max_value=space - 1, value=0, step=1)
    show_lookup = st.checkbox("Show lookup path from selected node to key ID", value=True)

    st.divider()
    st.write("### Display Options")
    show_fingers = st.checkbox("Show selected node's finger table chords", value=True)

# Compute finger table & lookup path
finger = build_finger_table(selected_node, nodes, m)
path = chord_lookup_path(selected_node, key_id, nodes, m) if show_lookup else None

# Main layout
left, right = st.columns([0.6, 0.4], gap="large")

with left:
    fig = ring_figure(
        nodes=nodes,
        m=m,
        selected=selected_node,
        key=key_id,
        finger_table=finger if show_fingers else None,
        lookup_path=path
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📇 Finger Table (selected node)")
    df = pd.DataFrame(
        [{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in finger],
        columns=["i", "start", "successor"]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("🧭 Lookup Steps")
    if show_lookup:
        succ = successor_of(key_id, nodes, 2 ** m)
        st.markdown(f"**Target key ID:** `{key_id}` → **responsible node (successor):** `{succ}`")
        if path and len(path) > 1:
            step_text = " → ".join(str(n) for n in path)
            st.code(f"path: {step_text}\nsteps: {len(path)-1}", language="text")
        else:
            st.write("No movement (already at successor or trivial case).")

    with st.expander("ℹ️ How to interpret this"):
        st.markdown(
            """
- **Ring**: The circle represents the identifier space `[0, 2^m - 1]`.
- **Nodes**: Blue dots are nodes; the **selected** node is red; the node responsible for the **key** is orange.
- **Finger chords**: Dashed lines from the selected node point to its finger entries (successors of `(n + 2^{i-1}) mod 2^m`).
- **Lookup path**: Solid arrows show the hop-by-hop path using Chord’s greedy routing to locate the key’s responsible node.
- **Table**: Lists `(i, start, successor)` for the selected node’s finger table.
            """
        )

st.divider()
with st.expander("📘 Teaching tips & suggested exercises"):
    st.markdown(
        """
1. **Vary `m`** to see how the space grows and fingers get longer.
2. **Increase/decrease N** to discuss scalability and `O(log N)` routing.
3. **Manually place nodes** (Custom IDs) to craft edge cases (e.g., sparse clusters).
4. **Trace lookups** for different key IDs and compare hop counts.
5. **Ask: what changes if a node joins/leaves?** (Discuss stabilization conceptually.)
        """
    )
