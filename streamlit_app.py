# app_stepper.py
# Chord DHT • Click-through Tutor: Key Space → Assign Nodes → Build Finger Table
# -----------------------------------------------------------------------------
# This version highlights n (selected node) and the current start[i] on the ring
# when you click "Next entry" during Step 3.

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Session helpers
# ---------------------------
def init_state():
    if "step" not in st.session_state:
        st.session_state.step = 1           # 1: key space, 2: nodes, 3: fingers
    if "m" not in st.session_state:
        st.session_state.m = 5              # default 0..31
    if "mode" not in st.session_state:
        st.session_state.mode = "Manual IDs (0..2^m-1)"
    if "nodes_sorted" not in st.session_state:
        st.session_state.nodes_sorted = []
    if "node_map" not in st.session_state:
        st.session_state.node_map = {}      # label -> id (if hash mode)
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    if "finger_k" not in st.session_state:
        st.session_state.finger_k = 0       # how many fingers revealed (0..m)
    if "show_radial" not in st.session_state:
        st.session_state.show_radial = True # show radial guide n -> start

def reset_all():
    for k in ["step", "m", "mode", "nodes_sorted", "node_map",
              "selected_node", "finger_k", "show_radial"]:
        if k in st.session_state:
            del st.session_state[k]
    init_state()

# ---------------------------
# Core math
# ---------------------------
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
    space = 2 ** m
    entries = []
    for i in range(1, m + 1):
        start = (n + 2 ** (i - 1)) % space
        succ = successor_of(start, nodes_sorted)
        entries.append(FingerEntry(i=i, start=start, node=succ))
    return entries

# ---------------------------
# Visualization
# ---------------------------
def node_xy(id_val: int, space: int, radius: float = 1.0) -> Tuple[float, float]:
    theta = 2 * math.pi * (id_val / space)
    return radius * math.cos(theta), radius * math.sin(theta)

def ring_figure(
    space: int,
    nodes_sorted: List[int],
    selected: int = None,
    fingers_to_draw: List[FingerEntry] = None,
    highlight_start: int = None,
    show_radial: bool = False
) -> go.Figure:
    """Draw ring, nodes, (optionally) some finger chords, and a highlight marker at start[i]."""
    R = 1.0
    circle_angles = np.linspace(0, 2*np.pi, 361)

    fig = go.Figure()
    # Ring
    fig.add_trace(go.Scatter(x=np.cos(circle_angles), y=np.sin(circle_angles),
                             mode="lines", name="Hash ring", hoverinfo="skip"))

    # Nodes
    xs, ys, labels, colors, sizes = [], [], [], [], []
    for nid in nodes_sorted:
        x, y = node_xy(nid, space, R)
        xs.append(x); ys.append(y)
        label = f"{nid}"
        if selected is not None and nid == selected:
            colors.append("crimson"); sizes.append(16); label += " (n)"
        else:
            colors.append("royalblue"); sizes.append(11)
        labels.append(label)

    if nodes_sorted:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=[str(n) for n in nodes_sorted],
            textposition="top center",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
            hovertext=labels, hoverinfo="text", name="Nodes"
        ))

    # Finger chords from selected node to successors
    if selected is not None and fingers_to_draw:
        sx, sy = node_xy(selected, space, R)
        for fe in fingers_to_draw:
            tx, ty = node_xy(fe.node, space, R)
            fig.add_trace(go.Scatter(
                x=[sx, tx], y=[sy, ty], mode="lines",
                line=dict(width=2, dash="dot"),
                name=f"finger[{fe.i}]→{fe.node}",
                hovertext=f"finger[{fe.i}] start={fe.start} → succ={fe.node}",
                hoverinfo="text",
                showlegend=False
            ))

    # Highlight start[i] as a diamond marker and optional radial guide from n
    if highlight_start is not None and selected is not None:
        hx, hy = node_xy(highlight_start, space, R)
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy], mode="markers+text",
            text=[f"start={highlight_start}"],
            textposition="bottom center",
            marker=dict(size=14, symbol="diamond", line=dict(width=1, color="black")),
            name="start[i]",
            hoverinfo="text"
        ))
        if show_radial:
            sx, sy = node_xy(selected, space, R)
            fig.add_trace(go.Scatter(
                x=[sx, hx], y=[sy, hy], mode="lines",
                line=dict(width=1, dash="dash"),
                name="radial n→start",
                hoverinfo="skip",
                showlegend=False
            ))

    fig.update_layout(
        width=820, height=820,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        title="Chord • Ring • Nodes • Fingers • start[i]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# ---------------------------
# UI helpers
# ---------------------------
def step_header(current: int):
    st.markdown(
        f"""
**Steps**
1. {'**Key space** ✅' if current > 1 else ('**Key space** ⬅️' if current == 1 else 'Key space')}
2. {'**Assign nodes** ✅' if current > 2 else ('**Assign nodes** ⬅️' if current == 2 else 'Assign nodes')}
3. {'**Build finger table** ⬅️' if current == 3 else 'Build finger table'}
        """
    )

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Chord DHT • Step-by-Step Tutor", layout="wide")
init_state()

st.title("🔗 Chord DHT — Step-by-Step Tutor")
step_header(st.session_state.step)

with st.sidebar:
    st.button("🔄 Reset", on_click=reset_all)
    st.caption("Click **Reset** to start fresh.")

# === STEP 1: Key space =======================================================
if st.session_state.step == 1:
    st.subheader("Step 1 — Choose the key space size")
    m = st.slider("m (identifier bits)", min_value=3, max_value=10, value=st.session_state.m,
                  help="The identifier space has size 2^m (IDs 0..2^m−1).")
    st.session_state.m = m
    space = 2 ** m

    st.markdown("**Equation:**")
    st.latex(rf"\text{{ID space size}} = 2^{m} = {space}")
    st.latex(rf"\text{{IDs}} \in \{{0,1,\dots,{space-1}\}}")

    fig = ring_figure(space=space, nodes_sorted=[], selected=None, fingers_to_draw=None)
    st.plotly_chart(fig, use_container_width=True)

    st.info("Proceed to Step 2 to place nodes on the ring.")
    st.button("Next → Assign nodes", type="primary", on_click=lambda: st.session_state.update(step=2))

# === STEP 2: Assign Nodes ====================================================
elif st.session_state.step == 2:
    st.subheader("Step 2 — Assign nodes to the ring")
    space = 2 ** st.session_state.m

    # Placement mode
    mode = st.radio("Node placement mode",
                    ["Manual IDs (0..2^m-1)", "Hash labels → IDs (SHA-1 mod 2^m)"],
                    index=0 if st.session_state.mode.startswith("Manual") else 1)
    st.session_state.mode = mode

    nodes_sorted = []
    node_map: Dict[str, int] = {}

    if mode.startswith("Manual"):
        default_ids = "1, 4, 9, 11, 14, 18, 20, 21, 28" if st.session_state.m == 5 else "1, 3, 6, 8, 12"
        ids_text = st.text_area("Node IDs (comma/space separated)",
                                value=", ".join(str(n) for n in (st.session_state.nodes_sorted or [])) or default_ids)
        raw = [t.strip() for t in ids_text.replace(",", " ").split()]
        try:
            nodes_sorted = sorted(set(int(x) % space for x in raw if x != ""))
        except ValueError:
            st.error("Invalid IDs. Please enter integers.")
            nodes_sorted = []
        st.markdown("**No hashing used in Manual mode** — you place nodes directly on 0..2^m−1.")
    else:
        labels_text = st.text_area("Node labels (one per line)", value="nodeA\nnodeB\nnodeC\nnodeD")
        labels = [s.strip() for s in labels_text.splitlines() if s.strip()]
        node_map = {lbl: sha1_mod(lbl, space) for lbl in labels}
        nodes_sorted = sorted(set(node_map.values()))
        st.markdown("**Hashing equation:**")
        st.latex(rf"\text{{node\_id}} = \operatorname{{SHA1}}(\text{{label}}) \bmod 2^{{{st.session_state.m}}}")

    st.session_state.nodes_sorted = nodes_sorted
    st.session_state.node_map = node_map

    if not nodes_sorted:
        st.warning("Add at least two nodes.")
    else:
        if st.session_state.selected_node not in nodes_sorted:
            st.session_state.selected_node = nodes_sorted[0]

        fig = ring_figure(space=space, nodes_sorted=nodes_sorted, selected=st.session_state.selected_node)
        st.plotly_chart(fig, use_container_width=True)

        if mode.startswith("Manual"):
            df_nodes = pd.DataFrame([{"node_id": n} for n in nodes_sorted])
        else:
            df_nodes = pd.DataFrame(
                [{"node_label": lbl, "node_id": nid} for lbl, nid in sorted(node_map.items(), key=lambda x: x[1])],
                columns=["node_label", "node_id"]
            )
        st.subheader("📍 Node placements")
        st.dataframe(df_nodes, use_container_width=True, hide_index=True)

    cols = st.columns([1, 1, 8])
    with cols[0]:
        st.button("← Back", on_click=lambda: st.session_state.update(step=1))
    with cols[1]:
        st.button("Next → Build finger table", type="primary",
                  disabled=(len(nodes_sorted) < 2),
                  on_click=lambda: st.session_state.update(step=3, finger_k=0))

# === STEP 3: Build Finger Table =============================================
elif st.session_state.step == 3:
    st.subheader("Step 3 — Build the finger table (click to reveal each entry)")
    m = st.session_state.m
    space = 2 ** m
    nodes_sorted = st.session_state.nodes_sorted or []
    if len(nodes_sorted) < 2:
        st.warning("Go back and add at least two nodes.")
    else:
        selected_node = st.selectbox("Selected node (by ID)", options=nodes_sorted,
                                     index=max(0, nodes_sorted.index(st.session_state.selected_node)
                                               if st.session_state.selected_node in nodes_sorted else 0))
        st.session_state.selected_node = selected_node

        all_entries = build_finger_table(selected_node, nodes_sorted, m)

        # How many entries are revealed?
        k = st.session_state.finger_k
        st.markdown(
            f"**Reveal progress:** {k}/{m} entries "
            "(click **Next entry** to compute the next start & successor)."
        )

        # Controls
        c1, c2, c3, c4 = st.columns([1,1,2,6])
        with c1:
            st.button("Reset entries", on_click=lambda: st.session_state.update(finger_k=0))
        with c2:
            st.button("Next entry", type="primary",
                      disabled=(k >= m),
                      on_click=lambda: st.session_state.update(finger_k=min(m, st.session_state.finger_k + 1)))
        with c3:
            st.button("Reveal all", disabled=(k >= m),
                      on_click=lambda: st.session_state.update(finger_k=m))

        st.checkbox("Show radial guide n → start[i]", value=st.session_state.show_radial,
                    key="show_radial")

        # Entries to draw and highlight
        shown_entries = all_entries[:k]
        current_start = shown_entries[-1].start if k > 0 else None

        fig = ring_figure(
            space=space,
            nodes_sorted=nodes_sorted,
            selected=selected_node,
            fingers_to_draw=shown_entries,
            highlight_start=current_start,
            show_radial=st.session_state.show_radial
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table of revealed entries
        st.subheader("📇 Finger table (revealed so far)")
        df_ft = pd.DataFrame(
            [{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in shown_entries],
            columns=["i", "start", "successor"]
        )
        st.dataframe(df_ft, use_container_width=True, hide_index=True)

        # Global finger definition
        st.markdown("**Finger definition:**")
        st.latex(rf"\text{{start}}[i] = (n + 2^{{i-1}}) \bmod 2^{{{m}}}")
        st.latex(r"\text{finger}[i] = \operatorname{succ}(\text{start}[i])")

        # If something is revealed, show the CURRENT step very clearly
        if k == 0:
            st.info("Click **Next entry** to compute finger[1].")
        else:
            fe = shown_entries[-1]
            st.subheader("🧮 Current step")
            st.latex(rf"n = {selected_node}")
            st.latex(
                rf"\text{{start}}[{fe.i}] = ({selected_node} + 2^{{{fe.i-1}}}) \bmod 2^{{{m}}} = {fe.start}"
            )
            st.latex(
                rf"\text{{finger}}[{fe.i}] = \operatorname{{succ}}(\text{{start}}[{fe.i}]) "
                rf"= \operatorname{{succ}}({fe.start}) = {fe.node}"
            )

            # Also show all revealed equations (history)
            with st.expander("Show all revealed equation steps"):
                for e in shown_entries:
                    st.latex(
                        rf"\text{{finger}}[{e.i}]:\;"
                        rf"\text{{start}}[{e.i}] = ({selected_node} + 2^{{{e.i-1}}}) \bmod 2^{{{m}}} = {e.start},\;"
                        rf"\text{{node}} = \operatorname{{succ}}({e.start}) = {e.node}"
                    )

        # Full table (optional)
        with st.expander("Show full finger table (all m entries)"):
            df_full = pd.DataFrame(
                [{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in all_entries],
                columns=["i", "start", "successor"]
            )
            st.dataframe(df_full, use_container_width=True, hide_index=True)

    cols = st.columns([1, 1, 8])
    with cols[0]:
        st.button("← Back", on_click=lambda: st.session_state.update(step=2))
    with cols[1]:
        st.button("Restart", on_click=reset_all)
