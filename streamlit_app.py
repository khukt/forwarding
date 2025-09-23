# app_fixed32_steps.py
# Chord DHT — 3 clear steps: (1) Assign nodes, (2) Build finger table, (3) Search/route
# Fixed ring 0..31 (m=5). Non-listed IDs are shown as disabled placeholders.

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------- Constants -----------------
M = 5
SPACE = 2 ** M
ALL_POSITIONS = list(range(SPACE))

# ----------------- Page / Style -----------------
st.set_page_config(page_title="Chord 0..31 • 3-Step Tutor", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container { padding-top: 0.75rem; padding-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.25rem 0.75rem; }
    .stMetric { background: #fafafa; border-radius: 12px; padding: 0.5rem 0.75rem; }
    .small-note { color: #666; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- State -----------------
def init_state():
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "active_nodes" not in st.session_state:
        st.session_state.active_nodes = [1, 4, 9, 11, 14, 18, 20, 21, 28]
    if "selected" not in st.session_state:
        st.session_state.selected = st.session_state.active_nodes[0]
    if "k" not in st.session_state:
        st.session_state.k = 0  # fingers revealed
    if "key_id" not in st.session_state:
        st.session_state.key_id = 26
    if "route_path" not in st.session_state:
        st.session_state.route_path: List[int] = []
    if "route_reasons" not in st.session_state:
        st.session_state.route_reasons: List[str] = []
    if "route_idx" not in st.session_state:
        st.session_state.route_idx = 0  # hops revealed

init_state()

# ----------------- Math helpers -----------------
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
    """Return (path, reasons) using only active nodes."""
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

# ----------------- Plot helpers -----------------
def node_xy(id_val: int, space: int, radius: float = 1.0) -> Tuple[float, float]:
    theta = 2 * math.pi * (id_val / space)
    return radius * math.cos(theta), radius * math.sin(theta)

def ring_figure(
    active_nodes: List[int],
    selected: Optional[int] = None,
    fingers: Optional[List[FingerEntry]] = None,
    highlight_start: Optional[int] = None,
    show_radial: bool = False,
    route_path: Optional[List[int]] = None,
    route_hops_to_show: int = 0,
    key: Optional[int] = None,
    width: int = 700,
    height: int = 700,
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

    fig.update_layout(
        width=width, height=height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=4, r=4, t=28, b=4),
        plot_bgcolor="white",
        title="Chord • Ring 0..31",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# ----------------- Header -----------------
st.subheader("🔗 Chord DHT — 3-Step Visual Tutor (Fixed 0..31)")
st.caption("Step 1: Assign nodes → Step 2: Build finger table → Step 3: Search (route). Non-listed IDs are grey placeholders.")

# ----------------- Step Nav -----------------
cnav1, cnav2, cnav3, cnav4 = st.columns([1, 1, 1, 6])
with cnav1:
    if st.button("← Prev"):
        st.session_state.step = max(1, st.session_state.step - 1)
with cnav2:
    if st.button("Next →"):
        st.session_state.step = min(3, st.session_state.step + 1)
with cnav3:
    st.write(f"**Step {st.session_state.step}/3**")

# ==========================================================
# STEP 1 — ASSIGN THE NODES
# ==========================================================
if st.session_state.step == 1:
    left, right = st.columns([0.55, 0.45])
    with left:
        # Always draw full ring 0..31, marking non-active as disabled
        fig = ring_figure(active_nodes=st.session_state.active_nodes, width=700, height=700)
        st.plotly_chart(fig, use_container_width=False)

    with right:
        st.markdown("### Step 1 — Assign the nodes")
        ids_text = st.text_area(
            "Active node IDs (0..31)",
            value=", ".join(str(n) for n in st.session_state.active_nodes),
            help="Comma/space separated. Other IDs will appear as disabled placeholders."
        )
        raw = [t.strip() for t in ids_text.replace(",", " ").split()]
        try:
            active_nodes = sorted(set(int(x) % SPACE for x in raw if x != ""))
        except ValueError:
            active_nodes = st.session_state.active_nodes
        st.session_state.active_nodes = active_nodes
        if not active_nodes:
            st.warning("Add at least one active node.")
        else:
            # Status
            succ_demo = successor_of(st.session_state.key_id, active_nodes)
            m1, m2, m3 = st.columns(3)
            m1.metric("Active nodes", len(active_nodes))
            m2.metric("Example key k", st.session_state.key_id)
            m3.metric("succ(k)", succ_demo)

        st.divider()
        st.markdown("**Optional: Hash labels → IDs**")
        labels_text = st.text_area("Node labels (one per line)", value="nodeA\nnodeB\nnodeC\nnodeD")
        labels = [s.strip() for s in labels_text.splitlines() if s.strip()]
        if st.button("Hash labels (SHA-1 mod 32)"):
            node_map = {lbl: sha1_mod(lbl, SPACE) for lbl in labels}
            st.session_state.active_nodes = sorted(set(node_map.values()))
        st.markdown("Equation:")
        st.latex(r"\text{node\_id} = \operatorname{SHA1}(\text{label}) \bmod 32")

# ==========================================================
# STEP 2 — BUILD THE FINGER TABLE
# ==========================================================
elif st.session_state.step == 2:
    if not st.session_state.active_nodes:
        st.warning("Go back to Step 1 and add active nodes.")
    else:
        # Controls row
        crow1, crow2, crow3, crow4 = st.columns([1.2, 1, 1, 6])
        with crow1:
            st.session_state.selected = st.selectbox(
                "Selected node n",
                options=st.session_state.active_nodes,
                index=min(
                    len(st.session_state.active_nodes)-1,
                    st.session_state.active_nodes.index(st.session_state.selected)
                    if st.session_state.selected in st.session_state.active_nodes else 0
                )
            )
        with crow2:
            if st.button("Reset fingers"):
                st.session_state.k = 0
        with crow3:
            if st.button("Next finger"):
                st.session_state.k = min(M, st.session_state.k + 1)

        # Compute finger table for selected
        selected = st.session_state.selected
        fingers_all = build_finger_table(selected, st.session_state.active_nodes, M)
        k = st.session_state.k
        fingers_shown = fingers_all[:k]
        current_start = fingers_shown[-1].start if k > 0 else None

        left, right = st.columns([0.55, 0.45])
        with left:
            fig = ring_figure(
                active_nodes=st.session_state.active_nodes,
                selected=selected,
                fingers=fingers_shown,
                highlight_start=current_start,
                show_radial=True,
                width=700, height=700
            )
            st.plotly_chart(fig, use_container_width=False)

        with right:
            st.markdown("### Step 2 — Build the finger table")
            st.markdown("**Definitions (m=5):**")
            st.latex(r"\text{start}[i] = (n + 2^{i-1}) \bmod 32")
            st.latex(r"\text{finger}[i] = \operatorname{succ}(\text{start}[i])")
            df_ft = pd.DataFrame(
                [{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in fingers_shown],
                columns=["i", "start", "successor"]
            )
            st.dataframe(df_ft, hide_index=True, height=240, use_container_width=True)

            if k > 0:
                fe = fingers_shown[-1]
                st.markdown("**Current step:**")
                st.latex(rf"n = {selected}")
                st.latex(rf"\text{{start}}[{fe.i}] = ({selected} + 2^{{{fe.i-1}}}) \bmod 32 = {fe.start}")
                st.latex(rf"\text{{finger}}[{fe.i}] = \operatorname{{succ}}({fe.start}) = {fe.node}")
            else:
                st.markdown('<div class="small-note">Click <b>Next finger</b> to reveal the first entry.</div>',
                            unsafe_allow_html=True)

# ==========================================================
# STEP 3 — SEARCH / FIND THE ROUTE
# ==========================================================
elif st.session_state.step == 3:
    if not st.session_state.active_nodes:
        st.warning("Go back to Step 1 and add active nodes.")
    else:
        # Controls row
        crow1, crow2, crow3, crow4 = st.columns([1.2, 1.2, 1, 6])
        with crow1:
            st.session_state.start_node = st.selectbox(
                "Start node",
                options=st.session_state.active_nodes,
                index=0
            )
        with crow2:
            st.session_state.key_id = st.number_input("Key k", min_value=0, max_value=31, value=st.session_state.key_id, step=1)
        with crow3:
            if st.button("Start lookup"):
                path, reasons = chord_lookup_full(
                    st.session_state.start_node,
                    st.session_state.key_id,
                    st.session_state.active_nodes,
                    M
                )
                st.session_state.route_path = path
                st.session_state.route_reasons = reasons
                st.session_state.route_idx = 0

        # Next hop row
        cnext, _ = st.columns([0.15, 0.85])
        with cnext:
            if st.button("Next hop"):
                if st.session_state.route_path:
                    st.session_state.route_idx = min(
                        len(st.session_state.route_path) - 1,
                        st.session_state.route_idx + 1
                    )
                else:
                    path, reasons = chord_lookup_full(
                        st.session_state.start_node, st.session_state.key_id, st.session_state.active_nodes, M
                    )
                    st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_idx = path, reasons, 0

        # Ensure route exists
        if not st.session_state.route_path:
            path, reasons = chord_lookup_full(
                st.session_state.start_node, st.session_state.key_id, st.session_state.active_nodes, M
            )
            st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_idx = path, reasons, 0

        route_path = st.session_state.route_path
        route_reasons = st.session_state.route_reasons
        route_hops_to_show = st.session_state.route_idx
        succ_k = successor_of(st.session_state.key_id, st.session_state.active_nodes)

        left, right = st.columns([0.55, 0.45])
        with left:
            # Draw ring + (optionally) last shown finger table for selected from Step 2
            selected = st.session_state.selected
            fingers_all = build_finger_table(selected, st.session_state.active_nodes, M)
            fingers_shown = fingers_all[:st.session_state.k]
            current_start = fingers_shown[-1].start if st.session_state.k > 0 else None

            fig = ring_figure(
                active_nodes=st.session_state.active_nodes,
                selected=selected,
                fingers=fingers_shown,
                highlight_start=current_start,
                show_radial=True,
                route_path=route_path,
                route_hops_to_show=route_hops_to_show,
                key=st.session_state.key_id,
                width=700, height=700
            )
            st.plotly_chart(fig, use_container_width=False)

        with right:
            st.markdown("### Step 3 — Search / Find the route")
            m1, m2, m3 = st.columns(3)
            m1.metric("Start node", st.session_state.start_node)
            m2.metric("Key k", st.session_state.key_id)
            m3.metric("succ(k)", succ_k)

            st.markdown("**Path**")
            st.code(" → ".join(str(n) for n in route_path), language="text")

            st.markdown("**Reasoning**")
            # Show reasons up to the number of hops drawn (+1 for final arrival message when reached)
            max_to_show = min(route_hops_to_show + 1, len(route_reasons))
            for i in range(max_to_show):
                st.latex(route_reasons[i])
