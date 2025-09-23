# app_fixed32_students.py
# Chord DHT — Student-friendly 3-step tutor for the fixed 0..31 ring (m=5)
# Step 1: Assign nodes • Step 2: Build finger table • Step 3: Search/route
# New: Show sectors (responsibility), Export PNG, Per-hop tooltips

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------- Constants -----------------
M = 5
SPACE = 2 ** M
ALL_POSITIONS = list(range(SPACE))

# ----------------- Page / Style -----------------
st.set_page_config(page_title="Chord 0..31 • Student Tutor", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container { padding-top: 0.6rem; padding-bottom: 0.4rem; }
    .step-chip { display:inline-block; padding:4px 10px; border-radius:999px; margin-right:6px;
                 background:#f1f5f9; font-weight:600; }
    .step-chip.active { background:#dbeafe; color:#1e3a8a; }
    .hint { background:#f8fafc; border:1px solid #e2e8f0; padding:10px 12px; border-radius:10px; }
    .legend { font-size:0.9rem; color:#334155; }
    .kbd { padding:2px 6px; border:1px solid #cbd5e1; border-radius:6px; background:#f8fafc; }
    .metric-card .stMetric { background: #fafafa; border-radius: 12px; padding: 0.4rem 0.6rem; }
    .preset-btn button { width:100%; }
    .quiz-card { background:#fffbeb; border:1px solid #fde68a; padding:10px; border-radius:10px; }
    </style>
    """, unsafe_allow_html=True
)

# ----------------- State -----------------
def init_state():
    if "step" not in st.session_state: st.session_state.step = 1
    if "active_nodes" not in st.session_state: st.session_state.active_nodes = [1,4,9,11,14,18,20,21,28]
    if "selected" not in st.session_state: st.session_state.selected = st.session_state.active_nodes[0]
    if "k" not in st.session_state: st.session_state.k = 0  # fingers revealed
    if "key_id" not in st.session_state: st.session_state.key_id = 26
    if "route_path" not in st.session_state: st.session_state.route_path: List[int] = []
    if "route_reasons" not in st.session_state: st.session_state.route_reasons: List[str] = []
    if "route_texts" not in st.session_state: st.session_state.route_texts: List[str] = []
    if "route_idx" not in st.session_state: st.session_state.route_idx = 0
    if "quiz" not in st.session_state:
        st.session_state.quiz = {"start": None, "key": None, "show": False, "answer": None, "route": [], "reasons": []}
    if "projector" not in st.session_state: st.session_state.projector = False
    if "colorblind" not in st.session_state: st.session_state.colorblind = True
    if "show_sectors" not in st.session_state: st.session_state.show_sectors = True

init_state()

# ----------------- Math helpers -----------------
def sha1_mod(s: str, space: int) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h, 16) % space

def mod_interval_contains(a: int, b: int, x: int, m: int, inclusive_right: bool = False) -> bool:
    if a == b: return inclusive_right
    if a < b: return (a < x <= b) if inclusive_right else (a < x < b)
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
    if not nodes_sorted: return entries
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
    """
    Return (path, reasons_latex, reasons_text) for iterative Chord lookup using only active nodes.
    reasons_text aligns to edges (for tooltips). reasons_latex includes final arrival messages.
    """
    if not nodes_sorted:
        return [start_node], [r"\text{No active nodes.}"], ["No active nodes"]
    path = [start_node]
    reasons_latex: List[str] = []
    reasons_text: List[str] = []   # one per hop
    succ_k = successor_of(key, nodes_sorted)
    finger_map: Dict[int, List[int]] = {n: [fe.node for fe in build_finger_table(n, nodes_sorted, m)]
                                        for n in nodes_sorted}

    while len(path) < max_steps:
        curr = path[-1]
        if curr == succ_k:
            reasons_latex.append(rf"\mathbf{{Stop:}}\ \text{{current}}={curr}=\operatorname{{succ}}({key})")
            break

        curr_idx = nodes_sorted.index(curr)
        curr_succ = nodes_sorted[(curr_idx + 1) % len(nodes_sorted)]

        if mod_interval_contains(curr, curr_succ, key, 2 ** m, inclusive_right=True):
            reasons_latex.append(
                rf"\text{{Since }} {key}\in({curr},{curr_succ}] \Rightarrow "
                rf"\text{{next}}=\operatorname{{succ}}({curr})={curr_succ}"
            )
            reasons_text.append(f"{key} in ({curr},{curr_succ}] ⇒ next = succ({curr}) = {curr_succ}")
            path.append(curr_succ)
            if curr_succ == succ_k:
                reasons_latex.append(rf"\mathbf{{Arrived}}\ \text{{at}}\ \operatorname{{succ}}({key})={succ_k}")
            continue

        cpf = closest_preceding_finger(curr, finger_map[curr], key, m)
        if cpf == curr:
            reasons_latex.append(
                rf"\text{{No finger in }}({curr},{key}) \Rightarrow "
                rf"\text{{fallback to }} \operatorname{{succ}}({curr})={curr_succ}"
            )
            reasons_text.append(f"No finger in ({curr},{key}) ⇒ fallback succ({curr}) = {curr_succ}")
            path.append(curr_succ)
        else:
            reasons_latex.append(
                rf"\text{{Choose closest preceding finger of }}{curr}\ \text{{toward }}{key}: "
                rf"{cpf}\in({curr},{key})"
            )
            reasons_text.append(f"Closest preceding finger toward {key}: {cpf} ∈ ({curr},{key})")
            path.append(cpf)

        if path[-1] == succ_k:
            reasons_latex.append(rf"\mathbf{{Arrived}}\ \text{{at}}\ \operatorname{{succ}}({key})={succ_k}")
            break

    return path, reasons_latex, reasons_text

# ----------------- Plot helpers (UI palette aware) -----------------
def node_xy(id_val: int, space: int, radius: float = 1.0) -> Tuple[float, float]:
    theta = 2 * math.pi * (id_val / space)
    return radius * math.cos(theta), radius * math.sin(theta)

def add_sector(fig: go.Figure, a: int, b: int, color: str, alpha: float = 0.08, steps: int = 40):
    """
    Shade the responsibility interval (a, b] clockwise on the ring as a thin wedge.
    """
    if a == b:
        return  # full circle responsibility is not meaningful here
    start = a / SPACE * 2 * math.pi
    end = b / SPACE * 2 * math.pi
    if a < b:
        thetas = np.linspace(start, end, steps)
    else:
        # wrap-around
        thetas = np.concatenate([np.linspace(start, 2*math.pi, steps//2),
                                 np.linspace(0, end, steps - steps//2)])

    R_outer = 1.02
    R_inner = 0.90
    xs = np.cos(thetas) * R_outer
    ys = np.sin(thetas) * R_outer
    xs2 = np.cos(thetas[::-1]) * R_inner
    ys2 = np.sin(thetas[::-1]) * R_inner
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs2]),
        y=np.concatenate([ys, ys2]),
        fill="toself",
        mode="lines",
        line=dict(width=0),
        fillcolor=color,
        opacity=alpha,
        hoverinfo="skip",
        showlegend=False
    ))

def ring_figure(
    active_nodes: List[int],
    selected: Optional[int] = None,
    fingers: Optional[List[FingerEntry]] = None,
    highlight_start: Optional[int] = None,
    show_radial: bool = False,
    route_path: Optional[List[int]] = None,
    route_hops_to_show: int = 0,
    route_texts: Optional[List[str]] = None,
    key: Optional[int] = None,
    width: int = 700,
    height: int = 700,
    projector: bool = False,
    colorblind: bool = True,
    show_sectors: bool = True,
) -> go.Figure:

    COLORS = {
        "ring": "#334155",
        "disabled": "#9ca3af",
        "active": "#1f77b4" if colorblind else "royalblue",
        "selected": "#d62728",
        "succ": "#ff7f0e",
        "start": "#9467bd",
        "radial": "#6b7280",
        "hop": "#111827",
        "sector": "#3b82f6" if colorblind else "#2563eb",  # blue-ish
    }

    R = 1.0
    circle_angles = np.linspace(0, 2*np.pi, 361)
    fig = go.Figure()

    # Base ring
    fig.add_trace(go.Scatter(x=np.cos(circle_angles), y=np.sin(circle_angles),
                             mode="lines", line=dict(color=COLORS["ring"], width=2 if projector else 1.5),
                             name="Ring", hoverinfo="skip"))

    # Responsibility sectors for active nodes (predecessor -> node)
    if show_sectors and active_nodes:
        for idx, nid in enumerate(active_nodes):
            pred = active_nodes[idx - 1]  # predecessor in sorted list (wraps)
            add_sector(fig, pred, nid, COLORS["sector"], alpha=0.08 if not projector else 0.12)

    # Disabled placeholders
    disabled_positions = [i for i in ALL_POSITIONS if i not in set(active_nodes)]
    if disabled_positions:
        xs, ys = [], []
        for nid in disabled_positions:
            x, y = node_xy(nid, SPACE, R)
            xs.append(x); ys.append(y)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=[str(i) for i in disabled_positions], textposition="top center",
            marker=dict(size=10 if projector else 9, symbol="circle-open",
                        color=COLORS["disabled"], line=dict(width=1, color=COLORS["disabled"])),
            name="Disabled", opacity=0.45, hoverinfo="skip"
        ))

    # Active nodes (selected red, succ(key) orange)
    succ_k = successor_of(key, active_nodes) if (key is not None and active_nodes) else None
    xs, ys, sizes, colors, labels = [], [], [], [], []
    for nid in active_nodes:
        x, y = node_xy(nid, SPACE, R)
        xs.append(x); ys.append(y)
        if selected == nid:
            sizes.append(18 if projector else 16); colors.append(COLORS["selected"]); labels.append(f"{nid} (selected)")
        elif succ_k is not None and succ_k == nid:
            sizes.append(16 if projector else 14); colors.append(COLORS["succ"]); labels.append(f"{nid} (succ(key))")
        else:
            sizes.append(12 if projector else 11); colors.append(COLORS["active"]); labels.append(str(nid))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[str(n) for n in active_nodes], textposition="top center",
        hovertext=labels, hoverinfo="text",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
        name="Active nodes"
    ))

    # Finger chords
    if selected is not None and fingers:
        sx, sy = node_xy(selected, SPACE, R)
        for fe in fingers:
            tx, ty = node_xy(fe.node, SPACE, R)
            fig.add_trace(go.Scatter(
                x=[sx, tx], y=[sy, ty], mode="lines",
                line=dict(width=3 if projector else 2, dash="dot", color=COLORS["active"]),
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
            marker=dict(size=16 if projector else 13, symbol="diamond",
                        line=dict(width=1, color="black"), color=COLORS["start"]),
            name="start[i]", hoverinfo="text"
        ))
        if show_radial:
            sx, sy = node_xy(selected, SPACE, R)
            fig.add_trace(go.Scatter(
                x=[sx, hx], y=[sy, hy], mode="lines",
                line=dict(width=2 if projector else 1, dash="dash", color=COLORS["radial"]),
                name="n→start", hoverinfo="skip", showlegend=False
            ))

    # Route arrows with tooltips
    if route_path and route_hops_to_show > 0:
        for i in range(min(route_hops_to_show, len(route_path) - 1)):
            a = route_path[i]; b = route_path[i + 1]
            ax, ay = node_xy(a, SPACE, R); bx, by = node_xy(b, SPACE, R)
            tip = route_texts[i] if (route_texts and i < len(route_texts)) else f"hop {i+1}: {a} → {b}"
            fig.add_trace(go.Scatter(
                x=[ax, bx], y=[ay, by], mode="lines+markers",
                line=dict(width=4 if projector else 3, color=COLORS["hop"]),
                marker=dict(size=8 if projector else 6, color=COLORS["hop"]),
                name=f"hop {i+1}", hovertext=tip, hoverinfo="text", showlegend=False
            ))
            fig.add_annotation(
                x=bx, y=by, ax=ax, ay=ay,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.3 if projector else 1.15, arrowwidth=2, arrowcolor=COLORS["hop"]
            )

    fig.update_layout(
        width=width, height=height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=6, r=6, t=34, b=6),
        plot_bgcolor="white",
        title="Chord • Ring 0..31",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=11 if projector else 10)),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# ----------------- Header / global toggles -----------------
l, r = st.columns([0.7, 0.3])
with l:
    st.subheader("🔗 Chord DHT — Student Tutor (Fixed 0..31)")
    st.caption("Step 1: Assign nodes → Step 2: Build finger table → Step 3: Search/route. Non-listed IDs are grey placeholders.")
with r:
    st.session_state.projector = st.toggle("Projector mode", value=st.session_state.projector, help="Bigger fonts & thicker lines.")
    st.session_state.colorblind = st.toggle("Color-blind palette", value=st.session_state.colorblind, help="High-contrast colors.")
    st.session_state.show_sectors = st.toggle("Show sectors", value=st.session_state.show_sectors, help="Shade each node's responsibility interval.")

projector = st.session_state.projector
colorblind = st.session_state.colorblind
show_sectors = st.session_state.show_sectors

# Stepper chips
chips = []
for i, label in [(1, "1 Assign"), (2, "2 Fingers"), (3, "3 Search")]:
    cls = "step-chip active" if st.session_state.step == i else "step-chip"
    chips.append(f'<span class="{cls}">{label}</span>')
st.markdown(" ".join(chips), unsafe_allow_html=True)

# Nav + presets
cnav1, cnav2, cnav3, cnav4 = st.columns([1, 1, 1.5, 6])
with cnav1:
    if st.button("← Prev"): st.session_state.step = max(1, st.session_state.step - 1)
with cnav2:
    if st.button("Next →"): st.session_state.step = min(3, st.session_state.step + 1)
with cnav3:
    st.markdown("**Presets**")
    cpa, cpb = st.columns(2)
    with cpa:
        if st.button("k=12 from 28", use_container_width=True):
            st.session_state.active_nodes = [1,4,9,11,14,18,20,21,28]
            st.session_state.selected = 28
            st.session_state.key_id = 12
            st.session_state.k = 5
            path, reasons_latex, reasons_text = chord_lookup_full(28, 12, st.session_state.active_nodes, M)
            st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_texts = path, reasons_latex, reasons_text
            st.session_state.route_idx = len(path)-1
    with cpb:
        if st.button("k=26 from 1", use_container_width=True):
            st.session_state.active_nodes = [1,4,9,11,14,18,20,21,28]
            st.session_state.selected = 1
            st.session_state.key_id = 26
            st.session_state.k = 5
            path, reasons_latex, reasons_text = chord_lookup_full(1, 26, st.session_state.active_nodes, M)
            st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_texts = path, reasons_latex, reasons_text
            st.session_state.route_idx = len(path)-1

# ----------------- STEP 1 — Assign nodes -----------------
if st.session_state.step == 1:
    left, right = st.columns([0.56, 0.44])
    with left:
        fig = ring_figure(
            active_nodes=st.session_state.active_nodes,
            width=720 if projector else 680,
            height=720 if projector else 680,
            projector=projector, colorblind=colorblind, show_sectors=show_sectors
        )
        st.plotly_chart(fig, use_container_width=False)
        # Export PNG
        png = fig.to_image(format="png", scale=2)  # requires kaleido
        st.download_button("⬇️ Download ring as PNG", data=png, file_name="chord_ring.png", mime="image/png")

        st.markdown(
            f"""
            <div class="legend">
            <b>Legend:</b> <span style="color:#1f77b4">● Active</span>,
            <span style="color:#d62728">● Selected</span>,
            <span style="color:#ff7f0e">● succ(key)</span>,
            <span style="color:#9ca3af">◌ Disabled</span>
            </div>
            """, unsafe_allow_html=True
        )

    with right:
        st.markdown("### Step 1 — Assign the nodes")
        st.markdown('<div class="hint">Enter IDs (0–31). Others appear as disabled placeholders. Toggle <b>Show sectors</b> to visualize responsibility intervals.</div>', unsafe_allow_html=True)
        ids_text = st.text_area(
            "Active node IDs",
            value=", ".join(str(n) for n in st.session_state.active_nodes),
            help="Comma/space separated integers between 0 and 31."
        )
        raw = [t.strip() for t in ids_text.replace(",", " ").split()]
        try:
            active_nodes = sorted(set(int(x) % SPACE for x in raw if x != ""))
        except ValueError:
            active_nodes = st.session_state.active_nodes
        st.session_state.active_nodes = active_nodes

        st.markdown("**Optional: hash labels → IDs**")
        labels_text = st.text_area("Node labels (one per line)", value="nodeA\nnodeB\nnodeC\nnodeD", height=100)
        if st.button("Hash labels (SHA-1 mod 32)"):
            node_map = {lbl.strip(): sha1_mod(lbl.strip(), SPACE) for lbl in labels_text.splitlines() if lbl.strip()}
            st.session_state.active_nodes = sorted(set(node_map.values()))
        st.latex(r"\text{node\_id} = \operatorname{SHA1}(\text{label}) \bmod 32")

# ----------------- STEP 2 — Finger table -----------------
elif st.session_state.step == 2:
    if not st.session_state.active_nodes:
        st.warning("Go to Step 1 to add active nodes.")
    else:
        top = st.columns([1.2, 1, 1, 6])
        with top[0]:
            st.session_state.selected = st.selectbox("Selected node n", options=st.session_state.active_nodes,
                                                     index=min(
                                                        len(st.session_state.active_nodes)-1,
                                                        st.session_state.active_nodes.index(st.session_state.selected)
                                                        if st.session_state.selected in st.session_state.active_nodes else 0))
        with top[1]:
            if st.button("Reset fingers"): st.session_state.k = 0
        with top[2]:
            if st.button("Next finger"): st.session_state.k = min(M, st.session_state.k + 1)

        selected = st.session_state.selected
        fingers_all = build_finger_table(selected, st.session_state.active_nodes, M)
        k = st.session_state.k
        fingers_shown = fingers_all[:k]
        current_start = fingers_shown[-1].start if k > 0 else None

        left, right = st.columns([0.56, 0.44])
        with left:
            fig = ring_figure(
                active_nodes=st.session_state.active_nodes, selected=selected,
                fingers=fingers_shown, highlight_start=current_start, show_radial=True,
                width=720 if projector else 680, height=720 if projector else 680,
                projector=projector, colorblind=colorblind, show_sectors=show_sectors
            )
            st.plotly_chart(fig, use_container_width=False)
            png = fig.to_image(format="png", scale=2)
            st.download_button("⬇️ Download ring as PNG", data=png, file_name="chord_fingers.png", mime="image/png")

        with right:
            st.markdown("### Step 2 — Build the finger table")
            st.markdown('<div class="hint">Click <b>Next finger</b> to reveal entries one by one.</div>', unsafe_allow_html=True)
            st.latex(r"\text{start}[i] = (n + 2^{i-1}) \bmod 32")
            st.latex(r"\text{finger}[i] = \operatorname{succ}(\text{start}[i])")
            df_ft = pd.DataFrame([{"i": fe.i, "start": fe.start, "successor": fe.node} for fe in fingers_shown],
                                 columns=["i", "start", "successor"])
            st.dataframe(df_ft, hide_index=True, height=240, use_container_width=True)
            if k > 0:
                fe = fingers_shown[-1]
                st.markdown("**Current step**")
                st.latex(rf"n = {selected}")
                st.latex(rf"\text{{start}}[{fe.i}] = ({selected} + 2^{{{fe.i-1}}}) \bmod 32 = {fe.start}")
                st.latex(rf"\text{{finger}}[{fe.i}] = \operatorname{{succ}}({fe.start}) = {fe.node}")
            else:
                st.info("Click **Next finger** to reveal finger[1].")

# ----------------- STEP 3 — Search / route -----------------
elif st.session_state.step == 3:
    if not st.session_state.active_nodes:
        st.warning("Go to Step 1 to add active nodes.")
    else:
        ctr = st.columns([1.2, 1.1, 1, 1.2, 6])
        with ctr[0]:
            start_node = st.selectbox("Start node", options=st.session_state.active_nodes, index=0, key="start_node")
        with ctr[1]:
            st.session_state.key_id = st.number_input("Key k", 0, 31, st.session_state.key_id, 1)
        with ctr[2]:
            if st.button("Start lookup"):
                path, reasons_latex, reasons_text = chord_lookup_full(start_node, st.session_state.key_id, st.session_state.active_nodes, M)
                st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_texts = path, reasons_latex, reasons_text
                st.session_state.route_idx = 0
        with ctr[3]:
            if st.button("Next hop"):
                if st.session_state.route_path:
                    st.session_state.route_idx = min(len(st.session_state.route_path)-1, st.session_state.route_idx + 1)
                else:
                    path, reasons_latex, reasons_text = chord_lookup_full(start_node, st.session_state.key_id, st.session_state.active_nodes, M)
                    st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_texts = path, reasons_latex, reasons_text
                    st.session_state.route_idx = 0

        # Ensure route exists
        if not st.session_state.route_path:
            path, reasons_latex, reasons_text = chord_lookup_full(start_node, st.session_state.key_id, st.session_state.active_nodes, M)
            st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_texts = path, reasons_latex, reasons_text
            st.session_state.route_idx = 0

        route_path = st.session_state.route_path
        route_reasons = st.session_state.route_reasons
        route_texts = st.session_state.route_texts
        route_hops_to_show = st.session_state.route_idx
        succ_k = successor_of(st.session_state.key_id, st.session_state.active_nodes)

        left, right = st.columns([0.56, 0.44])
        with left:
            # keep fingers shown from Step 2 for continuity
            selected = st.session_state.selected
            fingers_shown = build_finger_table(selected, st.session_state.active_nodes, M)[:st.session_state.k]
            current_start = fingers_shown[-1].start if st.session_state.k > 0 else None
            fig = ring_figure(
                active_nodes=st.session_state.active_nodes, selected=selected,
                fingers=fingers_shown, highlight_start=current_start, show_radial=True,
                route_path=route_path, route_hops_to_show=route_hops_to_show, route_texts=route_texts,
                key=st.session_state.key_id,
                width=720 if projector else 680, height=720 if projector else 680,
                projector=projector, colorblind=colorblind, show_sectors=show_sectors
            )
            st.plotly_chart(fig, use_container_width=False)
            png = fig.to_image(format="png", scale=2)
            st.download_button("⬇️ Download ring as PNG", data=png, file_name="chord_route.png", mime="image/png")

        with right:
            st.markdown("### Step 3 — Search / Find the route")
            st.markdown('<div class="hint">Use <b>Start lookup</b>, then click <b>Next hop</b> to step through the route. Hover a hop line for the interval equation.</div>',
                        unsafe_allow_html=True)
            metr = st.container()
            with metr:
                m1, m2, m3 = st.columns(3)
                m1.metric("Start node", start_node)
                m2.metric("Key k", st.session_state.key_id)
                m3.metric("succ(k)", succ_k)
            st.markdown("**Path**")
            st.code(" → ".join(str(n) for n in route_path), language="text")
            st.markdown("**Reasoning (LaTeX)**")
            max_to_show = min(route_hops_to_show + 1, len(route_reasons))
            for i in range(max_to_show):
                st.latex(route_reasons[i])

            # --- Tiny Quiz Mode (optional) ---
            st.markdown("---")
            st.markdown("#### 🧪 Quick Quiz (optional)")
            if not st.session_state.quiz["show"]:
                if st.button("Generate quiz"):
                    if len(st.session_state.active_nodes) >= 2:
                        q_start = random.choice(st.session_state.active_nodes)
                        q_key = random.randint(0, 31)
                        q_route, q_reasons_latex, _ = chord_lookup_full(q_start, q_key, st.session_state.active_nodes, M)
                        st.session_state.quiz = {"start": q_start, "key": q_key, "show": True,
                                                 "answer": successor_of(q_key, st.session_state.active_nodes),
                                                 "route": q_route, "reasons": q_reasons_latex}
            else:
                q = st.session_state.quiz
                st.markdown(
                    f'<div class="quiz-card"><b>Quiz:</b> From start node <b>{q["start"]}</b>, route to key '
                    f'<b>{q["key"]}</b>. Who is responsible?</div>', unsafe_allow_html=True
                )
                guess = st.selectbox("Your guess for succ(key):", options=st.session_state.active_nodes)
                colg1, colg2 = st.columns([1,1])
                with colg1:
                    if st.button("Check"):
                        if guess == q["answer"]:
                            st.success("Correct! 🎉")
                        else:
                            st.error(f"Not quite. succ(key) = {q['answer']}.")
                with colg2:
                    if st.button("Reveal route"):
                        st.info("Route: " + " → ".join(map(str, q["route"])))
                        for rr in q["reasons"]:
                            st.latex(rr)
                if st.button("New quiz"):
                    st.session_state.quiz["show"] = False
