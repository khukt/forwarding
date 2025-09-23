# app_chord_tutor_click_join_v3.py
# Chord DHT — stepper tutor with click-to-join in Step 2 + ring shape fallback.
# Updated for Streamlit deprecation:
# - Replaced use_container_width with width='content'
# - Removed unsupported 'config' arg from plotly_events()

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

# ---------- Plotly config ----------
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
        "autoScale2d","resetScale2d","toggleSpikelines"
    ],
    "toImageButtonOptions": {"format":"png","filename":"chord_ring","height":900,"width":900,"scale":2},
}

# ---------- Constants ----------
M = 5
SPACE = 2 ** M
ALL_POSITIONS = list(range(SPACE))
SECTOR_COLORS = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
                 "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]

# ---------- Page ----------
st.set_page_config(page_title="Chord DHT • Stepper Tutor (click to join)", layout="wide", initial_sidebar_state="collapsed")
st.title("🔗 Chord DHT — Step-by-step Tutor")
st.caption("Step 1: Allocate → Step 2: Build finger table (click grey IDs to join) → Step 3: Search/route")

st.markdown("""
<style>
  .block-container { padding-top: 0.8rem; }
  .legend { color:#334155; font-size:0.95rem; }
  .tip { background:#f8fafc; border:1px solid #e2e8f0; padding:10px 12px; border-radius:10px; }
  .btn-row > div button { width:100%; height:42px; font-weight:600; }
  .chips span { background:#f1f5f9; padding:4px 8px; border-radius:8px; margin-right:6px; display:inline-block; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ---------- State ----------
def init_state():
    if "step" not in st.session_state: st.session_state.step = 1
    if "auto_advance" not in st.session_state: st.session_state.auto_advance = True

    if "active_nodes" not in st.session_state: st.session_state.active_nodes: List[int] = []
    if "allocated_nodes" not in st.session_state: st.session_state.allocated_nodes: List[int] = []
    if "selected" not in st.session_state: st.session_state.selected: Optional[int] = None
    if "fingers_revealed" not in st.session_state: st.session_state.fingers_revealed = 0

    if "key_id" not in st.session_state: st.session_state.key_id = 26
    if "route_path" not in st.session_state: st.session_state.route_path: List[int] = []
    if "route_reasons" not in st.session_state: st.session_state.route_reasons: List[str] = []
    if "route_texts" not in st.session_state: st.session_state.route_texts: List[str] = []
    if "route_idx" not in st.session_state: st.session_state.route_idx = 0

    if "search_show_fingers" not in st.session_state: st.session_state.search_show_fingers = False
    if "allow_remove_click" not in st.session_state: st.session_state.allow_remove_click = False
init_state()

# ---------- Chord helpers ----------
def sha1_mod(s: str, space: int) -> int:
    import hashlib
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % space

def mod_between(a: int, b: int, x: int, m: int, incl_right=False) -> bool:
    if a == b: return incl_right
    if a < b: return (a < x <= b) if incl_right else (a < x < b)
    return ((a < x <= m-1) or (0 <= x <= b)) if incl_right else ((a < x < m) or (0 <= x < b))

def successor_of(x: int, nodes_sorted: List[int]) -> int:
    for n in nodes_sorted:
        if n >= x:
            return n
    return nodes_sorted[0]

@dataclass
class Finger:
    i: int
    start: int
    node: int

def build_fingers(n: int, nodes: List[int], m: int) -> List[Finger]:
    out: List[Finger] = []
    if not nodes: return out
    for i in range(1, m+1):
        s = (n + 2**(i-1)) % (2**m)
        out.append(Finger(i=i, start=s, node=successor_of(s, nodes)))
    return out

def closest_preceding(n: int, fingers: List[int], target: int, m: int) -> int:
    for f in reversed(fingers):
        if f != n and mod_between(n, target, f, 2**m, incl_right=False):
            return f
    return n

def chord_route(start: int, key: int, nodes: List[int], m: int, max_steps=64):
    if not nodes: return [start], [r"\text{No active nodes.}"], ["No active nodes"]
    path = [start]; latex=[]; texts=[]
    succ_k = successor_of(key, nodes)
    fmap: Dict[int,List[int]] = {n:[f.node for f in build_fingers(n, nodes, m)] for n in nodes}
    while len(path) < max_steps:
        curr = path[-1]
        if curr == succ_k:
            latex.append(rf"\mathbf{{Stop}}:\ {curr}=\operatorname{{succ}}({key})")
            break
        idx = nodes.index(curr); curr_succ = nodes[(idx+1) % len(nodes)]
        if mod_between(curr, curr_succ, key, 2**m, incl_right=True):
            latex.append(rf"{key}\in({curr},{curr_succ}] \Rightarrow \text{{next}}=\operatorname{{succ}}({curr})={curr_succ}")
            texts.append(f"{key} in ({curr},{curr_succ}] ⇒ next = succ({curr}) = {curr_succ}")
            path.append(curr_succ)
            if curr_succ == succ_k: latex.append(rf"\mathbf{{Arrived}}\ \operatorname{{succ}}({key})={succ_k}")
            continue
        cpf = closest_preceding(curr, fmap[curr], key, m)
        if cpf == curr:
            latex.append(rf"\text{{No finger in }}({curr},{key}) \Rightarrow \text{{use succ}}({curr})={curr_succ}")
            texts.append(f"No finger in ({curr},{key}) ⇒ use succ({curr}) = {curr_succ}")
            path.append(curr_succ)
        else:
            latex.append(rf"\text{{Closest preceding finger of }}{curr}\ \text{{toward }}{key}: {cpf}\in({curr},{key})")
            texts.append(f"Closest preceding finger toward {key}: {cpf} ∈ ({curr},{key})")
            path.append(cpf)
        if path[-1] == succ_k:
            latex.append(rf"\mathbf{{Arrived}}\ \operatorname{{succ}}({key})={succ_k}")
            break
    return path, latex, texts

# ---------- Drawing ----------
def node_xy(i: int, space=SPACE, r=1.0):
    t = 2*math.pi*i/space
    return r*math.cos(t), r*math.sin(t)

def add_sector(fig, a, b, color, alpha=0.12, steps=40):
    if a == b: return
    start = a/SPACE*2*math.pi; end = b/SPACE*2*math.pi
    th = np.linspace(start, end, steps) if a < b else np.concatenate([np.linspace(start, 2*math.pi, steps//2), np.linspace(0, end, steps-steps//2)])
    R1, R2 = 1.05, 0.88
    xs = np.cos(th)*R1; ys = np.sin(th)*R1
    xs2 = np.cos(th[::-1])*R2; ys2 = np.sin(th[::-1])*R2
    fig.add_trace(go.Scatter(x=np.r_[xs,xs2], y=np.r_[ys,ys2], fill="toself", mode="lines",
                             line=dict(width=0), fillcolor=color, opacity=alpha, hoverinfo="skip", showlegend=False))

def ring_figure(
    active: List[int],
    selected: Optional[int] = None,
    fingers: Optional[List[Finger]] = None,
    show_start: Optional[int] = None,
    show_radial: bool = False,
    route_path: Optional[List[int]] = None,
    route_hops: int = 0,
    route_texts: Optional[List[str]] = None,
    key: Optional[int] = None,
    show_sectors: bool = False,
    multicolor: bool = True,
    allocated_nodes: Optional[List[int]] = None,
    width: int = 780,
    height: int = 780,
):
    COLORS = {
        "ring": "#334155", "disabled": "#9ca3af",
        "active": "#1f77b4", "allocated": "#22c55e",
        "selected": "#d62728", "succ": "#ff7f0e",
        "start": "#9467bd", "radial": "#6b7280", "hop": "#111827",
        "sector": "#3b82f6",
    }
    fig = go.Figure()

    # ring trace
    ang = np.linspace(0, 2*np.pi, 361)
    fig.add_trace(go.Scatter(x=np.cos(ang), y=np.sin(ang), mode="lines",
                             line=dict(color=COLORS["ring"], width=2.2), hoverinfo="skip", name="Ring"))
    # ring fallback shape (always visible)
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1.0, y0=-1.0, x1=1.0, y1=1.0,
                  line=dict(color=COLORS["ring"], width=2))

    # sectors
    if show_sectors and active:
        for idx, nid in enumerate(active):
            pred = active[idx-1]
            color = SECTOR_COLORS[idx % len(SECTOR_COLORS)] if multicolor else COLORS["sector"]
            add_sector(fig, pred, nid, color, alpha=0.12)

    # Disabled (clickable)
    disabled = [i for i in ALL_POSITIONS if i not in set(active)]
    if disabled:
        xs, ys = zip(*[node_xy(i) for i in disabled])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=[str(i) for i in disabled], textposition="top center",
            marker=dict(size=12, symbol="circle-open",
                        color=COLORS["disabled"], line=dict(width=1.2, color=COLORS["disabled"])),
            name="Disabled", opacity=0.65, hoverinfo="text",
            customdata=[{"id": i, "kind": "disabled"} for i in disabled]
        ))

    # Active (clickable)
    succ_k = successor_of(key, active) if (key is not None and active) else None
    if active:
        xs, ys = zip(*[node_xy(i) for i in active])
        sizes, colors, labels, cds = [], [], [], []
        allocated_set = set(allocated_nodes or [])
        for nid in active:
            if selected == nid:
                sizes.append(22); colors.append(COLORS["selected"]); labels.append(f"{nid} (selected)")
            elif succ_k == nid:
                sizes.append(20); colors.append(COLORS["succ"]); labels.append(f"{nid} (succ(key))")
            elif allocated_set and nid in allocated_set:
                sizes.append(18); colors.append(COLORS["allocated"]); labels.append(f"{nid} (just allocated)")
            else:
                sizes.append(18); colors.append(COLORS["active"]); labels.append(str(nid))
            cds.append({"id": nid, "kind": "active"})
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=[str(n) for n in active], textposition="top center",
            hovertext=labels, hoverinfo="text",
            marker=dict(size=sizes, color=colors, line=dict(width=1.3, color="white")),
            name="Active",
            customdata=cds
        ))

    # Fingers
    if selected is not None and fingers:
        sx, sy = node_xy(selected)
        for f in fingers:
            tx, ty = node_xy(f.node)
            fig.add_trace(go.Scatter(
                x=[sx, tx], y=[sy, ty], mode="lines",
                line=dict(width=3, dash="dot", color="#1f77b4"),
                hovertext=f"start={f.start} → succ={f.node}", hoverinfo="text",
                showlegend=False
            ))

    # Highlight start[i]
    if show_start is not None and selected is not None:
        hx, hy = node_xy(show_start)
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy], mode="markers+text",
            text=[f"start={show_start}"], textposition="bottom center",
            marker=dict(size=18, symbol="diamond", line=dict(width=1.2, color="black"), color=COLORS["start"]),
            showlegend=False
        ))
        if show_radial:
            sx, sy = node_xy(selected)
            fig.add_trace(go.Scatter(
                x=[sx, hx], y=[sy, hy], mode="lines",
                line=dict(width=1.5, dash="dash", color=COLORS["radial"]),
                showlegend=False
            ))

    # Route (if any)
    if route_path and route_hops > 0:
        for i in range(min(route_hops, len(route_path)-1)):
            a, b = route_path[i], route_path[i+1]
            ax, ay = node_xy(a); bx, by = node_xy(b)
            tip = route_texts[i] if (route_texts and i < len(route_texts)) else f"{a} → {b}"
            fig.add_trace(go.Scatter(
                x=[ax,bx], y=[ay,by], mode="lines+markers",
                line=dict(width=4, color=COLORS["hop"]),
                marker=dict(size=8, color=COLORS["hop"]),
                hovertext=tip, hoverinfo="text", showlegend=False
            ))
            fig.add_annotation(x=bx, y=by, ax=ax, ay=ay, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2, arrowcolor=COLORS["hop"])

    fig.update_layout(
        width=780, height=780,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=8,r=8,t=30,b=8),
        plot_bgcolor="white",
        title="Chord • Ring 0..31",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=11)),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# ---------- Stepper ----------
step_labels = ["① Allocate", "② Fingers", "③ Search"]
c1, c2 = st.columns([0.7, 0.3])
with c1:
    sel = st.radio("Steps", step_labels, index=st.session_state.step-1, horizontal=True,
                   label_visibility="collapsed", key="step_radio")
with c2:
    st.session_state.auto_advance = st.toggle("Auto-advance after Allocate", value=st.session_state.auto_advance)
st.session_state.step = step_labels.index(sel) + 1

# ========= STEP 1: Allocate =========
if st.session_state.step == 1:
    left, right = st.columns([0.6, 0.4], gap="large")

    with right:
        st.markdown("### Step 1 — Allocate nodes")
        st.markdown('<div class="tip">Enter labels and click <b>Allocate</b>. IDs are <code>SHA1(label) mod 32</code>. Initially, all positions are disabled.</div>', unsafe_allow_html=True)
        labels = st.text_area("Node labels", value="nodeA\nnodeB\nnodeC\nnodeD", height=140)

        allocated_now = False
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Allocate"):
                lab = [s.strip() for s in labels.splitlines() if s.strip()]
                ids = sorted(set(sha1_mod(x, SPACE) for x in lab))
                st.session_state.active_nodes = ids
                st.session_state.allocated_nodes = ids[:]
                st.session_state.selected = ids[0] if ids else None
                st.session_state.fingers_revealed = 0
                allocated_now = True
        with b2:
            if st.button("Load example"):
                ids = [1,4,9,11,14,18,20,21,28]
                st.session_state.active_nodes = ids
                st.session_state.allocated_nodes = ids[:]
                st.session_state.selected = 1
                st.session_state.fingers_revealed = 0
                allocated_now = True
        with b3:
            if st.button("Clear"):
                st.session_state.active_nodes = []
                st.session_state.allocated_nodes = []
                st.session_state.selected = None
                st.session_state.fingers_revealed = 0

        manual = st.text_input("Manual IDs (optional, e.g. 1,4,9,11)")
        if st.button("Use manual IDs"):
            raw = [t.strip() for t in manual.replace(",", " ").split()]
            try:
                ids = sorted(set(int(x) % SPACE for x in raw if x != ""))
                st.session_state.active_nodes = ids
                st.session_state.allocated_nodes = ids[:]
                st.session_state.selected = ids[0] if ids else None
                st.session_state.fingers_revealed = 0
                allocated_now = True
            except ValueError:
                st.warning("Manual IDs must be integers 0–31.")

        if st.session_state.active_nodes:
            st.write("**Active IDs:**")
            st.markdown('<div class="chips">'+" ".join(f"<span>{n}</span>" for n in st.session_state.active_nodes)+"</div>", unsafe_allow_html=True)

        st.markdown("**Hash equation**"); st.latex(r"\text{node\_id} = \operatorname{SHA1}(\text{label}) \bmod 32")

        if st.session_state.active_nodes:
            if st.button("Proceed to Step 2 ▸ Fingers"):
                st.session_state.step = 2; st.rerun()
        if allocated_now and st.session_state.auto_advance and st.session_state.active_nodes:
            st.session_state.step = 2; st.rerun()

        st.markdown("""<div class="legend"><b>Legend:</b>
        <span style="color:#22c55e">● Just allocated</span>,
        <span style="color:#1f77b4">● Active</span>,
        <span style="color:#d62728">● Selected</span>,
        <span style="color:#ff7f0e">● succ(key)</span>,
        <span style="color:#9ca3af">◌ Disabled</span></div>""", unsafe_allow_html=True)

    with left:
        fig = ring_figure(active=st.session_state.active_nodes, selected=st.session_state.selected,
                          allocated_nodes=st.session_state.allocated_nodes, show_sectors=False, multicolor=True)
        st.plotly_chart(fig, width="content", config=PLOTLY_CONFIG, key="ring_step1")

# ========= STEP 2: Fingers (click to join/leave/select) =========
elif st.session_state.step == 2:
    if st.session_state.allocated_nodes: st.session_state.allocated_nodes = []

    left, right = st.columns([0.6, 0.4], gap="large")

    with right:
        st.markdown("### Step 2 — Finger table")
        if not st.session_state.active_nodes:
            st.info("Load the example in Step 1 or allocate labels first.")
        else:
            st.session_state.selected = st.selectbox(
                "Node n", st.session_state.active_nodes,
                index=0 if (st.session_state.selected not in st.session_state.active_nodes or st.session_state.selected is None)
                else st.session_state.active_nodes.index(st.session_state.selected)
            )
            st.session_state.allow_remove_click = st.toggle("Allow remove on click (active → disabled)", value=False)

            f_all = build_fingers(st.session_state.selected, st.session_state.active_nodes, M)
            k = st.session_state.fingers_revealed
            f_show = f_all[:k]

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Reveal next"):
                    st.session_state.fingers_revealed = min(M, k+1)
            with c2:
                if st.button("Reveal all"):
                    st.session_state.fingers_revealed = M
            with c3:
                if st.button("Proceed to Step 3 ▸ Search"):
                    st.session_state.step = 3; st.rerun()

            df = pd.DataFrame([{"i": f.i, "start": f.start, "successor": f.node} for f in f_show],
                              columns=["i","start","successor"])
            st.dataframe(df, hide_index=True, height=240, use_container_width=True, key="ft_table")

            if k > 0:
                fe = f_show[-1]
                st.markdown("**Current entry**")
                st.latex(rf"n={st.session_state.selected}")
                st.latex(rf"\text{{start}}[{fe.i}] = (n + 2^{{{fe.i-1}}}) \bmod 32 = {fe.start}")
                st.latex(rf"\text{{finger}}[{fe.i}] = \operatorname{{succ}}({fe.start}) = {fe.node}")
            else:
                st.caption("Click **Reveal next** to build the table.")

            st.caption("Tip: Click a grey ID to join (becomes blue + selected red). Click a blue ID to select it. Enable removal to click a blue ID and remove it.")

    with left:
        if not st.session_state.active_nodes:
            fig = ring_figure(active=[], show_sectors=False)
            st.plotly_chart(fig, width="content", config=PLOTLY_CONFIG, key="ring_step2_empty")
        else:
            f_all = build_fingers(st.session_state.selected, st.session_state.active_nodes, M)
            k = st.session_state.fingers_revealed
            fig = ring_figure(
                active=st.session_state.active_nodes,
                selected=st.session_state.selected,
                fingers=f_all[:k],
                show_start=(f_all[k-1].start if k>0 else None),
                show_radial=True,
                show_sectors=True, multicolor=True
            )
            # CLICK HANDLER (select/join/remove)
            clicked = plotly_events(
                fig, click_event=True, select_event=False,
                override_width=780, override_height=780,
                key="ring_step2_click"   # NOTE: no 'config' arg here
            )
            if clicked:
                meta = clicked[0].get("customdata")  # {'id': int, 'kind': 'disabled'|'active'}
                if isinstance(meta, dict) and "id" in meta and "kind" in meta:
                    nid = int(meta["id"]); kind = meta["kind"]
                    nodes = set(st.session_state.active_nodes)
                    if kind == "disabled":
                        nodes.add(nid)  # join and select
                        st.session_state.active_nodes = sorted(nodes)
                        st.session_state.selected = nid
                        st.rerun()
                    elif kind == "active":
                        if st.session_state.allow_remove_click:
                            if len(nodes) > 1:
                                nodes.remove(nid)
                                st.session_state.active_nodes = sorted(nodes)
                                if st.session_state.selected not in nodes:
                                    st.session_state.selected = st.session_state.active_nodes[0]
                                st.rerun()
                            else:
                                st.warning("Cannot remove the last active node.")
                        else:
                            # Just select (turns red)
                            st.session_state.selected = nid
                            st.rerun()

# ========= STEP 3: Search =========
else:
    if st.session_state.allocated_nodes: st.session_state.allocated_nodes = []
    left, right = st.columns([0.6, 0.4], gap="large")

    with right:
        st.markdown("### Step 3 — Search / Route")
        if not st.session_state.active_nodes:
            st.info("Load the example in Step 1 or allocate labels first.")
        else:
            start = st.selectbox("Start node", st.session_state.active_nodes, index=0, key="start_node_select")
            k = st.number_input("Key k (0–31)", min_value=0, max_value=31, value=st.session_state.key_id, step=1)

            st.session_state.search_show_fingers = st.checkbox("Show fingers while searching (may clutter)", value=False)

            colb1, colb2 = st.columns(2)
            with colb1:
                if st.button("Route"):
                    path, reasons, texts = chord_route(start, k, st.session_state.active_nodes, M)
                    st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_texts = path, reasons, texts
                    st.session_state.route_idx = 0
                    st.session_state.key_id = k
            with colb2:
                if st.button("Next hop"):
                    if st.session_state.route_path:
                        st.session_state.route_idx = min(len(st.session_state.route_path)-1, st.session_state.route_idx+1)

            if not st.session_state.route_path and st.session_state.active_nodes:
                path, reasons, texts = chord_route(start, k, st.session_state.active_nodes, M)
                st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_texts = path, reasons, texts
                st.session_state.route_idx = 0

            succ_k = successor_of(k, st.session_state.active_nodes)
            m1, m2, m3 = st.columns(3)
            m1.metric("Start", start); m2.metric("Key k", k); m3.metric("succ(k)", succ_k)

            st.markdown("**Path**")
            st.code(" → ".join(map(str, st.session_state.route_path)), language="text")

            st.markdown("**Reasoning (per hop)**")
            for i in range(min(st.session_state.route_idx+1, len(st.session_state.route_reasons))):
                st.latex(st.session_state.route_reasons[i])

            st.caption("Hover a hop line on the ring to see the interval rule used for that hop.")

    with left:
        if not st.session_state.active_nodes:
            fig = ring_figure(active=[], show_sectors=False)
            st.plotly_chart(fig, width="content", config=PLOTLY_CONFIG, key="ring_step3_empty")
        else:
            if st.session_state.search_show_fingers:
                sel = st.session_state.selected if (st.session_state.selected in st.session_state.active_nodes and st.session_state.selected is not None) else st.session_state.active_nodes[0]
                f_all = build_fingers(sel, st.session_state.active_nodes, M)
                fingers_for_search = f_all[:st.session_state.fingers_revealed]
                show_start = f_all[st.session_state.fingers_revealed-1].start if st.session_state.fingers_revealed>0 else None
                selected = sel
            else:
                fingers_for_search = None; show_start = None; selected = None

            fig = ring_figure(
                active=st.session_state.active_nodes,
                selected=selected,
                fingers=fingers_for_search,
                show_start=show_start,
                show_radial=False,
                route_path=st.session_state.route_path,
                route_hops=st.session_state.route_idx,
                route_texts=st.session_state.route_texts,
                key=st.session_state.key_id,
                show_sectors=True, multicolor=True
            )
            st.plotly_chart(fig, width="content", config=PLOTLY_CONFIG, key="ring_step3")
