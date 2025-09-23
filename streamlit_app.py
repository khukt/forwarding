# app_chord_tutor_pro_fix_order.py
# Fix: define read_state_from_url()/write_state_to_url() BEFORE init_state()

import math, time, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

# -------------------- Config --------------------
st.set_page_config(page_title="Chord DHT • Pro Tutor", layout="wide", initial_sidebar_state="collapsed")

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
        "autoScale2d","resetScale2d","toggleSpikelines"
    ],
    "toImageButtonOptions": {"format":"png","filename":"chord_ring","height":900,"width":900,"scale":2},
}

# Color palettes
PALETTE_NORMAL = {
    "ring": "#334155", "disabled": "#9ca3af",
    "active": "#1f77b4", "allocated": "#22c55e",
    "selected": "#d62728", "succ": "#ff7f0e",
    "start": "#9467bd", "radial": "#6b7280", "hop": "#111827",
    "sector": "#3b82f6",
    "pill": "#0f172a",
}
PALETTE_CBLIND = {
    "ring": "#3B3B3B", "disabled": "#9c9c9c",
    "active": "#0072B2", "allocated": "#009E73",
    "selected": "#D55E00", "succ": "#CC79A7",
    "start": "#9467bd", "radial": "#6b7280", "hop": "#2b2b2b",
    "sector": "#56B4E9",
    "pill": "#000000",
}
SECTOR_COLORS_A = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
                   "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]
SECTOR_COLORS_B = ["#0072B2","#009E73","#E69F00","#56B4E9","#F0E442",
                   "#D55E00","#CC79A7","#999999","#009E73","#0072B2"]

# -------------------- Constants --------------------
M = 5
SPACE = 2 ** M
ALL_POSITIONS = list(range(SPACE))

# -------------------- Utilities --------------------
def sha1_mod(s: str, space: int) -> int:
    import hashlib
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % space

def node_xy(i: int, space=SPACE, r=1.0) -> Tuple[float,float]:
    t = 2*math.pi*i/space
    return r*math.cos(t), r*math.sin(t)

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
            latex.append(rf"\mathbf{{Stop}}:\ {curr}=\operatorname{{succ}}({key})"); texts.append("Arrived.")
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
            texts.append(f"No finger in ({curr},{key}) ⇒ succ({curr}) = {curr_succ}")
            path.append(curr_succ)
        else:
            latex.append(rf"\text{{Closest preceding finger of }}{curr}\ \text{{toward }}{key}: {cpf}\in({curr},{key})")
            texts.append(f"Closest preceding finger toward {key}: {cpf} ∈ ({curr},{key})")
            path.append(cpf)
        if path[-1] == succ_k:
            latex.append(rf"\mathbf{{Arrived}}\ \operatorname{{succ}}({key})={succ_k}")
            break
    return path, latex, texts

def add_sector(fig, a, b, color, alpha=0.12, steps=40):
    if a == b: return
    start = a/SPACE*2*math.pi; end = b/SPACE*2*math.pi
    th = np.linspace(start, end, steps) if a < b else np.concatenate([
        np.linspace(start, 2*math.pi, steps//2), np.linspace(0, end, steps-steps//2)
    ])
    R1, R2 = 1.05, 0.88
    xs = np.cos(th)*R1; ys = np.sin(th)*R1
    xs2 = np.cos(th[::-1])*R2; ys2 = np.sin(th[::-1])*R2
    fig.add_trace(go.Scatter(x=np.r_[xs,xs2], y=np.r_[ys,ys2], fill="toself", mode="lines",
                             line=dict(width=0), fillcolor=color, opacity=alpha,
                             hoverinfo="text", hovertext="", showlegend=False))

# -------------------- URL share (MOVED UP) --------------------
def read_state_from_url():
    qp = st.experimental_get_query_params()
    if not qp: return
    ss = st.session_state
    try:
        if "nodes" in qp:
            ids = sorted(set(int(x) % SPACE for x in json.loads(qp["nodes"][0])))
            ss["active_nodes"] = ids
        if "sel" in qp and qp["sel"][0] != "":
            ss["selected"] = int(qp["sel"][0])
        if "k" in qp:
            ss["key_id"] = int(qp["k"][0])
        if "step" in qp:
            ss["step"] = int(qp["step"][0])
    except Exception:
        pass

def write_state_to_url():
    ss = st.session_state
    st.experimental_set_query_params(
        nodes=json.dumps(ss.active_nodes),
        sel=ss.selected if ss.selected is not None else "",
        k=ss.key_id, step=ss.step
    )

# -------------------- State --------------------
def init_state():
    ss = st.session_state
    ss.setdefault("mode", "Student")  # or Explainer
    ss.setdefault("step", 1)
    ss.setdefault("auto_advance", True)
    ss.setdefault("color_blind", False)

    ss.setdefault("active_nodes", [])
    ss.setdefault("allocated_nodes", [])
    ss.setdefault("selected", None)
    ss.setdefault("fingers_revealed", 0)

    ss.setdefault("key_id", 26)
    ss.setdefault("route_path", [])
    ss.setdefault("route_reasons", [])
    ss.setdefault("route_texts", [])
    ss.setdefault("route_idx", 0)
    ss.setdefault("route_play", False)
    ss.setdefault("last_tick", 0.0)

    # tours
    ss.setdefault("tour_seen_step1", False)
    ss.setdefault("tour_seen_step2", False)
    ss.setdefault("tour_seen_step3", False)

    # quizzes
    ss.setdefault("quiz1_k", 7)
    ss.setdefault("quiz2_i", 4)
    ss.setdefault("quiz3_next_from", None)

    # URL state
    read_state_from_url()
init_state()

def palette():
    return PALETTE_CBLIND if st.session_state.color_blind else PALETTE_NORMAL

def sector_palette():
    return SECTOR_COLORS_B if st.session_state.color_blind else SECTOR_COLORS_A

# -------------------- Drawing --------------------
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
):
    COLORS = palette()
    fig = go.Figure()

    # ring trace + fallback shape
    ang = np.linspace(0, 2*np.pi, 361)
    fig.add_trace(go.Scatter(x=np.cos(ang), y=np.sin(ang), mode="lines",
                             line=dict(color=COLORS["ring"], width=2.2), hoverinfo="skip", name="Ring"))
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1.0, y0=-1.0, x1=1.0, y1=1.0,
                  line=dict(color=COLORS["ring"], width=2))

    # sectors + hover text
    if show_sectors and active:
        sp = sector_palette()
        for idx, nid in enumerate(active):
            pred = active[idx-1]
            color = sp[idx % len(sp)]
            add_sector(fig, pred, nid, color, alpha=0.12)
            fig.data[-1].hovertext = f"node {nid} owns ({pred}, {nid}]"

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
            name="Active", customdata=cds
        ))

    # Fingers
    if selected is not None and fingers:
        sx, sy = node_xy(selected)
        for j, f in enumerate(fingers):
            tx, ty = node_xy(f.node)
            fig.add_trace(go.Scatter(
                x=[sx, tx], y=[sy, ty], mode="lines",
                line=dict(width=4 if j==len(fingers)-1 else 3,
                          dash="dot", color=palette()["active"]),
                hovertext=f"start[{f.i}]={f.start} → succ={f.node}", hoverinfo="text",
                showlegend=False
            ))

    # Highlight start[i]
    if show_start is not None and selected is not None:
        hx, hy = node_xy(show_start)
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy], mode="markers+text",
            text=[f"start={show_start}"], textposition="bottom center",
            marker=dict(size=18, symbol="diamond", line=dict(width=1.2, color="black"), color=palette()["start"]),
            showlegend=False
        ))
        if show_radial:
            sx, sy = node_xy(selected)
            fig.add_trace(go.Scatter(
                x=[sx, hx], y=[sy, hy], mode="lines",
                line=dict(width=1.5, dash="dash", color=palette()["radial"]),
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
                line=dict(width=4, color=palette()["hop"]),
                marker=dict(size=8, color=palette()["hop"]),
                hovertext=tip, hoverinfo="text", showlegend=False
            ))
            fig.add_annotation(x=bx, y=by, ax=ax, ay=ay, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.05, arrowwidth=2, arrowcolor=palette()["hop"])

    fig.update_layout(
        width=780, height=780,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=8,r=8,t=30,b=8), plot_bgcolor="white",
        title="Chord • Ring 0..31",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=11)),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# -------------------- UI Helpers --------------------
def pills_path(path: List[int], current_idx: int):
    if not path: st.write("—"); return
    col = st.container()
    items = []
    for i, v in enumerate(path):
        style = "font-weight:700;" if i <= current_idx else ""
        items.append(f"""<span style="border:1px solid #e2e8f0;border-radius:999px;padding:4px 10px;margin-right:6px;{style}">{v}</span>""")
        if i < len(path)-1: items.append("→")
    col.markdown("".join(items), unsafe_allow_html=True)

def preset_row():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Example fig (9 nodes)", help="1,4,9,11,14,18,20,21,28"):
            ids = [1,4,9,11,14,18,20,21,28]
            st.session_state.active_nodes = ids
            st.session_state.selected = ids[0]
            st.session_state.fingers_revealed = 0
    with c2:
        if st.button("Sparse (6 random)"):
            ids = sorted(np.random.choice(ALL_POSITIONS, 6, replace=False).tolist())
            st.session_state.active_nodes = ids
            st.session_state.selected = ids[0]
            st.session_state.fingers_revealed = 0
    with c3:
        if st.button("Clustered"):
            base = np.random.randint(0, SPACE)
            ids = sorted({base, (base+1)%SPACE, (base+2)%SPACE, (base+5)%SPACE, (base+6)%SPACE, (base+10)%SPACE})
            st.session_state.active_nodes = ids
            st.session_state.selected = ids[0]
            st.session_state.fingers_revealed = 0
    with c4:
        if st.button("Random 12"):
            ids = sorted(np.random.choice(ALL_POSITIONS, 12, replace=False).tolist())
            st.session_state.active_nodes = ids
            st.session_state.selected = ids[0]
            st.session_state.fingers_revealed = 0

# -------------------- Header --------------------
topL, topR = st.columns([0.7, 0.3])
with topL:
    st.title("🔗 Chord DHT — Pro Tutor")
    st.caption("Step 1: Allocate → Step 2: Build finger table → Step 3: Search/route")
with topR:
    st.session_state.mode = st.selectbox("Mode", ["Student","Explainer"], index=0 if st.session_state.mode=="Student" else 1)
    st.session_state.color_blind = st.toggle("Color-blind palette", value=st.session_state.color_blind)

# Stepper
labels = ["① Allocate", "② Fingers", "③ Search"]
sel = st.radio("Steps", labels, index=st.session_state.step-1, horizontal=True, label_visibility="collapsed")
st.session_state.step = labels.index(sel) + 1
st.session_state.auto_advance = st.toggle("Auto-advance after Allocate", value=st.session_state.auto_advance)

# -------------------- STEP 1 --------------------
if st.session_state.step == 1:
    left, right = st.columns([0.60, 0.40], gap="large")

    with right:
        st.markdown("### Step 1 — Allocate nodes")
        if not st.session_state.tour_seen_step1:
            st.info("① Paste labels → ② Click **Allocate** (IDs = SHA1(label) mod 32) → ③ Use **Presets** to try patterns.", icon="🎓")
            if st.button("Got it"):
                st.session_state.tour_seen_step1 = True
        st.markdown('<div class="tip">IDs are <code>SHA1(label) mod 32</code>. Initially, all positions are disabled.</div>', unsafe_allow_html=True)

        preset_row()

        labels_box = st.text_area("Node labels", value="nodeA\nnodeB\nnodeC\nnodeD", height=120)
        r1, r2, r3 = st.columns(3)
        allocated_now = False
        with r1:
            if st.button("Allocate"):
                labs = [s.strip() for s in labels_box.splitlines() if s.strip()]
                ids = sorted(set(sha1_mod(x, SPACE) for x in labs))
                st.session_state.active_nodes = ids
                st.session_state.allocated_nodes = ids[:]
                st.session_state.selected = ids[0] if ids else None
                st.session_state.fingers_revealed = 0
                allocated_now = True
        with r2:
            if st.button("Clear"):
                st.session_state.active_nodes = []; st.session_state.allocated_nodes = []
                st.session_state.selected = None; st.session_state.fingers_revealed = 0
        with r3:
            if st.button("Proceed ▸ Step 2"):
                st.session_state.step = 2; write_state_to_url(); st.rerun()

        manual = st.text_input("Manual IDs (optional, e.g. 1,4,9,11)")
        if st.button("Use manual IDs"):
            try:
                ids = sorted(set(int(x) % SPACE for x in manual.replace(",", " ").split() if x.strip()!=""))
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

        with st.expander("Quick check: which node stores key k?"):
            k = st.number_input("Choose key k", 0, 31, value=st.session_state.quiz1_k)
            st.session_state.quiz1_k = k
            if st.session_state.active_nodes:
                owner = successor_of(k, st.session_state.active_nodes)
                st.write(f"**Answer:** successor({k}) = **{owner}**")
            else:
                st.caption("Allocate or load a preset to try this.")

        if allocated_now and st.session_state.auto_advance and st.session_state.active_nodes:
            st.session_state.step = 2; write_state_to_url(); st.rerun()

    with left:
        fig = ring_figure(
            active=st.session_state.active_nodes,
            selected=st.session_state.selected,
            allocated_nodes=st.session_state.allocated_nodes,
            show_sectors=False
        )
        st.plotly_chart(fig, width="content", config=PLOTLY_CONFIG, key="fig_step1")

# -------------------- STEP 2 --------------------
elif st.session_state.step == 2:
    if st.session_state.allocated_nodes: st.session_state.allocated_nodes = []
    left, right = st.columns([0.60, 0.40], gap="large")

    with right:
        st.markdown("### Step 2 — Finger table")
        if not st.session_state.tour_seen_step2:
            st.info("Click a **grey** ID to **join** (blue + selected red). Click a **blue** ID to **select**. Enable removal to click blue and remove.", icon="🖱️")
            if st.button("Got it  ✓"):
                st.session_state.tour_seen_step2 = True

        if not st.session_state.active_nodes:
            st.info("Load a preset or allocate labels first.")
        else:
            st.session_state.selected = st.selectbox(
                "Node n", st.session_state.active_nodes,
                index=0 if (st.session_state.selected not in st.session_state.active_nodes or st.session_state.selected is None)
                else st.session_state.active_nodes.index(st.session_state.selected)
            )
            allow_remove = st.toggle("Allow remove on click (active → disabled)", value=False)
            st.session_state.allow_remove_click = allow_remove

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
                if st.button("Proceed ▸ Step 3"):
                    st.session_state.step = 3; write_state_to_url(); st.rerun()

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

            with st.expander("Self-check: compute start[i] for this n"):
                n = st.session_state.selected if st.session_state.selected is not None else 0
                i_val = st.number_input("i (1..5)", 1, 5, value=st.session_state.quiz2_i)
                st.session_state.quiz2_i = i_val
                calc = (n + 2**(i_val-1)) % 32
                succ = successor_of(calc, st.session_state.active_nodes) if st.session_state.active_nodes else None
                st.latex(rf"\text{{start}}[{i_val}] = (n + 2^{{{i_val-1}}}) \bmod 32 = {calc}")
                if succ is not None: st.latex(rf"\text{{finger}}[{i_val}] = \operatorname{{succ}}({calc}) = {succ}")

    with left:
        if not st.session_state.active_nodes:
            fig = ring_figure(active=[], show_sectors=False)
            st.plotly_chart(fig, width="content", config=PLOTLY_CONFIG, key="fig_step2_empty")
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
            clicked = plotly_events(fig, click_event=True, select_event=False, override_width=780, override_height=780, key="fig_step2_click")
            if clicked:
                meta = clicked[0].get("customdata")
                if isinstance(meta, dict) and "id" in meta and "kind" in meta:
                    nid, kind = int(meta["id"]), meta["kind"]
                    nodes = set(st.session_state.active_nodes)
                    if kind == "disabled":
                        nodes.add(nid); st.session_state.active_nodes = sorted(nodes)
                        st.session_state.selected = nid; st.rerun()
                    elif kind == "active":
                        if st.session_state.allow_remove_click:
                            if len(nodes) > 1:
                                nodes.remove(nid); st.session_state.active_nodes = sorted(nodes)
                                if st.session_state.selected not in nodes:
                                    st.session_state.selected = st.session_state.active_nodes[0]
                                st.rerun()
                            else:
                                st.warning("Cannot remove the last active node.")
                        else:
                            st.session_state.selected = nid; st.rerun()

# -------------------- STEP 3 --------------------
else:
    left, right = st.columns([0.60, 0.40], gap="large")
    if not st.session_state.tour_seen_step3:
        st.info("Press **Route** then use **Play** or **Next hop**. Click the **path pills** to jump to any hop.", icon="🎯")
        if st.button("Got it ✓"):
            st.session_state.tour_seen_step3 = True

    with right:
        st.markdown("### Step 3 — Search / Route")
        if not st.session_state.active_nodes:
            st.info("Load a preset or allocate labels first (Step 1).")
        else:
            start = st.selectbox("Start node", st.session_state.active_nodes, index=0, key="start_node_select")
            k = st.number_input("Key k (0–31)", 0, 31, value=st.session_state.key_id)
            st.session_state.key_id = k

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Route"):
                    path, reasons, texts = chord_route(start, k, st.session_state.active_nodes, M)
                    st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_texts = path, reasons, texts
                    st.session_state.route_idx = 0
                    st.session_state.route_play = False
            with b2:
                if st.button("Next hop"):
                    if st.session_state.route_path:
                        st.session_state.route_idx = min(len(st.session_state.route_path)-1, st.session_state.route_idx+1)
            with b3:
                if st.session_state.route_play:
                    if st.button("⏸ Pause"): st.session_state.route_play = False
                else:
                    if st.button("▶ Play"): st.session_state.route_play = True

            if st.session_state.route_play and st.session_state.route_path:
                now = time.time()
                if now - st.session_state.last_tick > 0.6:
                    st.session_state.last_tick = now
                    if st.session_state.route_idx < len(st.session_state.route_path)-1:
                        st.session_state.route_idx += 1
                    else:
                        st.session_state.route_play = False
                    st.experimental_rerun()

            if not st.session_state.route_path and st.session_state.active_nodes:
                path, reasons, texts = chord_route(start, k, st.session_state.active_nodes, M)
                st.session_state.route_path, st.session_state.route_reasons, st.session_state.route_texts = path, reasons, texts
                st.session_state.route_idx = 0

            succ_k = successor_of(k, st.session_state.active_nodes)
            m1, m2, m3 = st.columns(3)
            m1.metric("Start", start); m2.metric("Key k", k); m3.metric("succ(k)", succ_k)

            st.markdown("**Path**")
            # pills
            if st.session_state.route_path:
                items = []
                for i, v in enumerate(st.session_state.route_path):
                    bold = i <= st.session_state.route_idx
                    style = "font-weight:700;" if bold else ""
                    items.append(f"""<span style="border:1px solid #e2e8f0;border-radius:999px;padding:4px 10px;margin-right:6px;{style}">{v}</span>""")
                    if i < len(st.session_state.route_path)-1: items.append("→")
                st.markdown("".join(items), unsafe_allow_html=True)
            else:
                st.write("—")

            st.markdown("**Reasoning (per hop)**")
            for i in range(min(st.session_state.route_idx+1, len(st.session_state.route_reasons))):
                st.latex(st.session_state.route_reasons[i])
            st.caption("Hover a hop line on the ring to see the interval rule used for that hop.")

            with st.expander("Self-check: predict next hop"):
                if len(st.session_state.route_path) >= 1:
                    cur = st.session_state.route_path[min(st.session_state.route_idx, len(st.session_state.route_path)-1)]
                    st.write(f"Current node: **{cur}**, key **k={k}**")
                    if st.button("Show expected next"):
                        nodes = st.session_state.active_nodes
                        idx = nodes.index(cur); curr_succ = nodes[(idx+1) % len(nodes)]
                        if mod_between(cur, curr_succ, k, 2**M, incl_right=True):
                            st.success(f"Next = succ({cur}) = {curr_succ}")
                        else:
                            fmap = [f.node for f in build_fingers(cur, nodes, M)]
                            cpf = closest_preceding(cur, fmap, k, M)
                            nxt = curr_succ if cpf == cur else cpf
                            st.success(f"Next = {nxt} (closest preceding finger = {cpf})")
                else:
                    st.caption("Click Route first.")

    with left:
        if not st.session_state.active_nodes:
            fig = ring_figure(active=[], show_sectors=False)
            st.plotly_chart(fig, width="content", config=PLOTLY_CONFIG, key="fig_step3_empty")
        else:
            fig = ring_figure(
                active=st.session_state.active_nodes,
                selected=None,
                fingers=None,
                show_sectors=True, multicolor=True,
                route_path=st.session_state.route_path,
                route_hops=st.session_state.route_idx,
                route_texts=st.session_state.route_texts,
                key=st.session_state.key_id,
            )
            st.plotly_chart(fig, width="content", config=PLOTLY_CONFIG, key="fig_step3")

# persist sharable state in URL
write_state_to_url()
