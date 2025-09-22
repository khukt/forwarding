
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

st.set_page_config(page_title="Distributed Systems Tutor — Forwarding Pointers (Dual View)", layout="wide")

# =========================
# Timeline frames (0..5)
# =========================
# 0: Baseline: P1 → P4 (object at P4)
# 1: Migration: object moves P4→P2, forwarding pointer P4→P2
# 2: First call after migration: P1 → P4 → P2 (forwarded)
# 3: Shortcut installed: P1 → P2
# 4: Second client P1' appears (identical proxy) still pointing to P4; first call forwarded P1' → P4 → P2, then shortcut
# 5: Both clients direct: P1 → P2, P1' → P2; forwarding cleaned

FRAMES = [
    {"name":"0. Direct call", "obj":"P4", "fwd":{}, "invocations":[("P1","P4")]},
    {"name":"1. Migration (leave forward)", "obj":"P2", "fwd":{"P4":"P2"}, "invocations":[]},
    {"name":"2. First call (forwarded)", "obj":"P2", "fwd":{"P4":"P2"}, "invocations":[("P1","P4","P2")]},
    {"name":"3. Shortcut installed", "obj":"P2", "fwd":{}, "invocations":[("P1","P2")]},
    {"name":"4. Identical proxy (P1') first call", "obj":"P2", "fwd":{"P4":"P2"}, "invocations":[("P1p","P4","P2")]},
    {"name":"5. Both direct; cleaned up", "obj":"P2", "fwd":{}, "invocations":[("P1","P2"),("P1p","P2")]},
]

# -------------------------
# Positions & drawing utils
# -------------------------
POS = {
    "P1":  (-1.6,  0.6),
    "P1p": (-1.6, -0.6),
    "P2":  ( 1.0,  1.0),
    "P3":  ( 1.0,  0.0),
    "P4":  ( 1.0, -1.0),
}

def draw_box(ax, center, label_top, label_bottom, highlight=False):
    x,y = center
    ax.add_patch(plt.Rectangle((x-0.55,y-0.22), 1.1, 0.44, fill=False, linewidth=2 if not highlight else 3))
    ax.text(x, y+0.17, label_top, ha="center", va="bottom", fontsize=12)
    ax.text(x, y-0.03, label_bottom, ha="center", va="top", fontsize=11)

def arrow(ax, a, b, thick=False, dashed=False, yoffset=0.0):
    x1,y1 = POS[a]; x2,y2 = POS[b]
    y1 += yoffset; y2 += yoffset
    ax.annotate("", xy=(x2-0.65 if x2>x1 else x2+0.65, y2),
                xytext=(x1+0.65 if x2>x1 else x1-0.65, y1),
                arrowprops=dict(arrowstyle="->", lw=3 if thick else 1.8, linestyle="--" if dashed else "-"))

def path(ax, nodes, thick=True, yoffset=0.0):
    for i in range(len(nodes)-1):
        arrow(ax, nodes[i], nodes[i+1], thick=thick, dashed=False, yoffset=yoffset)

def draw_abstract(frame_idx, spotlight):
    F = FRAMES[frame_idx]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    # Processes
    draw_box(ax, POS["P1"], "P1", "client (Process)", highlight=(spotlight=="Process"))
    draw_box(ax, POS["P1p"], "P1'", "client (Process)", highlight=(spotlight=="Process"))
    draw_box(ax, POS["P2"], "P2", "server (Process)")
    draw_box(ax, POS["P3"], "P3", "server (Process)")
    draw_box(ax, POS["P4"], "P4", "server (Process)")

    # Proxies at clients
    ax.text(POS["P1"][0], POS["P1"][1]-0.28, "Proxy", ha="center", fontsize=12,
            bbox=dict(boxstyle="round", fill=False) if spotlight=="Proxy" else None)
    ax.text(POS["P1p"][0], POS["P1p"][1]-0.28, "Proxy", ha="center", fontsize=12,
            bbox=dict(boxstyle="round", fill=False) if spotlight=="Proxy" else None)

    # Skeletons on servers
    for srv in ["P2","P3","P4"]:
        ax.text(POS[srv][0], POS[srv][1]-0.28, "Skeleton", ha="center", fontsize=12,
                bbox=dict(boxstyle="round", fill=False) if spotlight=="Skeleton" else None)

    # Object location
    obj_at = F["obj"]
    ax.text(POS[obj_at][0], POS[obj_at][1]+0.24, "Object", ha="center", fontsize=12,
            bbox=dict(boxstyle="round", fill=False) if spotlight=="Object" else None)

    # IPC canvas (light background arrows)
    if spotlight in ("IPC","Invocation","Forwarding","Process","Proxy","Skeleton","Object",""):
        for src in ["P1","P1p"]:
            for dst in ["P2","P3","P4"]:
                arrow(ax, src, dst, thick=False, dashed=True, yoffset=0.18)  # faint network

    # Forwarding pointers (dashed, bold)
    for k,v in F["fwd"].items():
        arrow(ax, k, v, thick=True, dashed=True, yoffset=-0.06)
        cx = (POS[k][0]+POS[v][0])/2
        cy = (POS[k][1]+POS[v][1])/2 - 0.12
        ax.text(cx, cy, "forward pointer", ha="center", fontsize=10)

    # Invocation requests (thick solid)
    for inv in F["invocations"]:
        path(ax, inv, thick=True, yoffset=0.0)

    ax.set_xlim(-2.4, 1.8); ax.set_ylim(-1.6, 1.4); ax.axis("off")
    st.pyplot(fig)

def real_app_mapping(app_choice, frame_idx, spotlight):
    F = FRAMES[frame_idx]
    lines = []

    if app_choice == "Food Delivery":
        roles = {
            "P1":"Customer app (you)",
            "P1p":"Another customer app",
            "P2":"Restaurant Server A",
            "P3":"Restaurant Server B",
            "P4":"Restaurant Server C",
            "Object":"Order Service",
            "Proxy":"Mobile API stub",
            "Skeleton":"Server adapter",
        }
        scenario = [
            "• **Process**: app or server instance (Customer app, Restaurant servers).",
            "• **Proxy**: your app’s local stub that sends RPC to restaurant servers.",
            "• **Skeleton**: server adapter that turns RPC into real service calls.",
            "• **Object**: Order Service (where state lives).",
            "• **Interprocess Communication**: RPC/HTTP between app and servers.",
            "• **Forwarding pointer**: after service migrates, old server forwards to new.",
            "• **Invocation**: tap “Place Order” → proxy → skeleton → object.",
            "• **Identical proxy**: P1' is another phone with the same proxy behavior."
        ]
    else:
        roles = {
            "P1":"Your laptop (Drive client)",
            "P1p":"Another laptop (Drive client)",
            "P2":"Storage Node A",
            "P3":"Storage Node B",
            "P4":"Storage Node C",
            "Object":"File Service",
            "Proxy":"Drive client stub",
            "Skeleton":"Storage node stub",
        }
        scenario = [
            "• **Process**: client or storage node process.",
            "• **Proxy**: Drive client stub exposing file ops.",
            "• **Skeleton**: storage node stub receiving RPC.",
            "• **Object**: File Service (data + ops).",
            "• **Interprocess Communication**: gRPC/HTTPS.",
            "• **Forwarding pointer**: old node forwards to new after migration.",
            "• **Invocation**: Save file → proxy → skeleton → object.",
            "• **Identical proxy**: P1' is another client with same stub."
        ]

    # Frame-specific explanation
    step_explain = [
        "Direct call: clients contact the node with the object.",
        "Migration: object moves; old node leaves a forward pointer.",
        "First call after migration: forwarded through old node, then client learns new location.",
        "Shortcut: client now calls the new node directly (no extra hop).",
        "Identical proxy: a second client appears; its first call is forwarded the same way.",
        "Cleanup: both clients now call directly; forwarding removed."
    ][frame_idx]

    # Spotlight emphasis sentence
    emph = {
        "Process":"Each box is a separate process executing in parallel.",
        "Proxy":"Proxies live at clients and make remote calls look local.",
        "Skeleton":"Skeletons live on servers and dispatch to the object.",
        "Object":"Only one live object instance; it can migrate between servers.",
        "IPC":"IPC is the network path; proxies never share address space with skeletons.",
        "Forwarding":"Forward pointers prevent breakage while clients still hold old locations.",
        "Invocation":"An invocation is a method call traveling proxy → skeleton → object.",
        "Identical":"Another client’s proxy behaves identically even though it’s a different instance."
    }.get(spotlight, "")

    st.markdown(f"#### Real App: **{app_choice}**")
    st.caption("Roles mapping")
    st.write(f"- P1 = {roles['P1']}  \n- P1' = {roles['P1p']}  \n- P2 = {roles['P2']}  \n- P3 = {roles['P3']}  \n- P4 = {roles['P4']}")
    st.write(f"- Proxy = {roles['Proxy']}  \n- Skeleton = {roles['Skeleton']}  \n- Object = {roles['Object']}")
    st.markdown("---")
    st.markdown("**What’s happening in this step**")
    st.write(step_explain)
    st.markdown("**Concept checklist**")
    for line in scenario:
        st.write(line)
    if emph:
        st.info(emph)

# ==========================
# UI Layout
# ==========================
st.title("Forwarding Pointers with Proxies & Skeletons — Dual View (Abstract + Real App)")

with st.sidebar:
    st.markdown("### Controls")
    colA, colB = st.columns([1,1])
    if "frame" not in st.session_state:
        st.session_state.frame = 0
    if colA.button("⏮ Reset"):
        st.session_state.frame = 0
    if colB.button("⏭ Last"):
        st.session_state.frame = len(FRAMES)-1

    play = st.checkbox("▶ Auto-play", value=False)
    frame = st.slider("Timeline", 0, len(FRAMES)-1, value=st.session_state.frame,
                      help="0: direct call → 5: both clients direct, cleaned up")
    st.session_state.frame = frame

    st.markdown("---")
    app_choice = st.radio("Real App Mapping", ["Food Delivery", "Cloud Storage"], horizontal=False)
    spotlight = st.selectbox("Spotlight a concept", ["", "Process", "Proxy", "Skeleton", "Object", "IPC", "Invocation", "Forwarding", "Identical"])

    st.markdown("---")
    st.markdown("**Legend**")
    st.write("- Solid thick arrows = **Invocation request** path")
    st.write("- Dashed arrows = **Forwarding pointer** or background IPC")
    st.write("- Boxes = **Processes**; labels show **Proxy/Skeleton/Object**")

# Auto-play logic
if play:
    next_frame = (st.session_state.frame + 1) % len(FRAMES)
    st.session_state.frame = next_frame
    time.sleep(0.8)
    st.experimental_rerun()

left, right = st.columns([1.1, 0.9])
with left:
    st.subheader(f"Abstract View — Step {st.session_state.frame}: {FRAMES[st.session_state.frame]['name']}")
    draw_abstract(st.session_state.frame, spotlight)

with right:
    real_app_mapping(app_choice, st.session_state.frame, spotlight)

st.markdown("---")
st.markdown("**Teaching tip**: Ask students to predict the next invocation path when migration happens. Then toggle the spotlight to reinforce each concept explicitly.")
