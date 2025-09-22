
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import time

st.set_page_config(page_title="Classic Representation — Proxies, Skeletons & Forwarding Pointers", layout="wide")

# ---------------- Layout matching the textbook style ----------------
POS = {
    "P1": (-2.2, -0.8),
    "P2": (-2.2,  0.8),
    "P3": ( 0.0,  0.0),
    "P4": ( 2.2,  0.0),
}

BOX_W, BOX_H = 2.0, 1.6

def process_box(ax, pid, extra_label=None):
    x, y = POS[pid]
    ax.add_patch(Rectangle((x, y), BOX_W, BOX_H, fill=False, linewidth=2))
    ax.text(x+0.08, y+BOX_H-0.12, f"Process {pid}", fontsize=13, ha="left", va="top")
    if extra_label:
        ax.text(x+BOX_W+0.2, y+BOX_H*0.5, extra_label, fontsize=12, ha="left", va="center")

def stub(ax, pid, side="right", name=""):
    # Draw the classic pentagon stub flush with the box edge
    x, y = POS[pid]
    w, h = 0.32, 0.24
    if side == "right":
        sx, sy = x + BOX_W - 0.46, y + BOX_H*0.5 - h/2
        pts = [(sx,sy),(sx+w*0.55,sy),(sx+w,sy+h*0.5),(sx+w*0.55,sy+h),(sx,sy+h)]
        text_xy = (x+BOX_W+0.12, y+BOX_H*0.5)
        ha = "left"
    else:  # left side (for a symmetric look if needed)
        sx, sy = x + 0.14, y + BOX_H*0.5 - h/2
        pts = [(sx+w,sy),(sx+w*0.45,sy),(sx,sy+h*0.5),(sx+w*0.45,sy+h),(sx+w,sy+h)]
        text_xy = (x-0.15, y+BOX_H*0.5)
        ha = "right"
    ax.add_patch(Polygon(pts, closed=True, fill=False, linewidth=2))
    if name:
        ax.text(*text_xy, name, fontsize=12, ha=ha, va="center")
    # Return anchor where arrows should connect
    anchor = (sx+w if side=="right" else sx, y + BOX_H*0.5)
    return anchor

def arrow(ax, src, dst, style="active"):
    lw = 3 if style=="active" else 1.2
    ls = "-" if style!="forward" else "--"
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle="->", lw=lw, linestyle=ls))

# ---------------- Scenes: single clean path per step ----------------
# Keys in `points`: P1_proxy, P2_proxy, P3_skel, P4_obj
SCENES = [
    {
        "title": "I. Basic RPC view",
        "explain": "One active invocation highlighted: Proxy p in P1 calls the Skeleton in P3, which in turn performs a local invocation to the Object in P4.",
        "active": [("P1_proxy","P3_skel"), ("P3_skel","P4_obj")],
        "forward": [],
        "notes": {"P1_proxy":"Proxy p", "P2_proxy":"Proxy p′", "P3_skel":"Skeleton", "P4_obj":"Object"},
        "show_p2": True
    },
    {
        "title": "II. Migration with forwarding pointer",
        "explain": "The object migrates to P3. First call still goes to P4; a forwarding pointer sends it to P3. (Here we highlight the forwarding pointer only.)",
        "active": [("P1_proxy","P4_obj")],
        "forward": [("P4_obj","P3_skel")],
        "notes": {"P1_proxy":"Proxy p", "P2_proxy":"Proxy p′", "P3_skel":"Skeleton / Object", "P4_obj":"(old)"},
        "show_p2": True
    },
    {
        "title": "III. Shortcutting (after first forwarded call)",
        "explain": "After learning the new location, Proxy p calls P3 directly. The forwarding chain can be removed.",
        "active": [("P1_proxy","P3_skel")],
        "forward": [],
        "notes": {"P1_proxy":"Proxy p", "P2_proxy":"Proxy p′", "P3_skel":"Skeleton / Object", "P4_obj":""},
        "show_p2": True
    },
]

if "scene" not in st.session_state:
    st.session_state.scene = 0

# ---------------- Draw ----------------
def draw_scene(idx):
    S = SCENES[idx]
    fig, ax = plt.subplots(figsize=(11, 6))

    # Process boxes
    process_box(ax, "P2")
    process_box(ax, "P1")
    process_box(ax, "P3")
    process_box(ax, "P4", extra_label="Object" if idx==0 else "Object" if idx==2 else "")

    # Stubs
    p1_anchor = stub(ax, "P1", side="right", name=S["notes"].get("P1_proxy",""))
    if S.get("show_p2", False):
        stub(ax, "P2", side="right", name=S["notes"].get("P2_proxy",""))
    p3_anchor = stub(ax, "P3", side="left", name=S["notes"].get("P3_skel",""))
    # object is drawn as a stub on the left side of P4 for arrow consistency
    x4, y4 = POS["P4"]
    obj_anchor = (x4, y4 + BOX_H*0.5)  # left edge center
    ax.text(x4 + BOX_W + 0.2, y4 + BOX_H*0.5, S["notes"].get("P4_obj",""), fontsize=12, ha="left", va="center")

    # Background IPC (thin lines) from P1/P2 to P3
    arrow(ax, p1_anchor, p3_anchor, style="ipc")
    if S.get("show_p2", False):
        p2_anchor = (POS["P2"][0] + BOX_W, POS["P2"][1] + BOX_H*0.5)
        arrow(ax, p2_anchor, p3_anchor, style="ipc")

    # Active path for this step
    # Map logical keys to actual anchor coordinates
    anchors = {
        "P1_proxy": p1_anchor,
        "P2_proxy": (POS["P2"][0] + BOX_W, POS["P2"][1] + BOX_H*0.5),
        "P3_skel":  p3_anchor,
        "P4_obj":   obj_anchor,
    }

    for (a,b) in S["active"]:
        arrow(ax, anchors[a], anchors[b], style="active")

    # Forwarding pointer (dashed bold)
    for (a,b) in S["forward"]:
        arrow(ax, anchors[a], anchors[b], style="forward")

    ax.set_xlim(-3.6, 4.0)
    ax.set_ylim(-1.6, 2.4)
    ax.axis("off")
    st.pyplot(fig)
    st.markdown(f"**{S['title']}**")
    st.write(S["explain"])

# ---------------- UI ----------------
st.title("Classic Representation — Proxies, Skeletons & Forwarding Pointers")

c1, c2, c3, c4 = st.columns([1,1,1,3])
if c1.button("⏮ Reset"):
    st.session_state.scene = 0
if c2.button("◀ Prev", disabled=st.session_state.scene==0):
    st.session_state.scene = max(0, st.session_state.scene-1)
if c3.button("Next ▶", disabled=st.session_state.scene==len(SCENES)-1):
    st.session_state.scene = min(len(SCENES)-1, st.session_state.scene+1)
auto = c4.checkbox("▶ Auto-advance")

draw_scene(st.session_state.scene)

if auto:
    time.sleep(1.0)
    st.session_state.scene = (st.session_state.scene + 1) % len(SCENES)
    st.experimental_rerun()

st.caption("Legend: thin arrows = background IPC, thick solid = active invocation, thick dashed = forwarding pointer.")
