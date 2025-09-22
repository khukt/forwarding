
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
    process_box(ax, "P4", extra_label="Object")

    # Stubs
    p1_anchor = stub(ax, "P1", side="right", name="")  # we'll place label outside
    p3_anchor = stub(ax, "P3", side="left", name="")   # external label "Skeleton"
    # P2 proxy p′ inside P2 at right edge
    p2_anchor = stub(ax, "P2", side="right", name="")

    # External labels (clean, outside boxes)
    ax.text(POS["P1"][0]+0.2, POS["P1"][1]-0.25, S["notes"].get("P1_proxy","Proxy p"), fontsize=12, ha="left", va="top")
    ax.text(POS["P2"][0]+0.2, POS["P2"][1]+BOX_H+0.15, S["notes"].get("P2_proxy","Proxy p′"), fontsize=12, ha="left", va="bottom")
    ax.text(POS["P3"][0]-0.25, POS["P3"][1]+BOX_H*0.5+0.22, S["notes"].get("P3_skel","Skeleton"), fontsize=12, ha="right", va="center")
    ax.text(POS["P4"][0]+BOX_W+0.2, POS["P4"][1]+BOX_H*0.5, S["notes"].get("P4_obj","Object"), fontsize=12, ha="left", va="center")

    # Background IPC (thin) P1->P3 and P2->P3
    arrow(ax, p1_anchor, p3_anchor, style="ipc")
    # slight upward offset for P2 to avoid overlap: move start y by +0.12
    p2_offset = (p2_anchor[0], p2_anchor[1]+0.12)
    arrow(ax, p2_offset, p3_anchor, style="ipc")

    # Anchors map
    anchors = {
        "P1_proxy": p1_anchor,
        "P2_proxy": p2_anchor,
        "P3_skel":  p3_anchor,
        "P4_obj":   (POS["P4"][0], POS["P4"][1] + BOX_H*0.5),
    }

    # Active path
    for (a,b) in S["active"]:
        # apply same y-offset for P2 active path to keep separation
        a_pt = anchors[a]
        if a == "P2_proxy":
            a_pt = (a_pt[0], a_pt[1]+0.12)
        arrow(ax, a_pt, anchors[b], style="active")

    # Forwarding pointer
    for (a,b) in S["forward"]:
        arrow(ax, anchors[a], anchors[b], style="forward")

    ax.set_xlim(-3.6, 4.0)
    ax.set_ylim(-1.8, 2.6)
    ax.axis("off")
    st.pyplot(fig)
    st.markdown(f"**{S['title']}**")
    st.write(S["explain"])


st.caption("Legend: thin arrows = background IPC, thick solid = active invocation, thick dashed = forwarding pointer.")
