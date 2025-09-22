
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import time

st.set_page_config(page_title="Forwarding Pointers — Classic Representation", layout="wide")

# ---------------- Positions that match the textbook-like figure -----------------
POS = {
    "P1": (-1.8, -0.6),
    "P2": (-1.8,  0.9),
    "P3": ( 0.0,  0.2),
    "P4": ( 1.8, -0.2),
}

def process_box(ax, pid, label_extra=""):
    x, y = POS[pid]
    w, h = 1.8, 1.4
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2))
    ax.text(x+0.1, y+h-0.15, f"Process {pid}", fontsize=13, ha="left", va="top")
    if label_extra:
        ax.text(x+0.12, y+0.55, label_extra, fontsize=12, ha="left")

    return (x, y, w, h)

def stub(ax, x, y, facing="right", label=None, label_offset=(0,0)):
    # Draws the proxy/skeleton pentagon used in the classic figure
    # facing 'right' or 'left'
    w, h = 0.28, 0.22
    if facing == "right":
        pts = [(x, y), (x+w*0.55, y), (x+w, y+h*0.5), (x+w*0.55, y+h), (x, y+h)]
    else:
        pts = [(x, y+h*0.5), (x+w*0.45, y+h), (x+w, y+h), (x+w, y), (x+w*0.45, y)]
    ax.add_patch(Polygon(pts, closed=True, fill=False, linewidth=2))
    if label:
        ax.text(x+label_offset[0], y+h+label_offset[1], label, fontsize=12, ha="left", va="bottom")

def arrow(ax, src, dst, thick=False, dashed=False):
    x1, y1 = src; x2, y2 = dst
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=3 if thick else 1.6,
                                linestyle="--" if dashed else "-"))

def line(ax, src, dst):
    ax.plot([src[0], dst[0]], [src[1], dst[1]], color="black", linewidth=1.2)

# ---------------- Scenes (match the 3 textbook slides) -----------------
SCENES = [
# I: Basic with two clients and one server skeleton forwarding to object
{
 "title": "Forwarding Pointers I — Basic RPC view",
 "explain": "Two client processes (P1, P2) hold proxies p and p′. Both refer to the same skeleton in P3. Interprocess communication connects P1/P2 to P3. Skeleton performs a local invocation to the object in P4.",
 "forward": {},
 "invocations": [("P1_proxy","P3_skel"), ("P2_proxy","P3_skel"), ("P3_local","P4_obj")],
 "labels": {
     "P1_proxy": ("P1", ( -1.55, -0.15), "right", "Proxy p", (0.0, -0.02)),
     "P2_proxy": ("P2", ( -1.55,  1.35), "right", "Proxy p′", (0.0, -0.02)),
     "P3_skel_left": ("P3", (  0.10,  0.45), "right", "", (0,0)),
     "P3_skel_right":("P3", (  0.58,  0.45), "left",  "Identical proxy", (0.05, 0.1)),
     "P3_local":     ("P3", (  0.52,  0.36), "right", "Local invocation\nIdentical skeleton", (0.2, -0.48)),
     "P4_obj":       ("P4", (  2.15,  0.32), "left",  "Object", (0.0, 0.0)),
 },
 "ipc": [("P1_proxy","P3_skel_left"), ("P2_proxy","P3_skel_left")],
},
# II: Migration — object moves, forwarding pointer left from old place to new
{
 "title": "Forwarding Pointers II — Object migration with forwarding pointer",
 "explain": "Object moves to P3. The skeleton at P3 now hosts the object; the previous path from proxies remains valid. (If object had moved elsewhere, a forward pointer would be left behind.)",
 "forward": {},
 "invocations": [("P1_proxy","P3_skel_left"), ("P2_proxy","P3_skel_left")],
 "labels": {
     "P1_proxy": ("P1", ( -1.55, -0.15), "right", "Proxy p", (0.0, -0.02)),
     "P2_proxy": ("P2", ( -1.55,  1.35), "right", "Proxy p′", (0.0, -0.02)),
     "P3_skel_left": ("P3", (  0.10,  0.45), "right", "Skeleton / Object", (0.0, 0.0)),
     "P3_skel_right":("P3", (  0.58,  0.45), "left",  "", (0,0)),
     "P4_obj":       ("P4", (  2.15,  0.32), "left",  "", (0.0, 0.0)),
 },
 "ipc": [("P1_proxy","P3_skel_left"), ("P2_proxy","P3_skel_left")],
},
# III: Shortcutting — proxy learns new location, sets direct path, intermediate pointers can be GC'd
{
 "title": "Forwarding Pointers III — Shortcutting after first forwarded call",
 "explain": "When the object migrates, the first call may be forwarded. After that, clients set a shortcut to the current location, eliminating extra hops. The intermediate skeleton not referenced by any proxy can be reclaimed.",
 "forward": {("P4_obj","P3_skel_left")},
 "invocations": [("P1_proxy","P3_skel_left"), ("P2_proxy","P3_skel_left")],
 "labels": {
     "P1_proxy": ("P1", ( -1.55, -0.15), "right", "Proxy p", (0.0, -0.02)),
     "P2_proxy": ("P2", ( -1.55,  1.35), "right", "Proxy p′", (0.0, -0.02)),
     "P3_skel_left": ("P3", (  0.10,  0.45), "right", "Skeleton", (0.0, 0.0)),
     "P3_skel_right":("P3", (  0.58,  0.45), "left",  "Identical proxy", (0.05, 0.1)),
     "P4_obj":       ("P4", (  2.15,  0.32), "left",  "Object", (0.0, 0.0)),
 },
 "ipc": [("P1_proxy","P3_skel_left"), ("P2_proxy","P3_skel_left")],
},
]

if "scene" not in st.session_state:
    st.session_state.scene = 0

# ---------------- drawing ----------------
def draw_scene(scene_idx):
    S = SCENES[scene_idx]
    fig, ax = plt.subplots(figsize=(10, 6))

    # Process boxes
    process_box(ax, "P1"); process_box(ax, "P2"); process_box(ax, "P3"); process_box(ax, "P4")

    # Stubs and labels
    for key, (pid, (sx, sy), facing, label, lo) in S["labels"].items():
        # Convert local coords relative to process box origin
        px, py = POS[pid]
        stub(ax, px+sx, py+sy, facing=facing, label=label, label_offset=lo)

    # Interprocess communication lines (thin)
    for a_key, b_key in S.get("ipc", []):
        ax.plot([], [])  # placeholder to avoid issues
        # compute points roughly from stub centers
        a_pid, (ax_rel, ay_rel), _, _, _ = S["labels"][a_key]
        b_pid, (bx_rel, by_rel), _, _, _ = S["labels"][b_key]
        x1 = POS[a_pid][0] + ax_rel + 0.28
        y1 = POS[a_pid][1] + ay_rel + 0.11
        x2 = POS[b_pid][0] + bx_rel + 0.02
        y2 = POS[b_pid][1] + by_rel + 0.11
        line(ax, (x1, y1), (x2, y2))

    # Invocation arrows (bold)
    for inv in S["invocations"]:
        points = []
        for k in inv:
            pid, (rx, ry), facing, _, _ = S["labels"][k]
            cx = POS[pid][0] + rx + (0.28 if facing=="right" else 0.02)
            cy = POS[pid][1] + ry + 0.11
            points.append((cx, cy))
        for i in range(len(points)-1):
            arrow(ax, points[i], points[i+1], thick=True, dashed=False)

    # Forwarding pointer arrows (dashed)
    for a_key, b_key in S.get("forward", set()):
        a_pid, (ax_rel, ay_rel), f1, _, _ = S["labels"][a_key]
        b_pid, (bx_rel, by_rel), f2, _, _ = S["labels"][b_key]
        x1 = POS[a_pid][0] + ax_rel + (0.28 if f1=="right" else 0.02)
        y1 = POS[a_pid][1] + ay_rel + 0.18
        x2 = POS[b_pid][0] + bx_rel + (0.02 if f2=="left" else 0.28)
        y2 = POS[b_pid][1] + by_rel + 0.18
        arrow(ax, (x1,y1), (x2,y2), thick=True, dashed=True)

    ax.set_xlim(-3.0, 3.6)
    ax.set_ylim(-1.2, 2.2)
    ax.axis("off")
    st.pyplot(fig)
    st.markdown(f"**{S['title']}**")
    st.write(S["explain"])

# ---------------- UI ----------------
st.title("Classic Representation — Proxies, Skeletons & Forwarding Pointers")

cols = st.columns([1,1,1,3])
if cols[0].button("⏮ Reset"):
    st.session_state.scene = 0
if cols[1].button("◀ Prev", disabled=st.session_state.scene==0):
    st.session_state.scene = max(0, st.session_state.scene-1)
if cols[2].button("Next ▶", disabled=st.session_state.scene==len(SCENES)-1):
    st.session_state.scene = min(len(SCENES)-1, st.session_state.scene+1)

autoplay = cols[3].checkbox("▶ Auto-advance")

draw_scene(st.session_state.scene)

if autoplay:
    time.sleep(0.9)
    st.session_state.scene = (st.session_state.scene + 1) % len(SCENES)
    st.experimental_rerun()

st.markdown("---")
st.markdown("**Legend**: small pentagon = **proxy/skeleton icon**, thin lines = **interprocess communication**, bold arrows = **invocation request**, dashed bold = **forwarding pointer**.")
