
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="DS Tutor: Forwarding Pointers (Clean Edition)", layout="wide")

# ---------------------------
# Scenario & Simple State
# ---------------------------
# We use a linear 4-step story. No networkx; we draw a clear schematic with big labels.
SCENES = [
    {
        "name": "Step 1 — Direct call (no migration)",
        "desc": "Client P1 calls Proxy → Skeleton at P4 → Object at P4 (one hop).",
        "obj_at": "P4",
        "proxy_points": "P4",
        "forward": {},  # no forwarding pointer
        "path": ["P1", "P4"]
    },
    {
        "name": "Step 2 — Migration happens",
        "desc": "The object moves from P4 to P2. A forwarding pointer is left at P4 pointing to P2. The proxy still thinks P4 is correct.",
        "obj_at": "P2",
        "proxy_points": "P4",
        "forward": {"P4": "P2"},
        "path": ["P1", "P4", "P2"]
    },
    {
        "name": "Step 3 — First call after migration",
        "desc": "Invocation takes two hops: P1 → P4 (forwarding) → P2. The proxy learns the new location.",
        "obj_at": "P2",
        "proxy_points": "P4",
        "forward": {"P4": "P2"},
        "path": ["P1", "P4", "P2"]
    },
    {
        "name": "Step 4 — Shortcut installed",
        "desc": "Proxy updates to call P2 directly. Forwarding chain is eliminated.",
        "obj_at": "P2",
        "proxy_points": "P2",
        "forward": {},  # cleaned up
        "path": ["P1", "P2"]
    },
]

if "scene_idx" not in st.session_state:
    st.session_state.scene_idx = 0

def pos(label):
    # fixed coordinates for clarity
    mapping = {
        "P1": (-1.4, 0.0),
        "P2": (0.8, 0.8),
        "P3": (0.8, 0.0),
        "P4": (0.8, -0.8),
    }
    return mapping[label]

def draw(scene):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw nodes
    for node in ["P1", "P2", "P3", "P4"]:
        x, y = pos(node)
        ax.add_patch(plt.Rectangle((x-0.35, y-0.18), 0.7, 0.36, fill=False, linewidth=2))
        role = "client" if node == "P1" else "server"
        ax.text(x, y+0.16, f"{node}\n({role})", ha="center", va="bottom", fontsize=12)

    # Proxy at P1
    x1, y1 = pos("P1")
    ax.text(x1, y1-0.05, "Proxy", ha="center", va="top", fontsize=12)

    # Skeleton labels on servers
    for srv in ["P2","P3","P4"]:
        xs, ys = pos(srv)
        ax.text(xs, ys-0.05, "Skeleton", ha="center", va="top", fontsize=12)

    # Object location
    xo, yo = pos(scene["obj_at"])
    ax.text(xo, yo+0.02, "Object", ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round", fill=False))

    # Draw invocation path
    p = scene["path"]
    for i in range(len(p)-1):
        xA, yA = pos(p[i])
        xB, yB = pos(p[i+1])
        ax.annotate("",
            xy=(xB-0.38 if xB>xA else xB+0.38, yB),
            xytext=(xA+0.38 if xB>xA else xA-0.38, yA),
            arrowprops=dict(arrowstyle="->", lw=3)
        )

    # Draw forwarding pointers (dashed)
    for k, v in scene["forward"].items():
        xA, yA = pos(k); xB, yB = pos(v)
        ax.annotate("",
            xy=(xB-0.38 if xB>xA else xB+0.38, yB+0.1),
            xytext=(xA+0.38 if xB>xA else xA-0.38, yA+0.1),
            arrowprops=dict(arrowstyle="->", lw=2, linestyle="dashed")
        )
        ax.text( (xA+xB)/2, (yA+yB)/2 + 0.12, "forward", ha="center", fontsize=10)

    ax.set_xlim(-2.0, 1.8)
    ax.set_ylim(-1.3, 1.2)
    ax.axis("off")
    st.pyplot(fig)

# ---------------------------
# Layout
# ---------------------------
left, right = st.columns([1,1])

with left:
    scene = SCENES[st.session_state.scene_idx]
    st.markdown(f"### {scene['name']}")
    st.write(scene["desc"])
    draw(scene)

with right:
    st.markdown("### Controls")
    st.write("Use **Next** to step through the story. This keeps attention on one idea at a time.")
    c1, c2, c3 = st.columns(3)
    if c1.button("⏮ Reset"):
        st.session_state.scene_idx = 0
    if c2.button("◀ Prev", disabled=st.session_state.scene_idx==0):
        st.session_state.scene_idx = max(0, st.session_state.scene_idx-1)
    if c3.button("Next ▶", disabled=st.session_state.scene_idx==len(SCENES)-1):
        st.session_state.scene_idx = min(len(SCENES)-1, st.session_state.scene_idx+1)

    st.markdown("---")
    st.markdown("### Key Terms")
    st.write("- **Process**: running program with its own address space (P1–P4).")
    st.write("- **Proxy**: client-side stub at P1.")
    st.write("- **Skeleton**: server-side stub at P2–P4.")
    st.write("- **Object**: the actual service code & state (migrates).")
    st.write("- **Interprocess Communication (IPC)**: network arrows between processes.")
    st.write("- **Invocation request**: thick arrows showing the call path.")
    st.write("- **Forwarding pointer**: dashed arrow left after migration.")
    st.write("- **Identical proxy**: add another client P1' in class discussion; same behavior.")

st.markdown("---")
st.markdown("**Teaching tip**: Pause at Step 2 and ask students to predict the next call path before clicking *Next*.")
