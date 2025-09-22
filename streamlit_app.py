# kid_bank_app.py
# "Bank City" — a kid-friendly animation of Proxy • Skeleton • Forwarding • Shortcut
# Uses big emoji houses and a moving courier.
# Run:
#   pip install streamlit plotly
#   streamlit run kid_bank_app.py

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="🏙️ Bank City (Kid Mode) — Proxy • Skeleton • Forwarding", layout="wide")

# -------------------------------
# Simple, fixed map coordinates
# -------------------------------
POS = {
    "ATM": (-1.2,  0.6),   # 🏧 ATM House
    "APP": (-1.2, -0.6),   # 📱 Mobile app house (for later ideas)
    "HELPER": (0.2,  0.0), # 🧑‍💼 Helper Station (Skeleton)
    "BANK_R1": (1.6,  0.0),# 🏦 Bank House (location 1)
    "BANK_R2": (-0.2, -0.6)# 🏦 Bank moves here (near APP) for migration demo
}

COLOR = {
    "road_proxy":  "#1f77b4",  # blue
    "road_forward":"#ff7f0e",  # orange (forwarding)
    "road_serve":  "#7f7f7f",  # gray (helper→bank)
    "highlight":   "#d62728",  # red (animation token + highlight)
}

EMOJI = {
    "ATM": "🏧 ATM House",
    "HELPER": "🧑‍💼 Helper Station (Skeleton)",
    "BANK": "🏦 Bank House (Object)"
}

def lerp(a, b, t):
    return (a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t)

def node(label, x, y, selected=False):
    return go.Scatter(
        x=[x], y=[y],
        mode="text+markers",
        text=[label],
        textposition="top center",
        marker=dict(
            size=50 if selected else 42,
            color="#fff6d5" if "🏦" in label else "#e8f4fa",
            line=dict(color="#1d3557", width=2)
        ),
        hoverinfo="skip",
        showlegend=False
    )

def road(x1,y1,x2,y2,color,width=4, dash=None, hover=None):
    return go.Scatter(
        x=[x1,x2], y=[y1,y2],
        mode="lines",
        line=dict(color=color, width=width, dash=dash if dash else "solid"),
        hoverinfo="text", text=hover or "",
        showlegend=False
    )

def arrow(x1,y1,x2,y2, text, color):
    ax = (x1-x2)*60
    ay = (y1-y2)*60
    return dict(
        x=x2, y=y2, ax=x2+ax, ay=y2+ay,
        xref="x", yref="y", axref="x", ayref="y",
        text=text, showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor=color,
        font=dict(color=color, size=16), bgcolor="rgba(255,255,255,0.7)"
    )

def build_scene(step, story, frames_per_segment=18, speed="normal"):
    """
    Returns a Plotly Figure with animation frames for the given step.
    Steps:
      1) ATM → Helper (proxy road)
      2) Helper → Bank at R1 (serve road)
      3) Bank MOVES to R2; ATM → Helper (proxy), Helper → Bank at R2 (forward road)
      4) Shortcut: ATM → Bank at R2 directly (proxy road)
    """
    # Speed to milliseconds (lower = faster)
    frame_ms = {"slow": 140, "normal": 90, "fast": 45}[speed]

    # Choose where the bank currently lives for this step
    bank_here = "BANK_R1" if step in (1,2) else "BANK_R2"

    # Base map (nodes)
    nodes = [
        node("🏧 ATM", *POS["ATM"], selected=(step==1)),
        node("🧑‍💼 Helper", *POS["HELPER"], selected=(step==2 or step==3)),
        node("🏦 Bank", *POS[bank_here], selected=(step in (2,3,4)))
    ]

    # Roads always visible (thin), so kids see the "city plan"
    base_roads = []
    # Proxy road ATM→HELPER
    base_roads.append(road(*POS["ATM"], *POS["HELPER"], COLOR["road_proxy"], 3,
                           hover="Proxy road: ATM sends letter to Helper"))
    # Serve road HELPER→BANK (both bank places)
    base_roads.append(road(*POS["HELPER"], *POS["BANK_R1"], COLOR["road_serve"], 2, dash="dot",
                           hover="Helper delivers to Bank"))
    base_roads.append(road(*POS["HELPER"], *POS["BANK_R2"], COLOR["road_serve"], 2, dash="dot",
                           hover="Helper delivers to Bank (if moved)"))
    # Forwarding road HELPER→BANK_R2 (same line, but orange)
    base_roads.append(road(*POS["HELPER"], *POS["BANK_R2"], COLOR["road_forward"], 3, dash="dash",
                           hover="Forwarding road: old place forwards to new place"))

    # Which segments animate in each step
    if step == 1:
        segments = [("ATM","HELPER","① Proxy road")]
    elif step == 2:
        segments = [("HELPER","BANK_R1","② Helper → Bank")]
    elif step == 3:
        segments = [("ATM","HELPER","① Proxy road"),
                    ("HELPER","BANK_R2","② Forwarding road (Bank moved!)")]
    else:  # step == 4
        segments = [("ATM","BANK_R2","① Shortcut (learned path)")]

    # Token label uses the story choice
    story_word = {"Login":"Login letter","Balance":"Balance letter","Deposit":"Deposit letter","Withdraw":"Withdraw letter"}[story]

    # Build the base figure
    fig = go.Figure(data=[*base_roads, *nodes])
    fig.update_layout(
        margin=dict(l=10,r=10,t=50,b=10),
        xaxis=dict(visible=False, range=[-1.9, 1.9]),
        yaxis=dict(visible=False, range=[-1.6, 1.2]),
        dragmode="pan",
        title=f"Step {step}: " + {
            1: "ATM sends a letter via the Proxy road",
            2: "Helper brings the letter to the Bank",
            3: "Bank MOVES! Helper forwards the letter to the new place",
            4: "Shortcut learned: ATM goes straight to the Bank",
        }[step],
    )

    # Legend (simple)
    legend = [
        arrow(-1.85,-1.45,-1.65,-1.45,"Proxy road", COLOR["road_proxy"]),
        arrow(-1.85,-1.25,-1.65,-1.25,"Forward road", COLOR["road_forward"]),
        arrow(-1.85,-1.05,-1.65,-1.05,"Serve road", COLOR["road_serve"]),
    ]
    fig.update_layout(annotations=legend)

    # Build frames: token moves along each segment
    frames = []
    circ = ["①","②","③","④"]
    for idx, (src, dst, label) in enumerate(segments):
        x1,y1 = POS[src]; x2,y2 = POS[dst]
        # Overlay a thick highlight for the active road segment
        highlight = road(x1,y1,x2,y2, COLOR["highlight"], 7)
        for k in range(frames_per_segment):
            t = (k+1) / frames_per_segment
            xt, yt = lerp((x1,y1),(x2,y2), t)
            token = go.Scatter(
                x=[xt], y=[yt],
                mode="markers+text",
                marker=dict(size=22, color=COLOR["highlight"], symbol="diamond"),
                text=[f"{circ[idx]} {story_word}"], textposition="top center",
                textfont=dict(size=16, color=COLOR["highlight"]),
                hoverinfo="skip",
                showlegend=False
            )
            ann = [arrow(x1,y1,x2,y2, label, COLOR["highlight"])]
            frames.append(go.Frame(
                name=f"s{step}_{idx}_{k}",
                data=[*base_roads, *nodes, highlight, token],
                layout=go.Layout(annotations=legend+ann)
            ))

    fig.update(frames=frames)

    # Animation controls
    fig.update_layout(
        updatemenus=[{
            "type":"buttons","direction":"left","x":0.0,"y":1.15,
            "pad":{"r":8,"t":8},
            "buttons":[
                {"label":"▶ Play","method":"animate","args":[None,{"frame":{"duration":frame_ms,"redraw":True},"fromcurrent":True}]},
                {"label":"⏸ Pause","method":"animate","args":[[None],{"frame":{"duration":0},"mode":"immediate"}]},
            ],
        }],
        sliders=[{
            "active":0,"y":1.09,"pad":{"t":28},
            "steps":[{
                "label":f"{i+1}",
                "method":"animate",
                "args":[[fr.name],{"frame":{"duration":0,"redraw":True},"mode":"immediate"}]
            } for i, fr in enumerate(frames)]
        }]
    )
    return fig

# -------------------------------
# Sidebar (super simple)
# -------------------------------
st.sidebar.header("Pick a Story")
story = st.sidebar.selectbox("Letter type", ["Login","Balance","Deposit","Withdraw"], index=0, key="kid_story")

st.sidebar.header("Step")
step = st.sidebar.select_slider("Scene", options=[1,2,3,4], value=1, key="kid_step")

st.sidebar.header("Speed")
speed = st.sidebar.select_slider("Animation speed", options=["slow","normal","fast"], value="normal", key="kid_speed")

st.sidebar.markdown("---")
st.sidebar.markdown("**What’s happening?**")
st.sidebar.markdown(
"""
- **Step 1**: 🏧 ATM uses the **Proxy road** to reach the 🧑‍💼 Helper.
- **Step 2**: The Helper follows the **Serve road** to the 🏦 Bank.
- **Step 3**: The 🏦 Bank **moves**! The Helper follows the **Forward road** to the new place.
- **Step 4**: The 🏧 ATM **learns a shortcut** to the new Bank location.
"""
)

# -------------------------------
# Draw the current step (kids can press ▶ Play on the graph)
# -------------------------------
fig = build_scene(step, story, frames_per_segment=18, speed=speed)
st.plotly_chart(fig, width='stretch')
