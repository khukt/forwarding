# streamlit_app.py
# ENSURE‑6G • Raw vs Semantic vs Hybrid — Multi‑Train Rail Demo (Sundsvall↔Stockholm)
# One‑file Streamlit app designed for Streamlit Cloud.
# Dependencies: streamlit, numpy, pandas, pydeck, altair
# Optional (auto-refresh): streamlit-autorefresh

import math
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

# ========= Page / session =========
st.set_page_config(page_title="ENSURE‑6G • Rail Semantic Safety Demo", layout="wide")
st.title("ENSURE‑6G: Raw vs Semantic Communication — Sundsvall ↔ Stockholm")
st.caption("Multi‑train safety demo • Lane A (safety alerts) vs Lane B (ops) • OSM map • Bandwidth & latency • TEN‑T corridor context")

# Session state
if "playing" not in st.session_state:
    st.session_state.playing = False
if "tick" not in st.session_state:
    st.session_state.tick = 0
if "alerts" not in st.session_state:
    st.session_state.alerts = []  # store delivered alert logs
if "laneA_bytes" not in st.session_state:
    st.session_state.laneA_bytes = 0
if "laneB_bytes" not in st.session_state:
    st.session_state.laneB_bytes = 0
if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None
if "seed" not in st.session_state:
    st.session_state.seed = 42

rng = np.random.default_rng(st.session_state.seed)

# ========= Sidebar (Scenario Controls) =========
with st.sidebar:
    st.header("Scenario")
    duration_min = st.slider("Journey duration (min, simulated)", 10, 60, 20, 5)
    two_trains = st.checkbox("Two trains (both directions)", value=True)
    env_noise = st.slider("Environmental variability", 0.0, 1.0, 0.35, 0.05)
    st.divider()
    st.header("Simulation")
    colpb1, colpb2, colpb3 = st.columns(3)
    with colpb1:
        if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause", use_container_width=True):
            st.session_state.playing = not st.session_state.playing
    with colpb2:
        if st.button("⏭ Step", use_container_width=True):
            st.session_state.tick += 1
    with colpb3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.tick = 0
            st.session_state.alerts = []
            st.session_state.laneA_bytes = 0
            st.session_state.laneB_bytes = 0
            st.session_state.last_latency_ms = None
    st.caption("Tip: Use Play for auto‑advance (refresh), Step for manual advance.")

# Auto‑advance via rerun (simple clock)
if st.session_state.playing:
    # Increase tick modestly; Streamlit Cloud refresh rate ≈ once per run
    st.session_state.tick += 1

# ========= Route & segments =========
# Approximate Sundsvall→Stockholm line with waypoints (lat, lon, env)
# env ∈ {"RMa" (rural macro), "UMa" (urban macro), "UMi" (urban micro)}
WAYPOINTS = [
    (62.3913, 17.3069, "UMa", "Sundsvall"),
    (61.7284, 17.1036, "RMa", "Hudiksvall"),
    (61.3037, 17.0582, "RMa", "Söderhamn"),
    (60.6749, 17.1413, "UMa", "Gävle"),  # Segment 14 vicinity
    (60.3402, 17.5193, "RMa", "Tierp"),
    (59.8586, 17.6389, "UMa", "Uppsala"),
    (59.3293, 18.0686, "UMi", "Stockholm")
]

# Build a densified polyline for animation
N_SAMPLES = 400
lats = np.interp(np.linspace(0, len(WAYPOINTS)-1, N_SAMPLES), np.arange(len(WAYPOINTS)), [w[0] for w in WAYPOINTS])
lons = np.interp(np.linspace(0, len(WAYPOINTS)-1, N_SAMPLES), np.arange(len(WAYPOINTS)), [w[1] for w in WAYPOINTS])
# Map each sample to environment by nearest waypoint segment
envs = []
labels = []
for i in range(N_SAMPLES):
    seg = min(int(np.floor(i / (N_SAMPLES/(len(WAYPOINTS)-1)))), len(WAYPOINTS)-2)
    envs.append(WAYPOINTS[seg][2])
    labels.append(WAYPOINTS[seg][3])

route_df = pd.DataFrame({"lat": lats, "lon": lons, "env": envs, "label": labels})

# Define risky segment indexes near Gävle (heuristic)
risk_start = int(N_SAMPLES * 3/6) - 10
risk_end = int(N_SAMPLES * 3/6) + 10

# ========= Train kinematics =========
# We map ticks to route index linearly across duration
TOTAL_TICKS = duration_min * 60  # one tick ≈ one second of simulated time

idx_A = min(st.session_state.tick, TOTAL_TICKS)
pos_A = int(np.interp(idx_A, [0, TOTAL_TICKS], [0, N_SAMPLES-1]))

if two_trains:
    idx_B = min(st.session_state.tick, TOTAL_TICKS)
    pos_B = int(np.interp(idx_B, [0, TOTAL_TICKS], [N_SAMPLES-1, 0]))
else:
    idx_B = None
    pos_B = None

# ========= Environment-driven raw signals =========
# Simulate diurnal rail temperature and correlated strain; solar and wind context
# Deterministic base + noise
phase = (st.session_state.tick % (24*60*60)) / (24*60*60)
rail_temp = 24 + 14 * math.sin(2*math.pi*(phase - 0.2))  # peak afternoon ~38°C
rail_temp += rng.normal(0, 0.8 + 1.5*env_noise)

strain = 8 + 0.25*(rail_temp - 24) + rng.normal(0, 0.6)
solar = max(0.0, 1000 * math.sin(math.pi * max(0, math.sin(2*math.pi*(phase - 0.25)))))  # rough daylight curve
wind = max(0.0, rng.normal(3.5, 1.2))

# Track temperature gradient (°C / 10 min)
if "prev_temp" not in st.session_state:
    st.session_state.prev_temp = rail_temp
temp_grad = (rail_temp - st.session_state.prev_temp) * (600.0)  # per 10 min if tick≈1s
st.session_state.prev_temp = rail_temp

# ========= Semantic encoder (Lane A / Lane B) =========
# Risk rule: active near risky segment and if rail_temp>38 and strain>10 & gradient>6°C/10min
risk_active = (risk_start <= pos_A <= risk_end) or (two_trains and pos_B is not None and risk_start <= pos_B <= risk_end)

risk_flag = False
recommendation = None
confidence = 0.0

if risk_active:
    conditions = [rail_temp > 38.0, strain > 10.0, temp_grad > 6.0]
    satisfied = sum(1 for c in conditions if c)
    if satisfied >= 2:
        risk_flag = True
        confidence = 0.6 + 0.2 * satisfied  # 0.8, 1.0, 1.2 capped later
        confidence = min(confidence, 0.98)
        recommendation = {"ops": "Impose 60 km/h TSR", "safety": "Driver caution"}

# Lane B: maintenance semantic (battery/health surrogate)
# Simple: every 5 minutes, emit a maintenance summary
emit_laneB = (st.session_state.tick % (5*60) == 1)

# ========= Channel model (simplified, rail-aware) =========
# Map env → pathloss & fading parameters (abstracted to SNR means)
ENV_SNR = {"RMa": 18, "UMa": 14, "UMi": 10}  # mean SNR dB
ENV_FADING = {"RMa": ("Rician", 8), "UMa": ("Mixed", 4), "UMi": ("Rayleigh", 0)}

def draw_snr(env: str) -> float:
    mean = ENV_SNR.get(env, 12)
    fading, k = ENV_FADING.get(env, ("Rayleigh", 0))
    # Convert mean dB → linear mean power scale fudge
    base = rng.normal(mean, 2.5)
    # Add small-scale fading delta
    if fading == "Rician":
        delta = rng.normal(0.5, 0.8)
    elif fading == "Rayleigh":
        delta = rng.normal(-1.0, 1.2)
    else:  # Mixed
        delta = rng.normal(-0.2, 1.0)
    return base + delta

# Packet error rate vs SNR (toy curve for small safety packets)

def per_from_snr_db(snr_db: float) -> float:
    # Smooth logistic: PER~1 at low SNR, ~0 above 12 dB
    x = (snr_db - 12.0) / 2.5
    per = 1.0 / (1.0 + math.exp(4.0 * x))
    return min(max(per, 0.0), 1.0)

# Lane A redundancy (N=2 repetitions) and latency estimate

def deliver_laneA(envA: str, envB: str | None) -> Tuple[bool, bool, float]:
    # Returns (to_trainA, to_trainB, latency_ms)
    if not risk_flag:
        return (False, False, 0.0)
    # Draw SNR & PER for each copy per train
    def one_train(env: str) -> Tuple[bool, float]:
        snr1 = draw_snr(env); snr2 = draw_snr(env)
        per1 = per_from_snr_db(snr1); per2 = per_from_snr_db(snr2)
        succ = (rng.random() > per1) or (rng.random() > per2)
        # Latency: base 40 ms + jitter per env
        lat = 40.0 + rng.normal(15.0 if env != "UMi" else 25.0, 8.0)
        return succ, max(lat, 10.0)

    succA, latA = one_train(envA)
    succB, latB = (False, 0.0)
    if envB is not None:
        succB, latB = one_train(envB)
    latency_ms = max(latA, latB)
    return succA, succB, latency_ms

# Compute environments at train positions
envA = route_df.iloc[pos_A]["env"]
envB = route_df.iloc[pos_B]["env"] if (two_trains and pos_B is not None) else None

succA, succB, laneA_latency = deliver_laneA(envA, envB)

# Lane B delivery (best-effort batching)
if emit_laneB:
    st.session_state.laneB_bytes += 900  # pretend a JSON kb summary

# If an alert fired, account bytes and log
if risk_flag:
    alert_obj = {
        "event_id": f"SEG14-{st.session_state.tick}",
        "type": "track_buckling_risk",
        "segment": "Segment 14 (Gävle)",
        "severity": "high",
        "affected": ["Train A"] + (["Train B"] if two_trains else []),
        "recommendation": recommendation,
        "confidence": round(confidence, 2),
        "evidence": {
            "rail_temp_C": round(rail_temp, 1),
            "strain_kN": round(strain, 1),
            "temp_grad_C_per_10min": round(temp_grad, 1)
        },
        "ts": st.session_state.tick
    }
    payload_bytes = len(json.dumps(alert_obj).encode("utf-8"))
    # two copies per receiver (simple), to control center + each train
    copies = 2 * (1 + (1 if two_trains else 0))
    st.session_state.laneA_bytes += payload_bytes * copies

    # Delivery log (who got it)
    delivered_to = []
    if succA:
        delivered_to.append("Train A")
    if two_trains and succB:
        delivered_to.append("Train B")
    if succA or (two_trains and succB):
        st.session_state.last_latency_ms = laneA_latency
        st.session_state.alerts.append({"tick": st.session_state.tick, "delivered_to": delivered_to, "alert": alert_obj})

# ========= UI Layout =========
col_map, col_panels = st.columns([2, 1])

with col_map:
    st.subheader("Live Map (OpenStreetMap)")
    # Polyline for route
    route_layer = pdk.Layer(
        "PathLayer",
        data=pd.DataFrame({
            "path": [[(lon, lat) for lat, lon in zip(route_df.lat, route_df.lon)]],
            "name": ["Sundsvall↔Stockholm"]
        }),
        get_path="path",
        get_width=4,
        width_min_pixels=2,
        get_color=[80, 80, 200]
    )
    # Risk segment highlight
    risk_df = route_df.iloc[risk_start:risk_end]
    risk_layer = pdk.Layer(
        "PathLayer",
        data=pd.DataFrame({
            "path": [[(lon, lat) for lat, lon in zip(risk_df.lat, risk_df.lon)]],
            "name": ["Risk Segment (Gävle)"]
        }),
        get_path="path",
        get_width=8,
        width_min_pixels=4,
        get_color=[220, 60, 60]
    )

    # Train markers
    trains = [{"id": "Train A", "lat": route_df.iloc[pos_A].lat, "lon": route_df.iloc[pos_A].lon}]
    if two_trains and pos_B is not None:
        trains.append({"id": "Train B", "lat": route_df.iloc[pos_B].lat, "lon": route_df.iloc[pos_B].lon})

    trains_df = pd.DataFrame(trains)
    train_layer = pdk.Layer(
        "ScatterplotLayer",
        data=trains_df,
        get_position='[lon, lat]',
        get_radius=120,
        pickable=True,
        radius_min_pixels=6,
        radius_max_pixels=20,
        get_fill_color=[20, 170, 20]
    )

    view_state = pdk.ViewState(latitude=60.3, longitude=17.5, zoom=6.2, bearing=0, pitch=0)

    st.pydeck_chart(pdk.Deck(layers=[route_layer, risk_layer, train_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9'))

    # Raw data charts (Rail temp, Strain, Solar, Wind)
    st.subheader("Raw streams (current tick)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rail temp (°C)", f"{rail_temp:.1f}")
    m2.metric("Strain (kN)", f"{strain:.1f}")
    m3.metric("Temp gradient (°C/10min)", f"{temp_grad:.1f}")
    m4.metric("Wind (m/s)", f"{wind:.1f}")

with col_panels:
    st.subheader("Semantic Communication")
    if risk_flag:
        st.success("Lane A: ⚠ Buckling risk detected at Segment 14 — TSR 60 km/h recommended")
        st.json(alert_obj)
        dlA = "✅" if succA else "❌"
        dlB = "✅" if (two_trains and succB) else ("—" if not two_trains else "❌")
        st.write(f"Delivery → Train A: {dlA} • Train B: {dlB}")
        if st.session_state.last_latency_ms is not None:
            st.write(f"Estimated end‑to‑end latency: {st.session_state.last_latency_ms:.0f} ms")
    else:
        st.info("No Lane A safety alerts at this tick. Lane B maintenance summaries may still be emitted periodically.")

    st.divider()
    st.subheader("Bandwidth usage (cumulative)")
    bw_df = pd.DataFrame({
        "Lane": ["Lane A (alerts)", "Lane B (ops summaries)"],
        "Bytes": [st.session_state.laneA_bytes, st.session_state.laneB_bytes]
    })
    bar = alt.Chart(bw_df).mark_bar().encode(x="Lane", y="Bytes")
    st.altair_chart(bar, use_container_width=True)

    st.caption("Lane A uses compact, redundant alerts; Lane B batches larger summaries.")

st.divider()

# ========= Logs =========
st.subheader("Recent delivered alerts")
if len(st.session_state.alerts) == 0:
    st.write("No alerts delivered yet.")
else:
    logs = [{
        "tick": a["tick"],
        "delivered_to": ", ".join(a["delivered_to"]) or "—",
        "confidence": a["alert"]["confidence"],
        "rail_temp_C": a["alert"]["evidence"]["rail_temp_C"],
        "strain_kN": a["alert"]["evidence"]["strain_kN"],
        "segment": a["alert"]["segment"]
    } for a in st.session_state.alerts[-10:][::-1]]
    st.dataframe(pd.DataFrame(logs), use_container_width=True)

st.caption("This demo approximates radio/channel effects and environmental signals for clarity. For research use, swap in 3GPP TR 38.901 channel models and real sensor feeds.")
