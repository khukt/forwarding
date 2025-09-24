# app.py
# ENSURE-6G • Sundsvall→Stockholm Rail Demo: Raw vs Semantic vs Hybrid
# Self-contained Streamlit simulator with OpenStreetMap, trains, sensors, BS, and E2E comms.

import math
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="ENSURE-6G • Rail Semantic Comms Demo",
    layout="wide",
    page_icon="🚆",
)

# =========================
# Helpers
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def polyline_length_km(points: List[Tuple[float, float]]) -> float:
    return sum(haversine_km(points[i][0], points[i][1], points[i+1][0], points[i+1][1]) for i in range(len(points)-1))

def interpolate_polyline(points: List[Tuple[float,float]], step_km: float) -> List[Tuple[float,float,float]]:
    """Return (lat, lon, s_km) samples along polyline at ~step_km spacing."""
    if len(points) < 2: return []
    out = [(points[0][0], points[0][1], 0.0)]
    acc = 0.0
    for i in range(len(points)-1):
        (a_lat, a_lon), (b_lat, b_lon) = points[i], points[i+1]
        seg_len = haversine_km(a_lat, a_lon, b_lat, b_lon)
        n = max(1, int(seg_len / step_km))
        for k in range(1, n+1):
            t = k / n
            lat = a_lat + (b_lat - a_lat) * t
            lon = a_lon + (b_lon - a_lon) * t
            acc += seg_len / n
            out.append((lat, lon, acc))
    return out

def find_index_by_distance(samples, target_km):
    # samples: list of (lat, lon, s_km)
    if target_km <= 0: return 0
    if target_km >= samples[-1][2]: return len(samples) - 1
    # binary search
    lo, hi = 0, len(samples) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid][2] < target_km:
            lo = mid + 1
        else:
            hi = mid
    return lo

def gaussian(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

# =========================
# Route + scenario (Sundsvall→Stockholm main stops)
# =========================
WAYPOINTS = [
    (62.3908, 17.3069),  # Sundsvall C
    (61.7281, 17.1016),  # Hudiksvall
    (61.3037, 17.0590),  # Söderhamn
    (60.6749, 17.1413),  # Gävle
    (60.3441, 17.5203),  # Tierp
    (59.8586, 17.6389),  # Uppsala
    (59.3293, 18.0686),  # Stockholm C
]
ROUTE_STEP_KM = 1.0
SAMPLES = interpolate_polyline(WAYPOINTS, ROUTE_STEP_KM)
TOTAL_KM = SAMPLES[-1][2]

# Define a realistic "risk segment" near Gävle (center around Gävle distance)
gävle_idx = find_index_by_distance(SAMPLES,  polyline_length_km(WAYPOINTS[:4]))  # up to Gävle
GAVLE_S_KM = SAMPLES[gävle_idx][2]
RISK_CENTER_KM = GAVLE_S_KM + 5.0       # 5 km south of Gävle
RISK_SPREAD_KM = 15.0                   # risk zone spread

# Place sensors every 20 km, BS every 40 km
SENSOR_EVERY_KM = 20.0
BS_EVERY_KM = 40.0
sensor_positions = [SAMPLES[find_index_by_distance(SAMPLES, d)] for d in np.arange(0, TOTAL_KM+1, SENSOR_EVERY_KM)]
bs_positions = [SAMPLES[find_index_by_distance(SAMPLES, d)] for d in np.arange(0, TOTAL_KM+1, BS_EVERY_KM)]

# =========================
# Sidebar – Controls
# =========================
with st.sidebar:
    st.title("ENSURE-6G Demo Controls")
    st.caption("Sundsvall → Stockholm • Semantic Comms for Railway Safety")

    # Simulation setup
    colA, colB = st.columns(2)
    with colA:
        sim_minutes = st.slider("Scenario duration (min)", 5, 30, 12, 1)
    with colB:
        speed_kmh = st.slider("Train speed (km/h)", 60, 200, 160, 10)

    steps_per_sec = st.slider("Animation speed (steps/sec)", 1, 10, 4, 1)

    mode = st.radio(
        "Communication mode",
        ["Raw only", "Semantic only", "Hybrid (Raw + Semantic)"],
        index=2,
        help=(
            "Raw only: high bandwidth telemetry.\n"
            "Semantic only: compressed, event-centric alerts.\n"
            "Hybrid: small, fast safety alerts (Lane A) + periodic ops batches (Lane B)."
        )
    )

    two_trains = st.checkbox("Enable two-train scenario (both directions)", value=True)

    st.markdown("---")
    st.subheader("Lane priorities")
    laneA_priority = st.selectbox("Lane A bearer", ["GSM-R (legacy)", "5G/FRMCS (URLLC)"], index=1)
    laneB_bearer = st.selectbox("Lane B bearer", ["Best-effort IP", "5G eMBB"], index=0)

    st.markdown("---")
    st.subheader("Play / Pause")
    play = st.toggle("Play simulation", value=True)
    reset = st.button("↺ Reset")

# =========================
# Session State
# =========================
if "t0" not in st.session_state:
    st.session_state.t0 = time.time()
if "tick" not in st.session_state:
    st.session_state.tick = 0
if "s_km_A" not in st.session_state:
    st.session_state.s_km_A = 0.0
if "s_km_B" not in st.session_state:
    st.session_state.s_km_B = TOTAL_KM  # opposite direction
if "stats" not in st.session_state:
    st.session_state.stats = {
        "raw_bytes": 0,
        "semantic_bytes": 0,
        "laneA_alerts": 0,
        "laneB_batches": 0,
        "laneA_latency_ms": [],
        "lost_alerts": 0,
    }

if reset:
    st.session_state.t0 = time.time()
    st.session_state.tick = 0
    st.session_state.s_km_A = 0.0
    st.session_state.s_km_B = TOTAL_KM
    st.session_state.stats = {
        "raw_bytes": 0,
        "semantic_bytes": 0,
        "laneA_alerts": 0,
        "laneB_batches": 0,
        "laneA_latency_ms": [],
        "lost_alerts": 0,
    }
    st.experimental_rerun()

# Auto-refresh for animation
if play:
    st_autorefresh(interval=int(1000/steps_per_sec), key="autorefresh_key")

# =========================
# Scenario Dynamics
# =========================
# Time base: map "sim_minutes" to the time it takes Train A to go full route at chosen speed.
route_hours = TOTAL_KM / speed_kmh
sim_hours = sim_minutes / 60.0
# Advance per tick (km)
km_per_tick = (TOTAL_KM / (sim_hours * steps_per_sec * 60.0)) if play else 0.0

# Move trains
if play:
    st.session_state.s_km_A = min(TOTAL_KM, st.session_state.s_km_A + km_per_tick)
    if two_trains:
        st.session_state.s_km_B = max(0.0, st.session_state.s_km_B - km_per_tick)

# Train positions
idx_A = find_index_by_distance(SAMPLES, st.session_state.s_km_A)
lat_A, lon_A, sA = SAMPLES[idx_A]
if two_trains:
    idx_B = find_index_by_distance(SAMPLES, st.session_state.s_km_B)
    lat_B, lon_B, sB = SAMPLES[idx_B]
else:
    idx_B, lat_B, lon_B, sB = None, None, None, None

# =========================
# Risk model (sun kink near Gävle)
# =========================
# Rail temperature profile (synthetic): base 28C + heat bubble centered at RISK_CENTER
base_temp = 28.0
peak_rise = 14.0
rail_temp_A = base_temp + peak_rise * gaussian(sA, RISK_CENTER_KM, RISK_SPREAD_KM)
rail_temp_B = base_temp + peak_rise * gaussian(sB if two_trains else 0, RISK_CENTER_KM, RISK_SPREAD_KM)

# Add strain as correlated indicator
strain_A = 6 + 8 * gaussian(sA, RISK_CENTER_KM, RISK_SPREAD_KM)
strain_B = 6 + 8 * gaussian(sB if two_trains else 0, RISK_CENTER_KM, RISK_SPREAD_KM)

# Risk threshold
TEMP_THR = 38.0
STRAIN_THR = 10.0
risk_A = (rail_temp_A >= TEMP_THR) and (strain_A >= STRAIN_THR)
risk_B = two_trains and (rail_temp_B >= TEMP_THR) and (strain_B >= STRAIN_THR)

# =========================
# Communications model (toy but explanatory)
# =========================
def laneA_latency_ms_fn(bearer: str) -> float:
    # URLLC is faster; GSM-R slower
    base = 35 if bearer.startswith("5G") else 180
    # add small jitter
    return np.random.normal(base, base*0.1)

def packet_loss_prob_snr(snr_db):
    # toy mapping: low SNR → more loss
    if snr_db >= 20: return 0.001
    if snr_db >= 10: return 0.01
    if snr_db >= 0:  return 0.05
    return 0.15

def approx_snr_db_at(pos_km):
    # Simple shadowing dip near risk center to stress delivery
    base_snr = 18.0
    dip = 6.0 * gaussian(pos_km, RISK_CENTER_KM, 8.0)
    return base_snr - dip + np.random.normal(0, 1.0)

def send_laneA_alert(for_train: str, pos_km: float, evidence: dict):
    snr = approx_snr_db_at(pos_km)
    loss_p = packet_loss_prob_snr(snr)
    # Repetition N=2 for reliability
    success = (np.random.rand() > loss_p) or (np.random.rand() > loss_p)
    if success:
        st.session_state.stats["laneA_alerts"] += 1
        lat_ms = laneA_latency_ms_fn(laneA_priority)
        st.session_state.stats["laneA_latency_ms"].append(lat_ms)
        st.session_state.stats["semantic_bytes"] += 260  # tiny JSON
        return True, lat_ms, snr
    else:
        st.session_state.stats["lost_alerts"] += 1
        st.session_state.stats["semantic_bytes"] += 520
        return False, None, snr

def send_raw_tick(for_train: str):
    # Raw telemetry ~ 20 sensors * 16B each / tick (toy)
    bytes_this = 20 * 16
    st.session_state.stats["raw_bytes"] += bytes_this
    return bytes_this

def send_laneB_batch():
    st.session_state.stats["laneB_batches"] += 1
    st.session_state.stats["semantic_bytes"] += 1500  # summary blob

# Per-tick comms
# RAW
if mode in ["Raw only", "Hybrid (Raw + Semantic)"]:
    _ = send_raw_tick("A")
    if two_trains:
        _ = send_raw_tick("B")

# SEMANTIC (Lane A alerts) – when risk crosses threshold
alerts = []
if mode in ["Semantic only", "Hybrid (Raw + Semantic)"]:
    if risk_A:
        ok, latms, snr = send_laneA_alert("A", sA, {"rail_temp": rail_temp_A, "strain": strain_A})
        alerts.append(("A", ok, latms, snr))
    if risk_B:
        ok, latms, snr = send_laneA_alert("B", sB, {"rail_temp": rail_temp_B, "strain": strain_B})
        alerts.append(("B", ok, latms, snr))
    # Lane B batches every ~3 minutes of sim time
    if (st.session_state.tick % max(1, int(3 * 60 * steps_per_sec))) == 0:
        send_laneB_batch()

st.session_state.tick += 1

# =========================
# Map Layers (pydeck on OpenStreetMap)
# =========================
route_df = pd.DataFrame([{"lat": lat, "lon": lon, "s_km": s} for (lat, lon, s) in SAMPLES])
sensor_df = pd.DataFrame([{"lat": lat, "lon": lon, "s_km": s} for (lat, lon, s) in sensor_positions])
bs_df = pd.DataFrame([{"lat": lat, "lon": lon, "s_km": s} for (lat, lon, s) in bs_positions])
train_df = pd.DataFrame([{"lat": lat_A, "lon": lon_A, "who": "Train A 🚆"}])
if two_trains and lat_B is not None:
    train_df = pd.concat([train_df, pd.DataFrame([{"lat": lat_B, "lon": lon_B, "who": "Train B 🚆"}])], ignore_index=True)

# Colors
COL_ROUTE = [30, 144, 255]      # blue
COL_SENSOR = [0, 200, 0]        # green
COL_BS = [255, 165, 0]          # orange
COL_RISK = [220, 20, 60]        # crimson

# Risk polygon (approx: highlight region by sampling around RISK_CENTER ± spread)
risk_idxs = [find_index_by_distance(SAMPLES, d) for d in np.linspace(RISK_CENTER_KM - RISK_SPREAD_KM, RISK_CENTER_KM + RISK_SPREAD_KM, 20)]
risk_poly = [{"lat": SAMPLES[i][0], "lon": SAMPLES[i][1]} for i in risk_idxs]

layers = [
    # Route polyline
    pdk.Layer(
        "PathLayer",
        data=pd.DataFrame([{"path": [[r["lon"], r["lat"]] for _, r in route_df.iterrows()]}]),
        get_color=COL_ROUTE,
        width_scale=10,
        width_min_pixels=2,
        get_width=2,
    ),
    # Risk zone (filled)
    pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": [[p["lon"], p["lat"]] for p in risk_poly]}],
        get_polygon="polygon",
        get_fill_color=COL_RISK + [40],
        get_line_color=COL_RISK,
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=False,
    ),
    # Sensors
    pdk.Layer(
        "ScatterplotLayer",
        data=sensor_df,
        get_position="[lon, lat]",
        get_fill_color=COL_SENSOR,
        get_line_color=[0, 0, 0],
        get_radius=200,
        pickable=True,
        stroked=True,
    ),
    # Base Stations
    pdk.Layer(
        "ScatterplotLayer",
        data=bs_df,
        get_position="[lon, lat]",
        get_fill_color=COL_BS,
        get_line_color=[0, 0, 0],
        get_radius=250,
        pickable=True,
        stroked=True,
    ),
    # Train labels (emoji)
    pdk.Layer(
        "TextLayer",
        data=train_df,
        get_position="[lon, lat]",
        get_text="who",
        get_size=24,
        get_color=[0, 0, 0, 255],
        get_angle=0,
        get_alignment_baseline="'bottom'",
        pickable=False,
    ),
]

mid_lat = (WAYPOINTS[0][0] + WAYPOINTS[-1][0]) / 2
mid_lon = (WAYPOINTS[0][1] + WAYPOINTS[-1][1]) / 2
view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=6.0, pitch=0)

r = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/light-v9"  # Streamlit uses Mapbox token under the hood
)

# =========================
# Layout
# =========================
st.title("ENSURE-6G: Raw vs Semantic vs Hybrid — Live Rail Demo (OpenStreetMap)")
st.caption("Sundsvall → Stockholm • Base stations + sensors • Moving train(s) • Lane A (safety) vs Lane B (ops) • Strategy comparison")

# Top row: Map + Legend/Help
map_col, legend_col = st.columns([3, 1])

with map_col:
    st.pydeck_chart(r, use_container_width=True)

with legend_col:
    st.subheader("Legend & Risk Model")
    st.markdown(
        """
**Colors:**
- Route = blue  
- Sensors = green • Base Stations = orange  
- Risk zone (sun-kink) = light crimson area near **Gävle**

**Risk trigger:**  
> Rail Temp ≥ **38 °C** AND Strain ≥ **10 kN**

**Mode tips:**  
- **Raw only** → high bandwidth telemetry, slower decisions  
- **Semantic only** → tiny, fast alerts (Lane A), minimal bandwidth  
- **Hybrid** → Lane A alerts + Lane B maintenance batches
        """
    )
    st.markdown("---")
    st.subheader("Current train state")
    st.write(f"Train A @ **{sA:,.1f} km** / {TOTAL_KM:,.0f} km  • Rail Temp ~ **{rail_temp_A:.1f} °C** • Strain ~ **{strain_A:.1f} kN**")
    if two_trains:
        st.write(f"Train B @ **{sB:,.1f} km** / {TOTAL_KM:,.0f} km  • Rail Temp ~ **{rail_temp_B:.1f} °C** • Strain ~ **{strain_B:.1f} kN**")

# Middle row: E2E flow (live status)
st.markdown("---")
st.subheader("End-to-End Communication Flow")
flow_cols = st.columns(4)
with flow_cols[0]:
    st.markdown("**Track Sensors**")
    st.caption("Rail temp, strain, ballast, OHL sag")
    st.success("✓ Streaming")
with flow_cols[1]:
    st.markdown("**Wayside Edge / BS**")
    st.caption("Semantic encoder, priority routing")
    if risk_A or risk_B:
        st.warning("⚠ Risk detected")
    else:
        st.info("No critical event")
with flow_cols[2]:
    st.markdown("**Control Center (TMS)**")
    if mode != "Raw only" and (risk_A or risk_B):
        st.warning("TSR proposed (Lane A)")
    else:
        st.info("Monitoring…")
with flow_cols[3]:
    st.markdown("**Train DAS**")
    if mode != "Raw only" and (risk_A or risk_B):
        st.error("Driver alert: Slow to 60 km/h")
    else:
        st.info("Normal")

# Alerts panel
if alerts:
    st.markdown("### Lane A Alerts (live)")
    for who, ok, latms, snr in alerts:
        if ok:
            st.success(f"{who}: ALERT delivered • ~{latms:.0f} ms • SNR≈{snr:.1f} dB")
        else:
            st.error(f"{who}: ALERT **lost** • SNR≈{snr:.1f} dB (retries failed)")

# Bottom row: Metrics
st.markdown("---")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Raw bytes sent", f"{st.session_state.stats['raw_bytes']:,} B")
m2.metric("Semantic bytes sent", f"{st.session_state.stats['semantic_bytes']:,} B")
m3.metric("Lane A alerts delivered", f"{st.session_state.stats['laneA_alerts']}")
m4.metric("Lane B batches", f"{st.session_state.stats['laneB_batches']}")
avg_lat = np.mean(st.session_state.stats["laneA_latency_ms"]) if st.session_state.stats["laneA_latency_ms"] else 0
m5.metric("Avg Lane A latency", f"{avg_lat:.0f} ms")

# Footnotes / explanation
with st.expander("What am I looking at? (self-explanatory guide)"):
    st.markdown(
        """
**Goal:** Show why **semantic communication** improves railway safety in summer heat (track buckling risk).

**How to use:**
1) Pick **communication mode** in the sidebar.  
   - *Raw only*: continuous telemetry floods the network.  
   - *Semantic only*: tiny, explainable alerts (Lane A); minimal bandwidth.  
   - *Hybrid*: Lane A alerts + Lane B maintenance summaries.
2) Toggle **two-train scenario** to see synchronized alerts in both directions.  
3) Adjust **Animation speed** to speed up or slow down the simulation.  
4) Watch the **map**: trains (🚆) move Sundsvall → Stockholm (and back if enabled).  
   - **Crimson** zone near *Gävle* marks a heat-risk area.  
   - **Green dots** are sensors; **Orange dots** are base stations.
5) Check the **End-to-End flow**: Sensors → Wayside/BS → Control Center → Train DAS.  
6) Inspect **metrics** to compare bandwidth & latency across modes.

**Lane A (Safety-critical):**  
Small JSON alerts like: “⚠ Buckling risk at Segment near Gävle; recommend 60 km/h; confidence 92%.”  
They’re prioritized (URLLC if 5G/FRMCS) and repeated for reliability.

**Lane B (Ops/Maintenance):**  
Periodic summaries (battery health, inspection tasks) sent with best-effort bearers.

**Why it matters:**  
Semantic comms reduces bandwidth, speeds decisions, and gives explainable alerts to drivers & dispatchers—aligned with **TEN-T** goals for a resilient, digital rail network.
        """
    )

# EOF
