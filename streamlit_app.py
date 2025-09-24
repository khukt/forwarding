# streamlit_app.py
# -------------------------------------------------------------
# ENSURE‑6G • Railway Safety Demo (Sundsvall → Stockholm)
# Raw vs Semantic vs Hybrid Communication • OpenStreetMap + Folium
# -------------------------------------------------------------
# Features
# - OpenStreetMap base using folium + streamlit-folium
# - Animated train (🚆) moving along Sundsvall→Stockholm polyline
# - Separate Train Speed (km/h) vs Animation Speed (steps/sec)
# - Modes: RAW / SEMANTIC / HYBRID with bandwidth + latency estimates
# - Lane A (safety alerts) vs Lane B (ops/maintenance)
# - Sensors & Base Stations rendered on the map
# - End-to-end flow diagram (sensor → BS/edge → network → TMS/DAS → train/maintenance)
# - Self-explanatory layout with help text and captions
# -------------------------------------------------------------
# Requirements (Streamlit Cloud):
#   streamlit==1.37.1
#   streamlit-folium==0.19.0
#   folium==0.17.0
#   numpy==1.26.4
#   pandas==2.2.2
#   geopy==2.4.1
#   shapely==2.0.4
#   altair==5.3.0
#   graphviz==0.20.3
#   scikit-learn==1.4.2 (optional)
# -------------------------------------------------------------

import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString, Point

# Map
try:
    import folium
    from streamlit_folium import st_folium
except Exception as e:
    st.error("Missing folium / streamlit-folium. Add them to requirements.txt.\n" + str(e))
    st.stop()

st.set_page_config(page_title="ENSURE‑6G • Raw vs Semantic vs Hybrid (Rail Demo)", layout="wide")
st.title("🚆 ENSURE‑6G: Raw vs Semantic vs Hybrid — Live Rail Demo (OpenStreetMap)")
st.caption("Sundsvall → Stockholm • Sensors + Base Stations • Lane A safety alerts vs Lane B ops • OSM map • Adjustable speeds")

# ------------------------
# Helper: Great-circle distance (meters)
# ------------------------

def haversine(p1, p2):
    R = 6371000.0
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

# ------------------------
# Route polyline (simplified): Sundsvall → Gävle → Uppsala → Stockholm
# ------------------------
ROUTE = [
    (62.3913, 17.3063),  # Sundsvall C
    (61.3039, 17.0600),  # Hudiksvall vicinity
    (60.6749, 17.1413),  # Gävle C
    (59.8586, 17.6389),  # Uppsala C
    (59.3293, 18.0686),  # Stockholm C
]

route_ls = LineString([(lon, lat) for lat, lon in ROUTE])  # shapely uses (x=lon, y=lat)

# Precompute cumulative distances per segment
seg_lengths = []
for i in range(len(ROUTE)-1):
    seg_lengths.append(haversine(ROUTE[i], ROUTE[i+1]))
route_len_m = float(sum(seg_lengths))
seg_cum = np.cumsum([0.0] + seg_lengths).tolist()  # len = len(ROUTE)

# ------------------------
# Sensor + Base Station placement
# ------------------------
# Place N sensors roughly evenly along the route, with a denser cluster near Gävle (risk segment)
N_SENSORS = 18
sensor_positions = []
for i in range(N_SENSORS):
    frac = i / (N_SENSORS - 1)
    # bias a few toward Gävle (~ middle of segment 2)
    if 0.35 < frac < 0.55:
        frac = 0.45 + (frac - 0.45) * 0.6
    dist = frac * route_len_m
    # interpolate along polyline
    point = route_ls.interpolate(dist)
    sensor_positions.append((point.y, point.x))  # (lat, lon)

# Base stations: put at city nodes + midpoints
BS_POINTS = [
    (62.3913, 17.3063),  # Sundsvall
    (61.80, 17.10),      # between Sundsvall & Hudiksvall
    (60.6749, 17.1413),  # Gävle
    (60.25, 17.40),      # between Gävle & Uppsala
    (59.8586, 17.6389),  # Uppsala
    (59.55, 17.85),      # between Uppsala & Stockholm
    (59.3293, 18.0686),  # Stockholm
]

# Risk segment around Gävle (Segment 14 in our story): define kilometer window
risk_center = (60.6749, 17.1413)
risk_radius_m = 15000  # 15 km around Gävle

# ------------------------
# State initialization
# ------------------------
if "sim_start" not in st.session_state:
    st.session_state.sim_start = datetime.now()
if "t_sim" not in st.session_state:
    st.session_state.t_sim = 0.0  # simulated seconds since start
if "playing" not in st.session_state:
    st.session_state.playing = False
if "last_tick" not in st.session_state:
    st.session_state.last_tick = time.time()

# ------------------------
# Sidebar controls
# ------------------------
with st.sidebar:
    st.header("Scenario Controls")
    mode = st.radio("Communication Mode", ["RAW", "SEMANTIC", "HYBRID"], index=2,
                    help="RAW: all raw telemetry. SEMANTIC: only high-level alerts. HYBRID: critical alerts + sparse raw.")

    train_speed_kmh = st.slider("Train Speed (km/h)", 60, 200, 140, 10, help="Physical speed used to compute train position along the route.")
    anim_speed = st.slider("Animation Speed (steps/sec)", 0, 10, 2, 1, help="UI refresh speed. Does NOT change train's physical speed.")

    show_sensors = st.checkbox("Show Sensor Nodes", True)
    show_bs = st.checkbox("Show Base Stations", True)

    colb1, colb2, colb3 = st.columns(3)
    with colb1:
        if st.button("⏯ Play / Pause"):
            st.session_state.playing = not st.session_state.playing
            st.session_state.last_tick = time.time()
    with colb2:
        if st.button("⏮ Reset"):
            st.session_state.t_sim = 0.0
            st.session_state.sim_start = datetime.now()
    with colb3:
        sim_len_min = st.number_input("Sim Length (min)", 5, 120, 30, 5)

    st.markdown("---")
    st.subheader("Lane Priority & QoS")
    laneA_latency_ms = st.slider("Lane A target latency (ms)", 20, 500, 100, 10)
    laneA_redundancy = st.slider("Lane A repetitions", 1, 3, 2, 1)

# ------------------------
# Time update based on animation speed (UI-driven), not physics
# ------------------------
now = time.time()
elapsed_real = now - st.session_state.last_tick
if st.session_state.playing and anim_speed > 0:
    st.session_state.t_sim += elapsed_real * anim_speed  # advance simulated seconds
st.session_state.last_tick = now

# Train physical progress uses train speed * simulated time
train_speed_mps = train_speed_kmh / 3.6
dist_travelled = train_speed_mps * st.session_state.t_sim

# Loop route if exceed length, to keep demo continuous
if dist_travelled > route_len_m:
    dist_travelled = dist_travelled % route_len_m

# Interpolate train position
pt = route_ls.interpolate(dist_travelled)
train_lat, train_lon = pt.y, pt.x

# Also simulate a counter-train from the opposite direction for realism
opp_dist = (route_len_m - dist_travelled) % route_len_m
pt2 = route_ls.interpolate(opp_dist)
opp_lat, opp_lon = pt2.y, pt2.x

# ------------------------
# Simple environment model → sensor readings
# ------------------------
# Day/night temperature profile + local noise; when in risk zone near Gävle, heat up slightly
sim_minutes = st.session_state.t_sim / 60.0
base_temp = 24 + 10 * math.sin(2*math.pi * (sim_minutes % 1440) / 1440)  # 24±10 °C daily

sensor_data = []
for i, (lat, lon) in enumerate(sensor_positions):
    # Add spatial variation + risk bubble near Gävle
    d_risk = haversine((lat, lon), risk_center)
    risk_boost = max(0, 1 - d_risk / risk_radius_m) * 14  # up to +14°C at center
    rail_temp = base_temp + np.random.normal(0, 0.6) + risk_boost
    strain = max(0.0, (rail_temp - 35) * 0.8 + np.random.normal(0, 0.5))  # kN (toy)
    ballast = max(0.0, np.random.normal(0.3, 0.1) + 0.02 * risk_boost)

    # Semantic risk scoring
    exceeded = []
    if rail_temp >= 38: exceeded.append("temp>38")
    if strain >= 10: exceeded.append("strain>10")
    risk_score = min(1.0, 0.01 * (rail_temp - 30)**2 + 0.04 * max(0, strain-8))
    risk_label = "high" if risk_score > 0.75 else ("medium" if risk_score > 0.4 else "low")

    sensor_data.append({
        "id": f"S{i:02d}",
        "lat": lat, "lon": lon,
        "rail_temp_C": round(rail_temp, 1),
        "strain_kN": round(strain, 1),
        "ballast_idx": round(ballast, 2),
        "risk_score": round(risk_score, 2),
        "risk_label": risk_label,
        "exceeded": exceeded,
    })

sdf = pd.DataFrame(sensor_data)

# Determine if train is entering risk window (nearest sensor to train)
nearest_idx = ((sdf["lat"]-train_lat)**2 + (sdf["lon"]-train_lon)**2).idxmin()
nearest = sdf.loc[nearest_idx]

# Produce communication payloads according to mode
RAW_RATE_HZ = 2.0   # each sensor
HYBRID_RATE_HZ = 0.2

laneA_alerts = []  # safety
laneB_msgs = []    # ops/maintenance
raw_points_sent = 0

# RAW mode: send all samples; SEMANTIC: only alerts; HYBRID: sparse raw + alerts
for _, row in sdf.iterrows():
    if mode in ("RAW", "HYBRID"):
        rate = RAW_RATE_HZ if mode == "RAW" else HYBRID_RATE_HZ
        raw_points_sent += int(rate)

    if row["risk_label"] in ("medium", "high") and ("temp>38" in row["exceeded"] or "strain>10" in row["exceeded"]):
        alert = {
            "event": "buckling_risk",
            "sensor": row["id"],
            "location": {"lat": row["lat"], "lon": row["lon"]},
            "severity": row["risk_label"],
            "confidence": round(0.6 + 0.4*row["risk_score"], 2),
            "evidence": {
                "rail_temp_C": row["rail_temp_C"],
                "strain_kN": row["strain_kN"],
                "ballast_idx": row["ballast_idx"],
                "exceeded": row["exceeded"],
            },
            "recommendation": {"tsr_kmh": 60},
            "ttl_s": 900,
        }
        laneA_alerts.append(alert)

# Summarize Lane B once per step
if mode in ("SEMANTIC", "HYBRID"):
    laneB_msgs.append({
        "type": "maintenance_summary",
        "low_battery_nodes": [],  # placeholder
        "ballast_hotspots": int((sdf.ballast_idx > 0.6).sum()),
        "window": f"t={int(st.session_state.t_sim)}s",
    })

# Estimate bandwidth (very rough)
BYTES_RAW_SAMPLE = 24  # e.g., temp+strain+ballast+ts
BYTES_ALERT = 280
BYTES_SUMMARY = 180
raw_bps = raw_points_sent * BYTES_RAW_SAMPLE
alert_bps = len(laneA_alerts) * BYTES_ALERT
laneB_bps = len(laneB_msgs) * BYTES_SUMMARY
bps_total = raw_bps + alert_bps + laneB_bps

# ------------------------
# Layout
# ------------------------
col1, col2 = st.columns([2.2, 1.8])

with col1:
    st.subheader("Live Map • OpenStreetMap")

    # Center the map roughly at the midpoint
    mid = ROUTE[len(ROUTE)//2]
    m = folium.Map(location=mid, zoom_start=6, tiles="OpenStreetMap", control_scale=True)

    # Draw route polyline
    folium.PolyLine([(lat, lon) for lat, lon in ROUTE], color="#0066ff", weight=4, opacity=0.8, tooltip="Route: Sundsvall → Stockholm").add_to(m)

    # Risk zone circle
    folium.Circle(location=risk_center, radius=risk_radius_m, color="#ff3333", weight=2, fill=True, fill_opacity=0.1, tooltip="Risk zone near Gävle").add_to(m)

    # Sensors
    if show_sensors:
        for _, r in sdf.iterrows():
            color = {"low":"green","medium":"orange","high":"red"}[r["risk_label"]]
            folium.CircleMarker(location=(r["lat"], r["lon"]), radius=4, color=color, fill=True, fill_opacity=0.9,
                                tooltip=f"{r['id']} • temp {r['rail_temp_C']}°C • strain {r['strain_kN']} kN • risk {r['risk_label']}").add_to(m)

    # Base Stations
    if show_bs:
        for i, (lat, lon) in enumerate(BS_POINTS):
            folium.Marker((lat, lon), icon=folium.Icon(color="blue", icon="tower", prefix="fa"),
                          tooltip=f"BS{i+1}").add_to(m)

    # Train markers (🚆)
    folium.map.Marker(
        [train_lat, train_lon],
        icon=folium.DivIcon(html=f'<div style="font-size:26px; transform: translate(-12px, -12px);">🚆</div>'),
        tooltip="Train A (southbound)",
    ).add_to(m)

    folium.map.Marker(
        [opp_lat, opp_lon],
        icon=folium.DivIcon(html=f'<div style="font-size:26px; transform: translate(-12px, -12px);">🚆</div>'),
        tooltip="Train B (northbound)",
    ).add_to(m)

    st_folium(m, width=None, height=520)

    st.caption("Train position is based on *Train Speed*. Animation pacing is based on *Animation Speed*. They are independent.")

with col2:
    st.subheader("Comms & Safety Status")

    # Headline card
    risk_nearby = nearest["risk_label"] in ("medium", "high")
    st.markdown(f"**Nearest sensor**: {nearest['id']} • temp **{nearest['rail_temp_C']}°C** • strain **{nearest['strain_kN']} kN** • risk **{nearest['risk_label'].upper()}**")

    if risk_nearby and mode in ("SEMANTIC", "HYBRID"):
        st.success("Lane A: ⚠ Buckling risk detected ahead. Recommend TSR 60 km/h. Alert sent to both trains + TMS.")
    elif risk_nearby and mode == "RAW":
        st.warning("RAW mode: Elevated readings nearby. Dispatcher must interpret raw graphs before issuing TSR.")
    else:
        st.info("No immediate risks. Monitoring…")

    # Bandwidth panel
    st.markdown("---")
    st.markdown("**Estimated bandwidth (this tick)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RAW (bps)", f"{raw_bps:,.0f}")
    c2.metric("Lane A (bps)", f"{alert_bps:,.0f}")
    c3.metric("Lane B (bps)", f"{laneB_bps:,.0f}")
    c4.metric("Total (bps)", f"{bps_total:,.0f}")

    st.markdown(f"*Mode **{mode}** → RAW rate per sensor: {RAW_RATE_HZ if mode=='RAW' else (HYBRID_RATE_HZ if mode=='HYBRID' else 0)} Hz; Lane A alerts: {len(laneA_alerts)}; Lane B msgs: {len(laneB_msgs)}.*")

    # Show current Lane A alerts
    st.markdown("---")
    st.markdown("**Lane A (Safety Alerts)**")
    if laneA_alerts:
        st.json(laneA_alerts[:5])
        if len(laneA_alerts) > 5:
            st.caption(f"(+{len(laneA_alerts)-5} more in this tick)")
    else:
        st.caption("No active Lane A alerts in this tick.")

    # Lane B summaries
    st.markdown("**Lane B (Ops/Maintenance)**")
    if laneB_msgs:
        st.json(laneB_msgs)
    else:
        st.caption("No Lane B messages in SEMANTIC-only RAW mode.")

# ------------------------
# End‑to‑End flow diagram (self-explanatory)
# ------------------------
st.markdown("---")
st.subheader("End‑to‑End Flow (Semantic Communication)")
flow_cols = st.columns([1.2, 1, 1])
with flow_cols[0]:
    st.markdown("""
**1. Trackside sensors** → rail temp, strain, ballast, catenary sag  
**2. Wayside BS / Edge** → runs semantic encoder (AI + physics), classifies risk  
**3. Network** → GSM‑R (today) / FRMCS 5G (future), Lane A = priority  
**4. Control Center (TMS)** → correlates, issues TSR  
**5. Train DAS** → shows synchronized safety alerts  
**6. Maintenance** → gets summaries for inspections
    """)
with flow_cols[1]:
    st.image("https://img.icons8.com/ios-filled/100/sensor.png", caption="Sensors", use_column_width=True)
with flow_cols[2]:
    st.markdown(
        f"Lane A target latency: **{laneA_latency_ms} ms** • Redundancy: **x{laneA_redundancy}**")

st.caption("HYBRID mode = Lane A alerts + sparse RAW telemetry to aid diagnostics (trustworthy + bandwidth‑efficient). RAW mode streams everything and risks operator overload.")

# ------------------------
# Auto‑refresh to animate based on Animation Speed
# ------------------------
# Use empty placeholder to trigger rerun; Streamlit reruns on any widget interaction.
# We emulate a timer with st.experimental_rerun through a tiny sleep in the browser loop via st_autorefresh isn't used here to keep deps minimal.
if st.session_state.playing and anim_speed > 0:
    # Ask Streamlit to rerun after ~500ms / anim_speed
    delay_ms = int(500 / max(anim_speed, 1))
    st.markdown(f"<script>setTimeout(() => window.parent.postMessage({{type:'streamlit:rerun'}}, '*'), {delay_ms});</script>", unsafe_allow_html=True)
