# streamlit_app.py
# -------------------------------------------------------------
# ENSURE‑6G • Railway Safety Demo (Sundsvall → Stockholm)
# Raw vs Semantic vs Hybrid Communication • OpenStreetMap + Folium
# Now with COMMUNICATION CHANNEL MODELS (GSM‑R vs FRMCS/5G):
#  - Environment-aware path loss (Hata 900 MHz, 3GPP-style RMa/UMa proxies)
#  - Shadowing (log-normal) with spatial correlation
#  - Small-scale fading (Rayleigh / Rician) + Doppler note
#  - SNR → PER mapping, Lane A repetition gain, latency estimation
#  - Delivery success/fail for Lane A alerts (advisory path)
# -------------------------------------------------------------
# Features
# - OpenStreetMap via folium + streamlit-folium
# - Animated trains (🚆) on Sundsvall→Stockholm polyline
# - Separate Train Speed (km/h) vs Animation Speed (steps/sec)
# - Modes: RAW / SEMANTIC / HYBRID with bandwidth + latency + PER
# - Lane A (safety alerts) vs Lane B (ops/maintenance)
# - Sensors & Base Stations markers, risk zone near Gävle
# - End-to-end flow section
# -------------------------------------------------------------
# Requirements:
#   streamlit==1.37.1
#   streamlit-folium==0.19.0
#   folium==0.17.0
#   numpy==1.26.4
#   pandas==2.2.2
#   shapely==2.0.4
# -------------------------------------------------------------

import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ENSURE‑6G • Raw vs Semantic vs Hybrid (Rail Demo)", layout="wide")
st.title("🚆 ENSURE‑6G: Raw vs Semantic vs Hybrid — Control Center (TMS) Dashboard")
st.caption("Sundsvall → Stockholm • TMS view • Sensors + Base Stations • Lane A safety alerts vs Lane B ops • Adjustable speeds • With channel model")

# ------------------------
# Helpers
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

# Log-normal shadowing with spatial correlation
class ShadowingTrack:
    def __init__(self, sigma_db=7.0, decor_m=80.0, seed=1):
        self.sigma = sigma_db
        self.decor = decor_m
        self.rng = np.random.default_rng(seed)
        self.val = 0.0
        self.last_s = 0.0
    def sample(self, s_pos_m):
        ds = max(0.0, s_pos_m - self.last_s)
        rho = math.exp(-ds / self.decor)
        w = self.rng.normal(0, self.sigma)
        self.val = rho * self.val + math.sqrt(1 - rho*rho) * w
        self.last_s = s_pos_m
        return self.val

# Small-scale fading (dB) Rayleigh or Rician
rng_fade = np.random.default_rng(2)

def fading_db(model: str, K_dB: float = 8.0):
    if model == "rayleigh":
        # Rayleigh power ~ exp(1); convert to dB gain
        h = rng_fade.rayleigh(scale=1.0)
        p_lin = h*h
        return 10*math.log10(p_lin + 1e-9)
    else:  # rician
        K = 10**(K_dB/10)
        s = math.sqrt(K/(K+1))
        sigma = math.sqrt(1/(2*(K+1)))
        i = rng_fade.normal(s, sigma)
        q = rng_fade.normal(0, sigma)
        p_lin = i*i + q*q
        return 10*math.log10(p_lin + 1e-9)

# Path loss models (simple, pragmatic)

def hata_pl_urban_900mhz(d_km, hb=30, hm=1.5):
    # Okumura-Hata urban PL in dB for 900 MHz (GSM-R proxy)
    f_mhz = 900.0
    a_hm = (1.1*math.log10(f_mhz)-0.7)*hm - (1.56*math.log10(f_mhz)-0.8)
    L = 69.55 + 26.16*math.log10(f_mhz) - 13.82*math.log10(hb) - a_hm + (44.9 - 6.55*math.log10(hb))*math.log10(max(d_km,1e-3))
    return L

# 3GPP-like RMa/UMa proxies for 3.5 GHz NR (very simplified)

def rma_pl_35ghz(d_m, h_bs=35, h_ue=1.5):
    f = 3.5  # GHz
    d = max(d_m, 10.0)
    L = 32.4 + 20*math.log10(f) + 21*math.log10(d/1000.0)  # FSPL-ish + extra loss
    # rural macro extra term
    return L + 8.0

def uma_pl_35ghz(d_m, h_bs=25, h_ue=1.5):
    f = 3.5
    d = max(d_m, 10.0)
    L = 32.4 + 20*math.log10(f) + 31.9*math.log10(d/1000.0)
    return L + 12.0

# SNR → PER mapping (toy logistic)

def per_from_snr(snr_db, mcs="urlc"):
    if mcs == "urlc":
        # Targeting very short Lane A packets with robust coding
        a, b = 0.9, 2.0
        return 1.0/(1.0 + math.exp((snr_db - 1.5)*b/a))  # ~1% PER around 6–8 dB
    else:
        a, b = 1.3, 2.0
        return 1.0/(1.0 + math.exp((snr_db - 6)*b/a))

# Noise floor

def noise_floor_dbm(bw_hz, nf_db=7.0):
    return -174 + 10*math.log10(bw_hz) + nf_db

# ------------------------
# Route & geo
# ------------------------
ROUTE = [
    (62.3913, 17.3063),  # Sundsvall C
    (61.3039, 17.0600),  # Hudiksvall vicinity
    (60.6749, 17.1413),  # Gävle C
    (59.8586, 17.6389),  # Uppsala C
    (59.3293, 18.0686),  # Stockholm C
]
route_ls = LineString([(lon, lat) for lat, lon in ROUTE])
seg_lengths = [haversine(ROUTE[i], ROUTE[i+1]) for i in range(len(ROUTE)-1)]
route_len_m = float(sum(seg_lengths))

# Sensors along track
N_SENSORS = 18
sensor_positions = []
for i in range(N_SENSORS):
    frac = i / (N_SENSORS - 1)
    # more density near Gävle
    if 0.35 < frac < 0.55:
        frac = 0.45 + (frac - 0.45) * 0.6
    dist = frac * route_len_m
    point = route_ls.interpolate(dist)
    sensor_positions.append((point.y, point.x))

# Base stations (macro sites)
BS_POINTS = [
    (62.3913, 17.3063),  # Sundsvall
    (61.80, 17.10),
    (60.6749, 17.1413),  # Gävle
    (60.25, 17.40),
    (59.8586, 17.6389),  # Uppsala
    (59.55, 17.85),
    (59.3293, 18.0686),  # Stockholm
]

risk_center = (60.6749, 17.1413)
risk_radius_m = 15000

# Environment tags along the line (very simple):
# near cities → UMa; elsewhere → RMa; stations → UMi can be implied by shorter distance
CITY_POINTS = [(62.3913,17.3063),(60.6749,17.1413),(59.8586,17.6389),(59.3293,18.0686)]

# Shadowing along-track (shared for simplicity)
shadow = ShadowingTrack(sigma_db=7.0, decor_m=80.0, seed=42)

# ------------------------
# App state & controls
# ------------------------
if "t_sim" not in st.session_state:
    st.session_state.t_sim = 0.0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "last_tick" not in st.session_state:
    st.session_state.last_tick = time.time()

with st.sidebar:
    st.header("Scenario Controls")
    mode = st.radio("Communication Mode", ["RAW", "SEMANTIC", "HYBRID"], index=2)

    bearer = st.selectbox("Bearer / Radio", ["GSM‑R (900 MHz)", "FRMCS / 5G‑NR (3.5 GHz)"])
    train_speed_kmh = st.slider("Train Speed (km/h)", 60, 200, 140, 10)
    anim_speed = st.slider("Animation Speed (steps/sec)", 0, 10, 2, 1,
                           help="Controls UI update rate; train speed stays physical.")

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
    with colb3:
        _ = st.caption("Use Play/Pause to animate")

    st.markdown("---")
    st.subheader("Lane A QoS")
    laneA_latency_ms = st.slider("Target latency (ms)", 20, 500, 100, 10)
    laneA_reps = st.slider("Repetitions (reliability boost)", 1, 3, 2, 1)

# Time / animation
now = time.time()
elapsed_real = now - st.session_state.last_tick
if st.session_state.playing and anim_speed > 0:
    st.session_state.t_sim += elapsed_real * anim_speed
st.session_state.last_tick = now

train_speed_mps = train_speed_kmh / 3.6
s_pos = (train_speed_mps * st.session_state.t_sim) % route_len_m
pt = route_ls.interpolate(s_pos)
train_lat, train_lon = pt.y, pt.x

opp_s = (route_len_m - s_pos) % route_len_m
pt2 = route_ls.interpolate(opp_s)
opp_lat, opp_lon = pt2.y, pt2.x

# ------------------------
# Sensor synthesis (risk bubble near Gävle)
# ------------------------
sim_minutes = st.session_state.t_sim / 60.0
base_temp = 24 + 10 * math.sin(2*math.pi * (sim_minutes % 1440) / 1440)

rows = []
for i, (lat, lon) in enumerate(sensor_positions):
    d_risk = haversine((lat, lon), risk_center)
    risk_boost = max(0, 1 - d_risk / risk_radius_m) * 14
    rail_temp = base_temp + np.random.normal(0, 0.6) + risk_boost
    strain = max(0.0, (rail_temp - 35) * 0.8 + np.random.normal(0, 0.5))
    ballast = max(0.0, np.random.normal(0.3, 0.1) + 0.02 * risk_boost)

    exceeded = []
    if rail_temp >= 38: exceeded.append("temp>38")
    if strain >= 10: exceeded.append("strain>10")
    risk_score = min(1.0, 0.01 * (rail_temp - 30)**2 + 0.04 * max(0, strain-8))
    risk_label = "high" if risk_score > 0.75 else ("medium" if risk_score > 0.4 else "low")

    rows.append({
        "id": f"S{i:02d}",
        "lat": lat, "lon": lon,
        "rail_temp_C": round(rail_temp, 1),
        "strain_kN": round(strain, 1),
        "ballast_idx": round(ballast, 2),
        "risk_score": round(risk_score, 2),
        "risk_label": risk_label,
        "exceeded": exceeded,
    })

sdf = pd.DataFrame(rows)

# Nearest BS to train (for radio link calc)
bs_dists = [haversine((train_lat, train_lon), b) for b in BS_POINTS]
bs_idx = int(np.argmin(bs_dists))
bs_lat, bs_lon = BS_POINTS[bs_idx]
link_d_m = max(10.0, bs_dists[bs_idx])

# Environment selection: near cities → UMa, else RMa; within 2 km of a city point → UMa
env = "RMa"
for c in CITY_POINTS:
    if haversine((train_lat, train_lon), c) < 2000:
        env = "UMa"
        break

# Channel calculations
if bearer.startswith("GSM"):
    # Tx / BW / gains
    tx_dbm = 43.0  # 20 W
    bw_hz = 200e3
    g_bs = 15.0; g_ue = 0.0
    nf_db = 7.0
    pl_db = hata_pl_urban_900mhz(link_d_m/1000.0, hb=30, hm=1.5)
    # fading: mixed; use Rayleigh in UMa, Rician in RMa
    fad_db = fading_db("rayleigh" if env=="UMa" else "rician", K_dB=8)
else:  # FRMCS / 5G‑NR @ 3.5 GHz
    tx_dbm = 46.0  # 40 W
    bw_hz = 10e6
    g_bs = 17.0; g_ue = 0.0
    nf_db = 7.0
    if env == "UMa":
        pl_db = uma_pl_35ghz(link_d_m)
    else:
        pl_db = rma_pl_35ghz(link_d_m)
    fad_db = fading_db("rayleigh" if env=="UMa" else "rician", K_dB=10)

# Shadowing (spatially correlated)
sh_db = shadow.sample(s_pos)

rx_dbm = tx_dbm + g_bs + g_ue - pl_db + sh_db + fad_db
n_dbm = noise_floor_dbm(bw_hz, nf_db)
snr_db = rx_dbm - n_dbm

# PER & latency per mode
per_laneA_single = per_from_snr(snr_db, mcs="urlc")
per_laneA_total = per_laneA_single ** laneA_reps  # repetition combining (OR)
# Latency base by bearer
base_lat_ms = 150 if bearer.startswith("GSM") else 25
# Add queueing jitter proportional to load (more RAW → more jitter)
RAW_RATE_HZ = 2.0; HYBRID_RATE_HZ = 0.2
raw_points_sent = 0

laneA_alerts = []
laneB_msgs = []

# Communications payload accounting + generate alerts
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
            "recommendation": {"tsr_kmh": 60},
        }
        laneA_alerts.append(alert)

BYTES_RAW_SAMPLE = 24
BYTES_ALERT = 280
BYTES_SUMMARY = 180
raw_bps = raw_points_sent * BYTES_RAW_SAMPLE

# Lane B summary once per tick in SEMANTIC/HYBRID
if mode in ("SEMANTIC", "HYBRID"):
    laneB_msgs.append({
        "type": "maintenance_summary",
        "ballast_hotspots": int((sdf.ballast_idx > 0.6).sum()),
        "window": f"t={int(st.session_state.t_sim)}s"
    })

alert_bps = len(laneA_alerts) * BYTES_ALERT
laneB_bps = len(laneB_msgs) * BYTES_SUMMARY
bps_total = raw_bps + alert_bps + laneB_bps

# Latency jitter increases with load (toy):
load_factor = min(1.0, bps_total / (200_000 if bearer.startswith("GSM") else 5_000_000))
jitter_ms = (80 if bearer.startswith("GSM") else 10) * (0.3 + 0.7*load_factor)
latency_laneA_ms = base_lat_ms + jitter_ms

# Delivery simulation for Lane A alerts (per tick): success if any repetition gets through
rng = np.random.default_rng(7)
alerts_delivered = 0
alerts_dropped = 0
for _ in laneA_alerts:
    p_succ = 1 - per_laneA_total
    if rng.random() < p_succ:
        alerts_delivered += 1
    else:
        alerts_dropped += 1

# ------------------------
# UI Layout
# ------------------------
col1, col2 = st.columns([2.2, 1.8])

with col1:
    st.subheader("Live Map • OpenStreetMap (Plotly)")

    # Build Plotly Mapbox figure
    mid = ROUTE[len(ROUTE)//2]
    fig = go.Figure()

    # Route polyline
    route_lats = [p[0] for p in ROUTE]
    route_lons = [p[1] for p in ROUTE]
    fig.add_trace(go.Scattermapbox(
        lat=route_lats, lon=route_lons, mode="lines",
        line=dict(width=4), name="Route"))

    # Risk circle polygon (approx)
    lat0, lon0 = risk_center
    R = 6371000.0
    m2deg_lat = 1/111111.0
    m2deg_lon = 1/(111111.0*math.cos(math.radians(lat0)))
    theta = np.linspace(0, 2*math.pi, 120)
    circ_lat = lat0 + (risk_radius_m * np.sin(theta)) * m2deg_lat
    circ_lon = lon0 + (risk_radius_m * np.cos(theta)) * m2deg_lon
    fig.add_trace(go.Scattermapbox(
        lat=circ_lat, lon=circ_lon, mode="lines",
        line=dict(width=2), name="Risk zone", fill="toself", opacity=0.15))

    # Sensors (colored by risk)
    risk_color = {"low":"green","medium":"orange","high":"red"}
    fig.add_trace(go.Scattermapbox(
        lat=sdf["lat"], lon=sdf["lon"], mode="markers",
        marker=dict(size=8, color=[sdf.loc[i, "risk_label"].replace("low","green").replace("medium","orange").replace("high","red") for i in sdf.index]),
        text=[f"{row['id']} • temp {row['rail_temp_C']}°C • strain {row['strain_kN']} kN • risk {row['risk_label']}" for _, row in sdf.iterrows()],
        hoverinfo="text", name="Sensors"))

    # Base Stations
    bs_lat = [p[0] for p in BS_POINTS]
    bs_lon = [p[1] for p in BS_POINTS]
    fig.add_trace(go.Scattermapbox(
        lat=bs_lat, lon=bs_lon, mode="markers",
        marker=dict(size=11, symbol="triangle", allowoverlap=True), name="Base Stations",
        text=[f"BS{i+1}" for i in range(len(BS_POINTS))], hoverinfo="text"))

    # Trains (as text markers with emoji)
    fig.add_trace(go.Scattermapbox(
        lat=[train_lat], lon=[train_lon], mode="text",
        text=["🚆"], textfont=dict(size=24), name="Train A (southbound)", hoverinfo="skip"))
    fig.add_trace(go.Scattermapbox(
        lat=[opp_lat], lon=[opp_lon], mode="text",
        text=["🚆"], textfont=dict(size=24), name="Train B (northbound)", hoverinfo="skip"))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=mid[0], lon=mid[1]), zoom=6),
        margin=dict(l=0, r=0, t=0, b=0), height=520,
        legend=dict(orientation="h", yanchor="bottom", y=0.01))

    st.plotly_chart(fig, use_container_width=True)
    st.caption("This is the **Control Center (TMS)** view. Train position uses *Train Speed*. Animation pacing uses *Animation Speed*. They are independent.")

with col2:
    st.subheader("Comms, Channel & Safety Status")

    # Headline risk near train
    nearest_idx = ((sdf["lat"]-train_lat)**2 + (sdf["lon"]-train_lon)**2).idxmin()
    nearest = sdf.loc[nearest_idx]
    risk_nearby = nearest["risk_label"] in ("medium", "high")

    st.markdown(f"**Nearest sensor**: {nearest['id']} • temp **{nearest['rail_temp_C']}°C** • strain **{nearest['strain_kN']} kN** • risk **{nearest['risk_label'].upper()}**")

    if risk_nearby and mode in ("SEMANTIC", "HYBRID"):
        st.success("Lane A: ⚠ Buckling risk detected ahead. TSR 60 km/h. Alert sent to both trains + TMS.")
    elif risk_nearby and mode == "RAW":
        st.warning("RAW mode: Elevated readings nearby. Dispatcher must interpret raw graphs before issuing TSR.")
    else:
        st.info("No immediate risks. Monitoring…")

    st.markdown("---")
    st.markdown(f"**Radio:** {bearer} • **Env:** {env} • **Serving BS:** BS{bs_idx+1}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance to BS", f"{link_d_m/1000:.2f} km")
    c2.metric("Rx SNR", f"{snr_db:.1f} dB")
    c3.metric("PER (single)", f"{per_laneA_single*100:.1f}%")
    c4.metric("PER (×{laneA_reps})", f"{per_laneA_total*100:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Lane A latency", f"{latency_laneA_ms:.0f} ms")
    c6.metric("Alerts this tick", f"{len(laneA_alerts)}")
    c7.metric("Delivered", f"{alerts_delivered}")
    c8.metric("Dropped", f"{alerts_dropped}")

    st.markdown("---")
    st.markdown("**Estimated bandwidth (this tick)**")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("RAW (bps)", f"{raw_bps:,.0f}")
    d2.metric("Lane A (bps)", f"{alert_bps:,.0f}")
    d3.metric("Lane B (bps)", f"{laneB_bps:,.0f}")
    d4.metric("Total (bps)", f"{bps_total:,.0f}")

    st.markdown(f"*Mode **{mode}** → RAW rate per sensor: {RAW_RATE_HZ if mode=='RAW' else (HYBRID_RATE_HZ if mode=='HYBRID' else 0)} Hz; Lane A alerts: {len(laneA_alerts)}; Lane B msgs: {len(laneB_msgs)}.*")

    st.markdown("**Lane A (Safety Alerts)**")
    if laneA_alerts:
        st.json(laneA_alerts[:5])
        if len(laneA_alerts) > 5:
            st.caption(f"(+{len(laneA_alerts)-5} more)")
    else:
        st.caption("No active Lane A alerts in this tick.")

    st.markdown("**Lane B (Ops/Maintenance)**")
    if laneB_msgs:
        st.json(laneB_msgs)
    else:
        st.caption("No Lane B messages in RAW-only mode.")

# ------------------------
# End‑to‑End flow explainer
# ------------------------
st.markdown("---")
st.subheader("End‑to‑End Flow (Semantic Communication)")
flow_cols = st.columns([1.3, 1, 1])
with flow_cols[0]:
    st.markdown("""
**1. Trackside sensors** → rail temp, strain, ballast, catenary sag  
**2. Wayside BS / Edge** → semantic encoder (AI + physics), classifies risk  
**3. Network** → GSM‑R / FRMCS 5G; Lane A gets priority + repetitions  
**4. Control Center (TMS)** → correlates, issues TSR  
**5. Train DAS** → synchronized safety alerts (advisory)  
**6. Maintenance** → summaries for inspections
    """)
with flow_cols[1]:
    st.markdown(f"Lane A target latency: **{laneA_latency_ms} ms** • Reps: **x{laneA_reps}** • PER(single): **{per_laneA_single*100:.1f}%**")
with flow_cols[2]:
    st.caption("RMa=Rural Macro, UMa=Urban Macro. Rician in LOS, Rayleigh in NLOS. Shadowing correlated along track.")

# Auto‑rerun for animation
if st.session_state.playing and anim_speed > 0:
    delay_ms = int(500 / max(anim_speed, 1))
    st.markdown(f"<script>setTimeout(() => window.parent.postMessage({{type:'streamlit:rerun'}}, '*'), {delay_ms});</script>", unsafe_allow_html=True)
