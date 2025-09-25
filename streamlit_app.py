# ENSURE-6G • TMS Rail Demo — PyDeck (robust) + PHY + Macro Capacity Model
# Raw vs Semantic vs Hybrid • Dynamic bearer (5G→LTE→3G→GSM-R)
# PHY: path loss + shadowing + fading → SNR→PER
# Macro: GOOD/PATCHY/POOR → capacity (bps) + random loss → overload queuing
# Deck.gl map (no basemap dependency), Sankey flow, TMS metrics

import math, time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from shapely.geometry import LineString
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ENSURE-6G • TMS Rail Demo (PHY + Capacity)", layout="wide")
st.title("🚆 ENSURE-6G: Raw vs Semantic vs Hybrid — Control Center (TMS) Dashboard")
st.caption("Sundsvall → Stockholm • PHY (SNR→PER, handover) + Macro capacity (GOOD/PATCHY/POOR) • Deck.gl map (no tiles needed)")

# =================== Geography & helpers ===================
R_EARTH = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    p = math.pi/180.0
    dlat = (lat2-lat1)*p
    dlon = (lon2-lon1)*p
    a = math.sin(dlat/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2
    return 2*R_EARTH*math.asin(min(1.0, math.sqrt(a)))

# Sundsvall -> Stockholm with intermediate bends (lat, lon)
RAIL_WAYPOINTS = [
    (62.3930,17.3070),(62.1200,17.1500),(61.8600,17.1400),(61.7300,17.1100),
    (61.5600,17.0800),(61.3900,17.0700),(61.3000,17.0600),(61.0700,17.1000),
    (60.8500,17.1600),(60.6749,17.1413),(60.3800,17.3300),(60.2000,17.4500),
    (60.0500,17.5200),(59.9300,17.6100),(59.8586,17.6389),(59.7500,17.8200),
    (59.6600,17.9400),(59.6100,17.9900),(59.5500,18.0300),(59.4800,18.0400),
    (59.4200,18.0600),(59.3700,18.0700),(59.3293,18.0686),
]

# Macro BS grid: (name, lat, lon, radius_of_good_service_m)
BASE_STATIONS = [
    ("BS-Sundsvall",62.386,17.325,16000),("BS-Njurunda",62.275,17.354,14000),
    ("BS-Harmånger",61.897,17.170,14000),("BS-Hudiksvall",61.728,17.103,15000),
    ("BS-Söderhamn",61.303,17.058,15000),("BS-Axmar",61.004,17.190,14000),
    ("BS-Gävle",60.675,17.141,16000),("BS-Tierp",60.345,17.513,14000),
    ("BS-Skyttorp",60.030,17.580,14000),("BS-Uppsala",59.858,17.639,16000),
    ("BS-Märsta",59.620,17.860,15000),("BS-Stockholm",59.330,18.070,18000),
]

def interpolate_polyline(points, n_pts):
    """
    Resample a polyline to n_pts evenly spaced by arc length.
    Returns DataFrame(lat, lon, s_m).
    """
    n_pts = max(2, int(n_pts))  # need at least 2 samples
    lat = np.array([p[0] for p in points], dtype=float)
    lon = np.array([p[1] for p in points], dtype=float)

    # cumulative distances per vertex
    cum = np.zeros(len(points), dtype=float)
    for i in range(1, len(points)):
        cum[i] = cum[i-1] + haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])

    total = float(cum[-1])
    if total <= 0:
        return pd.DataFrame({"lat":[lat[0]]*n_pts, "lon":[lon[0]]*n_pts, "s_m":np.linspace(0,0,n_pts)})

    tgt = np.linspace(0.0, total, n_pts)

    # search bins; clamp to [1, len(cum)-1] so i0>=0 and i1<=len(cum)-1
    idx = np.searchsorted(cum, tgt, side="right")
    idx = np.clip(idx, 1, len(cum)-1)

    i0 = idx - 1
    i1 = idx
    d0 = cum[i0]
    d1 = cum[i1]
    w = (tgt - d0) / np.maximum(d1 - d0, 1e-9)  # avoid divide-by-zero

    latp = lat[i0] + (lat[i1] - lat[i0]) * w
    lonp = lon[i0] + (lon[i1] - lon[i0]) * w
    return pd.DataFrame({"lat": latp, "lon": lonp, "s_m": tgt})

def label_segments(route_len_sec):
    SEG_NAMES = ["Sundsvall→Hudiksvall","Hudiksvall→Söderhamn","Söderhamn→Gävle","Gävle→Uppsala","Uppsala→Stockholm"]
    bounds = np.linspace(0, route_len_sec, len(SEG_NAMES)+1).astype(int)
    labels = np.empty(route_len_sec, dtype=object)
    for i in range(len(SEG_NAMES)):
        labels[bounds[i]:bounds[i+1]] = SEG_NAMES[i]
    return labels

def nearest_bs_quality(lat, lon):
    """
    Return (bs_name, distance_m, quality) with a simple 3-bin metric:
    GOOD (<=R), PATCHY (<=2.2R), POOR (>2.2R)
    """
    best = None
    for name, blat, blon, R in BASE_STATIONS:
        d = haversine_m(lat, lon, blat, blon)
        if d <= R:
            q = "GOOD"
        elif d <= 2.2*R:
            q = "PATCHY"
        else:
            q = "POOR"
        rank = {"GOOD":0, "PATCHY":1, "POOR":2}[q]
        if best is None or rank < best[3]:
            best = (name, d, q, rank)
    return best[0], best[1], best[2]

def cap_loss(quality, t_sec, base_capacity_kbps=800, burst_factor=1.4,
             good_loss_pct=0.5, bad_loss_pct=10.0):
    """Map quality to capacity (bps) & random loss."""
    cap_bps = int(base_capacity_kbps * 1000)
    if quality == "GOOD":
        cap_bps = int(cap_bps * burst_factor)
        loss = good_loss_pct / 100.0
    elif quality == "PATCHY":
        wobble = 0.6 + 0.2 * math.sin(2*math.pi * (t_sec % 30) / 30.0)
        cap_bps = max(int(cap_bps * wobble * 0.9), 1)
        loss = min(0.4, (bad_loss_pct * 0.5) / 100.0)
    else:
        cap_bps = int(cap_bps * 0.25)
        loss = bad_loss_pct / 100.0
    return max(1, cap_bps), float(loss)

# =================== Sidebar (controls) ===================
with st.sidebar:
    st.header("Scenario Controls (TMS)")
    mode = st.radio("Communication Mode", ["RAW","SEMANTIC","HYBRID"], index=2)
    train_speed_kmh = st.slider("Train Speed (km/h)", 60, 200, 140, 10)
    anim_fps = st.slider("Animation FPS", 1, 20, 6, 1)
    sim_minutes_total = st.number_input("Sim Length (minutes)", 5, 180, 30, 5)
    st.markdown("---")
    st.subheader("Lane A QoS")
    laneA_reps = st.slider("Repetitions (reliability)", 1, 3, 2, 1)
    st.markdown("---")
    st.subheader("Display")
    show_sankey = st.checkbox("Show Communication Flow (Sankey)", True)
    c1, c2 = st.columns(2)
    if c1.button("⏯ Play/Pause"): st.session_state.playing = not st.session_state.get("playing", True)
    if c2.button("⏮ Reset"): st.session_state.t_sim = 0.0

# =================== Simulation state ===================
if "playing" not in st.session_state: st.session_state.playing = True
if "t_sim" not in st.session_state: st.session_state.t_sim = 0.0
if "last_tick" not in st.session_state: st.session_state.last_tick = time.time()
if "bearer" not in st.session_state: st.session_state.bearer = "5G"
if "bearer_prev" not in st.session_state: st.session_state.bearer_prev = "5G"
if "bearer_ttt" not in st.session_state: st.session_state.bearer_ttt = 0.0

# Build route once (one sample per sim-second) — with guard
SECS = max(2, int(sim_minutes_total * 60))
if ("route_df" not in st.session_state) or (st.session_state.get("route_secs") != SECS):
    st.session_state.route_df = interpolate_polyline(RAIL_WAYPOINTS, SECS)
    st.session_state.seg_labels = label_segments(SECS)
    st.session_state.route_secs = SECS

route_df = st.session_state.route_df
seg_labels = st.session_state.seg_labels

# Animation tick
if st.session_state.playing and anim_fps>0:
    st_autorefresh(interval=int(1000/anim_fps), key="tick", limit=None)
now = time.time()
elapsed = now - st.session_state.last_tick
if st.session_state.playing and anim_fps>0:
    st.session_state.t_sim += elapsed
st.session_state.last_tick = now
if st.session_state.t_sim > sim_minutes_total*60:
    st.session_state.t_sim = 0.0

# =================== Train positions (from route_df) ===================
# Convert speed to index advance per second relative to route length (uniform sampling).
# We already sampled evenly in arc-length: 1 index ≈ equal meters. We'll move 1 idx/sec at 1x baseline.
# To reflect train_speed_kmh, scale step factor (baseline 140 km/h = 1 idx/sec).
baseline_kmh = 140.0
speed_factor = max(0.1, float(train_speed_kmh) / baseline_kmh)
t_idx = int(st.session_state.t_sim * speed_factor) % len(route_df)

trainA = (float(route_df.lat.iloc[t_idx]), float(route_df.lon.iloc[t_idx]))
t_idx_nb = (len(route_df) - t_idx) % len(route_df)
trainB = (float(route_df.lat.iloc[t_idx_nb]), float(route_df.lon.iloc[t_idx_nb]))
segment_name = seg_labels[int(st.session_state.t_sim) % len(seg_labels)]

# =================== PHY model (SNR→PER, handover) ===================
def env_class(lat, lon):
    # city proximity → UMa else RMa
    cities = [(62.3913,17.3063),(60.6749,17.1413),(59.8586,17.6389),(59.3293,18.0686)]
    for c in cities:
        if haversine_m(lat,lon,c[0],c[1]) < 15000: return "UMa"
    return "RMa"

def pathloss_db(freq_GHz, d_m, env):
    if d_m < 1: d_m = 1
    fspl = 32.4 + 20*np.log10(freq_GHz*1000) + 20*np.log10(d_m/1000)
    extra = 7 if env=="UMa" else 3
    return fspl + extra

class ShadowingTrack:
    def __init__(self, sigma_db=7, decor_m=100, seed=7):
        self.sigma, self.decor = sigma_db, decor_m
        self.rng = np.random.default_rng(seed)
        self.last_s, self.curr = 0.0, 0.0
    def sample(self, s_m):
        delta = abs(s_m - self.last_s)
        rho = np.exp(-delta/self.decor)
        self.curr = rho*self.curr + math.sqrt(max(1e-9,1-rho**2))*self.rng.normal(0,self.sigma)
        self.last_s = s_m
        return self.curr

shadow = ShadowingTrack()

def rician_db(K_dB=8):
    K = 10**(K_dB/10)
    h = math.sqrt(K/(K+1)) + (np.random.normal(0,1/np.sqrt(2))+1j*np.random.normal(0,1/np.sqrt(2)))
    p = (abs(h)**2)/(K+1)
    return 10*np.log10(max(p, 1e-6))

def rayleigh_db():
    h = np.random.normal(0,1/np.sqrt(2))+1j*np.random.normal(0,1/np.sqrt(2))
    p = abs(h)**2
    return 10*np.log10(max(p, 1e-6))

def noise_dbm(bw_hz):
    return -174 + 10*np.log10(bw_hz) + 5

TECH = {
    "5G":  dict(freq=3.5,  bw=5e6,   base_lat=20,  snr_ok=3,  snr_hold=1),
    "LTE": dict(freq=1.8,  bw=3e6,   base_lat=35,  snr_ok=0,  snr_hold=-2),
    "3G":  dict(freq=2.1,  bw=1.5e6, base_lat=60,  snr_ok=-2, snr_hold=-4),
    "GSMR":dict(freq=0.9,  bw=200e3, base_lat=120, snr_ok=-4, snr_hold=-6),
}
P_TX = 43  # dBm

def serving_bs(lat, lon):
    # nearest site from hard-coded grid; assume all techs available
    dists = [haversine_m(lat, lon, b[1], b[2]) for b in BASE_STATIONS]
    i = int(np.argmin(dists))
    caps = {"5G","LTE","3G","GSMR"}
    return dict(name=BASE_STATIONS[i][0], lat=BASE_STATIONS[i][1], lon=BASE_STATIONS[i][2], tech=caps), dists[i]

def per_from_snr(snr_db):
    x0, k = 2.0, -1.1
    per = 1/(1+math.exp(k*(snr_db-x0)))
    return max(1e-5, min(0.99, per))

def pick_bearer(snr_table, caps, curr_bearer):
    order = ["5G","LTE","3G","GSMR"]
    avail = [b for b in order if b in caps]
    for b in avail:
        if snr_table.get(b,-99) >= TECH[b]["snr_ok"]:
            return b, True
    if avail:
        best = max(avail, key=lambda x: snr_table.get(x,-99))
        return best, True
    return curr_bearer, False

# SNR table at Train A
bsA, dA = serving_bs(*trainA)
envA = env_class(*trainA)
s_along = float(route_df.s_m.iloc[t_idx])  # correlated shadowing by arc-length
snr_table = {}
for b in ["5G","LTE","3G","GSMR"]:
    if b in bsA["tech"]:
        pl = pathloss_db(TECH[b]["freq"], dA, envA)
        sh = shadow.sample(s_along)
        fad = rician_db(8) if envA=="RMa" else rayleigh_db()
        rx = P_TX - pl + sh + fad
        snr_table[b] = rx - noise_dbm(TECH[b]["bw"])

cand, valid = pick_bearer(snr_table, bsA["tech"], st.session_state.bearer)
TTT_MS = 1200
if valid and cand != st.session_state.bearer:
    st.session_state.bearer_ttt += elapsed*1000
    if st.session_state.bearer_ttt >= TTT_MS:
        st.session_state.bearer_prev = st.session_state.bearer
        st.session_state.bearer = cand
        st.session_state.bearer_ttt = 0.0
else:
    st.session_state.bearer_ttt = 0.0
bearer = st.session_state.bearer
snr_use = snr_table.get(bearer, -20)
per_single = per_from_snr(snr_use)
per_reps = per_single**laneA_reps
lat_ms_phy = TECH[bearer]["base_lat"]

# =================== Macro capacity/loss layer ===================
bs_macro_name, bs_macro_dist, quality = nearest_bs_quality(*trainA)
cap_bps, rand_loss = cap_loss(quality, int(st.session_state.t_sim))
badge = {"GOOD":"🟢","PATCHY":"🟠","POOR":"🔴"}[quality]

# =================== Traffic & semantics ===================
RAW_HZ, HYB_HZ = 2.0, 0.2
BYTES_RAW, BYTES_ALERT, BYTES_SUM = 24, 280, 180

def synth_sensors(t_sec):
    base = 24 + 10*math.sin(2*math.pi*((t_sec/60)%1440)/1440)
    data = []
    # 22 sensors placed along the route
    N_SENS = 22
    idxs = np.linspace(0, len(route_df)-1, N_SENS).astype(int)
    for i, j in enumerate(idxs):
        la, lo = float(route_df.lat.iloc[j]), float(route_df.lon.iloc[j])
        d_risk = haversine_m(la, lo, 60.6749, 17.1413)  # hotspot near Gävle
        boost = max(0, 1 - d_risk/15000) * 14
        temp = base + np.random.normal(0,0.6) + boost
        strain = max(0.0, (temp-35)*0.8 + np.random.normal(0,0.5))
        ballast = max(0.0, np.random.normal(0.3,0.1) + 0.02*boost)
        exceeded = []
        if temp >= 38: exceeded.append("temp>38")
        if strain >= 10: exceeded.append("strain>10")
        score = min(1.0, 0.01*(temp-30)**2 + 0.04*max(0, strain-8))
        label = "high" if score>0.75 else ("medium" if score>0.4 else "low")
        data.append(dict(id=f"S{i:02d}", lat=la, lon=lo,
                         rail_temp_C=round(temp,1), strain_kN=round(strain,1),
                         ballast_idx=round(ballast,2), risk_score=round(score,2),
                         risk_label=label, exceeded=exceeded))
    return pd.DataFrame(data)

sdf = synth_sensors(st.session_state.t_sim)

raw_points = 0
laneA_alerts, laneB_msgs = [], []
for _, row in sdf.iterrows():
    if mode in ("RAW","HYBRID"):
        raw_points += int(RAW_HZ if mode=="RAW" else HYB_HZ)
    if row["risk_label"] in ("medium","high") and (("temp>38" in row["exceeded"]) or ("strain>10" in row["exceeded"])):
        laneA_alerts.append(dict(
            event="buckling_risk", sensor=row["id"],
            location=dict(lat=row["lat"], lon=row["lon"]),
            severity=row["risk_label"],
            confidence=round(0.6+0.4*row["risk_score"],2),
            evidence=dict(rail_temp_C=row["rail_temp_C"], strain_kN=row["strain_kN"],
                          ballast_idx=row["ballast_idx"], exceeded=row["exceeded"]),
            recommendation=dict(tsr_kmh=60), ttl_s=900
        ))
if mode in ("SEMANTIC","HYBRID"):
    laneB_msgs.append(dict(type="maintenance_summary",
                           ballast_hotspots=int((sdf.ballast_idx>0.6).sum()),
                           window=f"t={int(st.session_state.t_sim)}s"))

raw_bps = raw_points*BYTES_RAW
laneA_bps = len(laneA_alerts)*BYTES_ALERT
laneB_bps = len(laneB_msgs)*BYTES_SUM
bps_total = raw_bps + laneA_bps + laneB_bps

# =================== Fuse PHY + Macro into E2E ===================
# Base latency from PHY bearer + tiny load jitter
lat_ms = lat_ms_phy + (bps_total/1000)

# Capacity pressure → queueing penalty (nonlinear clamp) and best-effort drops
if bps_total > cap_bps:
    overload = bps_total / cap_bps
    lat_ms *= min(4.0, 1.0 + 0.35*(overload-1))  # extra queueing
    overflow_drop = min(0.5, 0.10*(overload-1))
else:
    overflow_drop = 0.0

# Random loss applies to best-effort streams (RAW, Lane B); Lane A uses repetition on PHY
loss_be = max(0.0, min(0.8, rand_loss + overflow_drop))
effective_raw_bps   = int(raw_bps   * (1.0 - loss_be))
effective_laneB_bps = int(laneB_bps * (1.0 - loss_be))
effective_total_bps = effective_raw_bps + laneA_bps + effective_laneB_bps

# Lane-A end-to-end success probability (from PHY PER & repetition)
laneA_success = (1 - per_single) ** laneA_reps

# =================== UI: Map (Deck.gl, no tiles) ===================
col1, col2 = st.columns([2.2, 1.8])

with col1:
    st.subheader("Live Map • Deck.gl (no basemap dependency)")

    view_state = pdk.ViewState(latitude=60.1, longitude=17.7, zoom=6)

    # Route Path: draw directly from waypoints (no unit mismatch)
    route_path = [[lon, lat] for (lat, lon) in RAIL_WAYPOINTS]
    route_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": route_path}],
        get_path="path",
        get_color=[0,102,255,255],
        width_scale=1,
        width_min_pixels=4,
    )

    # Risk circle around Gävle
    def circle_polygon(center, radius_m, n=100):
        lat0, lon0 = center
        m2deg_lat = 1/111111.0
        m2deg_lon = 1/(111111.0*math.cos(math.radians(lat0)))
        return [[lon0 + (radius_m*math.cos(th))*m2deg_lon,
                 lat0 + (radius_m*math.sin(th))*m2deg_lat] for th in np.linspace(0, 2*math.pi, n)]
    risk_layer = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": circle_polygon((60.6749, 17.1413), 15000)}],
        get_polygon="polygon",
        get_fill_color=[255,80,80,70],
        get_line_color=[255,80,80],
        line_width_min_pixels=1,
    )

    # Sensors
    colors = {"low":[0,170,0,230], "medium":[255,160,0,230], "high":[230,0,0,255]}
    sens_df = pd.DataFrame({"lon": sdf["lon"], "lat": sdf["lat"],
                            "color": sdf["risk_label"].map(lambda x: colors[x])})
    sensors_layer = pdk.Layer(
        "ScatterplotLayer",
        data=sens_df,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius=3000,
    )

    # Base Stations
    bs_df = pd.DataFrame([{"lon": b[2], "lat": b[1], "name": b[0]} for b in BASE_STATIONS])
    bs_layer = pdk.Layer(
        "ScatterplotLayer",
        data=bs_df,
        get_position='[lon, lat]',
        get_fill_color=[30,144,255,255],
        get_radius=4000,
    )

    # Trains (white dot + icon atlas)
    trains_df = pd.DataFrame([{"lon": trainA[1], "lat": trainA[0], "name": "Train A"},
                              {"lon": trainB[1], "lat": trainB[0], "name": "Train B"}])
    train_dot = pdk.Layer(
        "ScatterplotLayer",
        data=trains_df,
        get_position='[lon, lat]',
        get_fill_color=[255,255,255,255],
        get_radius=4500,
    )
    icon_atlas = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png"
    icon_mapping = {"marker": {"x": 0, "y": 0, "width": 128, "height": 128, "anchorY": 128}}
    trains_df["icon"] = "marker"
    train_icon = pdk.Layer(
        "IconLayer",
        data=trains_df,
        get_icon="icon",
        icon_atlas=icon_atlas,
        icon_mapping=icon_mapping,
        get_position='[lon, lat]',
        size_scale=12,
        get_size=4,
    )

    deck = pdk.Deck(
        initial_view_state=view_state,
        map_provider=None,  # <- no tiles required
        map_style=None,
        layers=[route_layer, risk_layer, sensors_layer, bs_layer, train_dot, train_icon],
    )
    st.pydeck_chart(deck, use_container_width=True, height=540)
    st.caption("Route (blue), risk (red), sensors (green/orange/red), BS (blue), trains (white + icon). No map tiles needed.")

with col2:
    st.subheader("Comms, Channel & Safety Status")
    st.markdown(
        f"**Segment:** {segment_name}  \n"
        f"**Serving BS (PHY):** {bsA['name']} • env **{envA}** • dist **{int(dA/1000)} km**  \n"
        f"**Bearer:** {bearer} (prev: {st.session_state.bearer_prev}, TTT: {int(st.session_state.bearer_ttt)} ms)  \n"
        f"**SNR:** {snr_use:.1f} dB | **PER(single):** {per_single:.2%} | **PER(x{laneA_reps}):** {per_reps:.2%} | **PHY base latency:** ~{int(lat_ms_phy)} ms  \n"
        f"**Macro BS:** {bs_macro_name} • dist **{int(bs_macro_dist/1000)} km** • quality {badge} **{quality}**  \n"
        f"**Capacity budget:** ~{cap_bps/1000:.0f} kbps | **Random loss (best-effort):** {rand_loss*100:.1f}%"
    )
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RAW (bps)", f"{raw_bps:,.0f}")
    c2.metric("Lane A (bps)", f"{laneA_bps:,.0f}")
    c3.metric("Lane B (bps)", f"{laneB_bps:,.0f}")
    c4.metric("Total (bps)", f"{bps_total:,.0f}")
    st.markdown(
        f"**Effective best-effort (after loss):** {effective_raw_bps + effective_laneB_bps:,.0f} bps  \n"
        f"**E2E Lane-A success prob (PHY):** {(laneA_success*100):.1f}%  \n"
        f"**E2E latency (PHY + capacity):** ~{int(lat_ms)} ms"
    )
    st.markdown("---")
    st.markdown("**Lane A (Safety Alerts)**")
    st.json(laneA_alerts[:6] if laneA_alerts else [])
    st.markdown("**Lane B (Ops/Maintenance)**")
    st.json(laneB_msgs if laneB_msgs else [])

# =================== Sankey Communication Flow ===================
if show_sankey:
    st.markdown("---")
    st.subheader("Communication Flow (This Tick)")
    sensors_to_bs = max(1, raw_bps + laneA_bps + laneB_bps)
    bs_to_net = sensors_to_bs
    net_to_tms = laneA_bps + laneB_bps + (raw_bps if mode!="SEMANTIC" else 0)
    tms_to_train = max(1, len(laneA_alerts)*100)
    tms_to_maint = max(1, len(laneB_msgs)*100)

    nodes = ["Sensors","BS/Edge",f"Network ({bearer})","TMS","Train DAS","Maintenance"]
    idx = {n:i for i,n in enumerate(nodes)}
    sankey = go.Figure(data=[go.Sankey(
        node=dict(label=nodes),
        link=dict(
            source=[idx["Sensors"], idx["BS/Edge"], idx[f"Network ({bearer})"], idx["TMS"], idx["TMS"]],
            target=[idx["BS/Edge"], idx[f"Network ({bearer})"], idx["TMS"], idx["Train DAS"], idx["Maintenance"]],
            value=[sensors_to_bs, bs_to_net, net_to_tms, tms_to_train, tms_to_maint],
            label=["telemetry/alerts","uplink","ctrl+data","advisories","work orders"],
        )
    )])
    sankey.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(sankey, use_container_width=True, config={"displayModeBar": False})

# =================== Footnote ===================
st.caption(
    "Hybrid model: PHY (FSPL + shadowing + Rayleigh/Rician) → SNR→PER, with handover & repetition for Lane-A; "
    "Macro capacity (GOOD/PATCHY/POOR) gates throughput with random loss & queueing under overload; "
    "Lane-A never best-effort-dropped (relies on repetition), while RAW/Lane-B absorb loss."
)
