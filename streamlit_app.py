# -------------------------------------------------------------
# ENSURE-6G • TMS Dashboard — Raw vs Semantic vs Hybrid (Sundsvall→Stockholm)
# Plotly OSM • Channel model • Dynamic bearer handover (5G→LTE→3G→GSM-R)
# -------------------------------------------------------------

import math, time, random
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString

import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ENSURE-6G • TMS Rail Demo", layout="wide")
st.title("🚆 ENSURE-6G: Raw vs Semantic vs Hybrid — Control Center (TMS) Dashboard")
st.caption("Sundsvall → Stockholm • Sensors + Base Stations • Lane A safety alerts vs Lane B ops • Channel model • Dynamic bearer (5G/LTE/3G/GSM-R)")

# ---------- Helpers ----------
def haversine(p1, p2):
    R = 6371000.0
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * (2*math.atan2(math.sqrt(a), math.sqrt(1-a)))

def db_add(a_db, b_db):  # power domain sum
    return 10*np.log10(10**(a_db/10)+10**(b_db/10))

# ---------- Route ----------
ROUTE = [
    (62.3913, 17.3063),  # Sundsvall
    (61.3039, 17.0600),  # Hudiksvall vicinity
    (60.6749, 17.1413),  # Gävle
    (59.8586, 17.6389),  # Uppsala
    (59.3293, 18.0686),  # Stockholm
]
route_ls = LineString([(p[1], p[0]) for p in ROUTE])  # (lon,lat)
seg_len = [haversine(ROUTE[i], ROUTE[i+1]) for i in range(len(ROUTE)-1)]
route_len = float(sum(seg_len))

# ---------- Sensors & BS ----------
N_SENS = 20
def interp_point_by_dist(dist_m):
    if dist_m <= 0: return ROUTE[0]
    if dist_m >= route_len: return ROUTE[-1]
    rem = dist_m
    for i, L in enumerate(seg_len):
        if rem <= L:
            f = rem / L
            lat = ROUTE[i][0] + f*(ROUTE[i+1][0]-ROUTE[i][0])
            lon = ROUTE[i][1] + f*(ROUTE[i+1][1]-ROUTE[i][1])
            return (lat, lon)
        rem -= L
    return ROUTE[-1]

sensor_pts = []
for i in range(N_SENS):
    frac = i/(N_SENS-1)
    # bias density near Gävle
    if 0.35 < frac < 0.55:
        frac = 0.45 + (frac-0.45)*0.6
    d = frac*route_len
    sensor_pts.append(interp_point_by_dist(d))

# BS with tech capabilities (illustrative)
BS = [
    dict(name="Sundsvall",   lat=62.3913, lon=17.3063, tech={"5G","LTE","3G","GSMR"}),
    dict(name="Mid-North",   lat=61.80,   lon=17.10,   tech={"5G","LTE","3G","GSMR"}),
    dict(name="Gävle",       lat=60.6749, lon=17.1413, tech={"5G","LTE","3G","GSMR"}),
    dict(name="Mid-South",   lat=60.25,   lon=17.40,   tech={"LTE","3G","GSMR"}),  # no 5G here
    dict(name="Uppsala",     lat=59.8586, lon=17.6389, tech={"5G","LTE","3G","GSMR"}),
    dict(name="Near-Stock",  lat=59.55,   lon=17.85,   tech={"5G","LTE","3G","GSMR"}),
    dict(name="Stockholm",   lat=59.3293, lon=18.0686, tech={"5G","LTE","3G","GSMR"}),
]

# Risk zone near Gävle
RISK_CENTER = (60.6749, 17.1413)
RISK_RADIUS_M = 15000

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Scenario Controls (TMS)")
    mode = st.radio("Communication Mode", ["RAW","SEMANTIC","HYBRID"], index=2)
    train_speed_kmh = st.slider("Train Speed (km/h)", 60, 200, 140, 10)
    anim_speed = st.slider("Animation Speed (steps/sec)", 0, 10, 2, 1,
                           help="Controls UI refresh rate. Train speed is physical.")
    sim_minutes_total = st.number_input("Sim Length (minutes)", 5, 180, 30, 5)
    st.markdown("---")
    st.subheader("Lane A QoS")
    laneA_target_ms = st.slider("Target latency (ms)", 20, 500, 100, 10)
    laneA_reps = st.slider("Repetitions (reliability)", 1, 3, 2, 1)
    st.markdown("---")
    st.subheader("Display")
    show_sens = st.checkbox("Show Sensor Nodes", True)
    show_bs   = st.checkbox("Show Base Stations", True)
    show_sankey = st.checkbox("Show Communication Flow (Sankey)", True)

# ---------- Animation State ----------
if "playing" not in st.session_state: st.session_state.playing = True
if "t_sim" not in st.session_state: st.session_state.t_sim = 0.0  # seconds
if "last_tick" not in st.session_state: st.session_state.last_tick = time.time()
if "bearer" not in st.session_state: st.session_state.bearer = "5G"
if "bearer_prev" not in st.session_state: st.session_state.bearer_prev = "5G"
if "bearer_ttt" not in st.session_state: st.session_state.bearer_ttt = 0.0  # time-to-trigger

colpp1, colpp2 = st.sidebar.columns(2)
if colpp1.button("⏯ Play/Pause"):
    st.session_state.playing = not st.session_state.playing
if colpp2.button("⏮ Reset"):
    st.session_state.t_sim = 0.0

# Use autorefresh for animation
if st.session_state.playing and anim_speed > 0:
    st_autorefresh(interval=max(100, int(1000/anim_speed)), limit=None, key="tick")

# Advance simulated time by real elapsed (scaled by anim_speed)
now = time.time()
elapsed = now - st.session_state.last_tick
if st.session_state.playing and anim_speed>0:
    st.session_state.t_sim += elapsed * anim_speed
st.session_state.last_tick = now

# clamp duration
if st.session_state.t_sim > sim_minutes_total*60:
    st.session_state.t_sim = 0.0

# ---------- Train positions ----------
v_mps = train_speed_kmh/3.6
dist_travelled = (v_mps * st.session_state.t_sim) % route_len
pt = route_ls.interpolate(dist_travelled)
trainA = (pt.y, pt.x)  # (lat,lon)

opp_dist = (route_len - dist_travelled) % route_len
pt2 = route_ls.interpolate(opp_dist)
trainB = (pt2.y, pt2.x)

# ---------- Environment & Channel ----------
def env_class(lat, lon):
    # near cities → UMa, else RMa
    cities = [(p[0],p[1]) for p in [ROUTE[0], ROUTE[2], ROUTE[3], ROUTE[4]]]
    for c in cities:
        if haversine((lat,lon), c) < 15000:  # 15 km
            return "UMa"
    return "RMa"

def pathloss_db(freq_GHz, d_m, env):
    # Simple 3GPP-ish proxy: FSPL + env offset
    if d_m < 1: d_m = 1
    fspl = 32.4 + 20*np.log10(freq_GHz*1000) + 20*np.log10(d_m/1000)  # freq in MHz, dist in km
    extra = 7 if env=="UMa" else 3
    return fspl + extra

class ShadowingTrack:
    def __init__(self, sigma_db=7, decor_m=80, seed=7):
        self.sigma = sigma_db
        self.decor = decor_m
        self.rng = np.random.default_rng(seed)
        self.last_s = 0.0
        self.curr = 0.0
    def sample(self, s_m):
        # exponential correlation
        delta = abs(s_m - self.last_s)
        rho = np.exp(-delta/self.decor)
        self.curr = rho*self.curr + math.sqrt(1-rho**2)*self.rng.normal(0,self.sigma)
        self.last_s = s_m
        return self.curr

shadow = ShadowingTrack(sigma_db=7, decor_m=100)

def rician_fading_db(K_dB=8):
    # Rician power gain; return dB
    K = 10**(K_dB/10)
    # LOS + scattered (Rayleigh with unit power)
    s = math.sqrt(K/(K+1)) + np.random.normal(0, 1/math.sqrt(2)) + 1j*np.random.normal(0,1/math.sqrt(2))
    power = (abs(s)**2)/(K+1)
    return 10*np.log10(max(power, 1e-6))

def rayleigh_fading_db():
    h = np.random.normal(0, 1/math.sqrt(2)) + 1j*np.random.normal(0,1/math.sqrt(2))
    p = abs(h)**2
    return 10*np.log10(max(p, 1e-6))

def noise_floor_dbm(bw_hz=200e3):  # per-packet rough
    kT = -174  # dBm/Hz
    nf = 5     # dB
    return kT + 10*np.log10(bw_hz) + nf

TECH = {
    "5G":  dict(freq=3.5,  bw=5e6,  base_lat=20,  snr_ok=3,  snr_hold=1),   # dB
    "LTE": dict(freq=1.8,  bw=3e6,  base_lat=35,  snr_ok=0,  snr_hold=-2),
    "3G":  dict(freq=2.1,  bw=1.5e6,base_lat=60,  snr_ok=-2, snr_hold=-4),
    "GSMR":dict(freq=0.9,  bw=200e3,base_lat=120, snr_ok=-4, snr_hold=-6),
}
P_TX_dBm = 43  # eNodeB/gNodeB-ish

def serving_bs(lat, lon):
    # nearest site
    dists = [haversine((lat,lon), (b["lat"],b["lon"])) for b in BS]
    i = int(np.argmin(dists))
    return BS[i], dists[i]

def snr_for(lat, lon, tech, env, dist_to_bs_m, s_along_m):
    freq = TECH[tech]["freq"]
    bw = TECH[tech]["bw"]
    pl = pathloss_db(freq, dist_to_bs_m, env)
    sh = shadow.sample(s_along_m)
    fad = rician_fading_db(8) if env=="RMa" else rayleigh_fading_db()
    rx = P_TX_dBm - pl + sh + fad
    n = noise_floor_dbm(bw)
    return rx - n, rx, n

def per_from_snr(snr_db):
    # toy URLLC-ish logistic PER curve
    x0, k = 2.0, -1.1  # threshold 2 dB for 10% PER-ish
    per = 1/(1+math.exp(k*(snr_db-x0)))
    return max(1e-5, min(0.99, per))

# ---------- Simulation step (sensors → alerts/messages) ----------
def synth_sensors(t_sec):
    # diurnal + risk zone heat
    base = 24 + 10*math.sin(2*math.pi*((t_sec/60)%1440)/1440)
    data = []
    for i, (la,lo) in enumerate(sensor_pts):
        d_risk = haversine((la,lo), RISK_CENTER)
        boost = max(0, 1 - d_risk/RISK_RADIUS_M) * 14
        temp = base + np.random.normal(0,0.6) + boost
        strain = max(0.0, (temp-35)*0.8 + np.random.normal(0,0.5))
        ballast = max(0.0, np.random.normal(0.3,0.1) + 0.02*boost)
        exceeded = []
        if temp >= 38: exceeded.append("temp>38")
        if strain >= 10: exceeded.append("strain>10")
        score = min(1.0, 0.01*(temp-30)**2 + 0.04*max(0, strain-8))
        label = "high" if score>0.75 else ("medium" if score>0.4 else "low")
        data.append(dict(id=f"S{i:02d}", lat=la, lon=lo,
                         rail_temp_C=round(temp,1),
                         strain_kN=round(strain,1),
                         ballast_idx=round(ballast,2),
                         risk_score=round(score,2),
                         risk_label=label,
                         exceeded=exceeded))
    return pd.DataFrame(data)

sdf = synth_sensors(st.session_state.t_sim)

# ---------- Bearer selection w/ hysteresis & time-to-trigger ----------
def pick_bearer(lat, lon, snr_table, bs_caps, curr_bearer, dt):
    # prefer 5G→LTE→3G→GSMR if available and SNR >= snr_ok
    order = ["5G","LTE","3G","GSMR"]
    avail = [b for b in order if b in bs_caps]
    # candidate is best with snr >= ok
    for b in avail:
        if snr_table[b] >= TECH[b]["snr_ok"]:
            cand = b; break
    else:
        # fallback: best SNR among avail
        cand = max(avail, key=lambda x: snr_table[x])

    # Hysteresis: if current still >= hold, keep unless cand is two tiers higher
    if curr_bearer in avail and snr_table[curr_bearer] >= TECH[curr_bearer]["snr_hold"]:
        # require time-to-trigger to switch
        return curr_bearer, False
    # else we plan to switch after TTT
    return cand, True

# distance along track for shadowing correlation
def distance_along(train_latlon):
    # rough inverse: search closest point fraction by sampling
    samples = 300
    dists = []
    for i in range(samples+1):
        f = i/samples
        d = f*route_len
        p = route_ls.interpolate(d)
        dists.append(haversine((p.y,p.x), train_latlon))
    i = int(np.argmin(dists))
    return (i/samples)*route_len

# Compute SNR per tech for Train A at its serving BS
bsA, dA = serving_bs(*trainA)
envA = env_class(*trainA)
s_along = distance_along(trainA)

snr_table = {}
rx_table = {}
n_table = {}
for b in ["5G","LTE","3G","GSMR"]:
    if b in bsA["tech"]:
        snr, rx, n = snr_for(trainA[0], trainA[1], b, envA, dA, s_along)
        snr_table[b] = snr
        rx_table[b] = rx
        n_table[b] = n

cand, want_switch = pick_bearer(*trainA, snr_table=snr_table, bs_caps=bsA["tech"],
                                curr_bearer=st.session_state.bearer, dt=elapsed)

# Time-to-trigger (ms)
TTT_MS = 1200
if want_switch and cand != st.session_state.bearer:
    st.session_state.bearer_ttt += elapsed*1000
    if st.session_state.bearer_ttt >= TTT_MS:
        st.session_state.bearer_prev = st.session_state.bearer
        st.session_state.bearer = cand
        st.session_state.bearer_ttt = 0.0
else:
    st.session_state.bearer_ttt = 0.0

bearer = st.session_state.bearer

# ---------- Messaging (RAW / SEMANTIC / HYBRID) ----------
RAW_HZ = 2.0
HYB_HZ = 0.2
BYTES_RAW = 24
BYTES_ALERT = 280
BYTES_SUM = 180

raw_points = 0
laneA_alerts = []
laneB_msgs = []

for _, row in sdf.iterrows():
    if mode in ("RAW","HYBRID"):
        rate = RAW_HZ if mode=="RAW" else HYB_HZ
        raw_points += int(rate)

    if row["risk_label"] in ("medium","high") and (("temp>38" in row["exceeded"]) or ("strain>10" in row["exceeded"])):
        laneA_alerts.append(dict(
            event="buckling_risk", sensor=row["id"],
            location=dict(lat=row["lat"], lon=row["lon"]),
            severity=row["risk_label"],
            confidence=round(0.6+0.4*row["risk_score"],2),
            evidence=dict(rail_temp_C=row["rail_temp_C"], strain_kN=row["strain_kN"], ballast_idx=row["ballast_idx"],
                          exceeded=row["exceeded"]),
            recommendation=dict(tsr_kmh=60), ttl_s=900
        ))

if mode in ("SEMANTIC","HYBRID"):
    laneB_msgs.append(dict(
        type="maintenance_summary",
        ballast_hotspots=int((sdf.ballast_idx>0.6).sum()),
        window=f"t={int(st.session_state.t_sim)}s"
    ))

raw_bps = raw_points*BYTES_RAW
alert_bps = len(laneA_alerts)*BYTES_ALERT
laneB_bps = len(laneB_msgs)*BYTES_SUM
bps_total = raw_bps + alert_bps + laneB_bps

# Lane A reliability & latency on selected bearer
snr_use = snr_table.get(bearer, -20)
per_single = per_from_snr(snr_use)
per_reps = per_single**laneA_reps
lat_ms = TECH[bearer]["base_lat"] + (bps_total/1000)  # trivial jitter vs load

# ---------- UI: Map ----------
col1, col2 = st.columns([2.2, 1.8])

with col1:
    st.subheader("Live Map • Plotly OSM (TMS view)")
    mid = ROUTE[len(ROUTE)//2]
    fig = go.Figure()

    # route
    fig.add_trace(go.Scattermapbox(
        lat=[p[0] for p in ROUTE], lon=[p[1] for p in ROUTE],
        mode="lines", line=dict(width=4), name="Route"))

    # risk zone polygon
    lat0, lon0 = RISK_CENTER
    m2deg_lat = 1/111111.0
    m2deg_lon = 1/(111111.0*math.cos(math.radians(lat0)))
    theta = np.linspace(0, 2*math.pi, 120)
    circ_lat = lat0 + (RISK_RADIUS_M*np.sin(theta))*m2deg_lat
    circ_lon = lon0 + (RISK_RADIUS_M*np.cos(theta))*m2deg_lon
    fig.add_trace(go.Scattermapbox(lat=circ_lat, lon=circ_lon, mode="lines",
                                   line=dict(width=2), name="Risk zone", fill="toself", opacity=0.15))

    # sensors
    if show_sens:
        colors = dict(low="green", medium="orange", high="red")
        fig.add_trace(go.Scattermapbox(
            lat=sdf["lat"], lon=sdf["lon"], mode="markers",
            marker=dict(size=8, color=[colors[r] for r in sdf["risk_label"]]),
            text=[f"{r.id} • {r.rail_temp_C}°C • {r.strain_kN}kN • {r.risk_label}" for r in sdf.itertuples()],
            hoverinfo="text", name="Sensors"))

    # base stations
    if show_bs:
        fig.add_trace(go.Scattermapbox(
            lat=[b["lat"] for b in BS], lon=[b["lon"] for b in BS], mode="markers",
            marker=dict(size=11, symbol="triangle"), name="Base Stations",
            text=[f"{b['name']} ({'/'.join(sorted(b['tech']))})" for b in BS], hoverinfo="text"))

    # trains
    fig.add_trace(go.Scattermapbox(lat=[trainA[0]], lon=[trainA[1]], mode="text",
                                   text=["🚆"], textfont=dict(size=24), name="Train A (southbound)"))
    fig.add_trace(go.Scattermapbox(lat=[trainB[0]], lon=[trainB[1]], mode="text",
                                   text=["🚆"], textfont=dict(size=24), name="Train B (northbound)"))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=mid[0], lon=mid[1]), zoom=6),
        margin=dict(l=0,r=0,t=0,b=0), height=520,
        legend=dict(orientation="h", yanchor="bottom", y=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("TMS perspective: Train position uses *Train Speed*. Animation pace uses *Animation Speed* (independent).")

with col2:
    st.subheader("Comms, Channel & Safety Status")
    st.markdown(f"**Serving BS:** {bsA['name']} • env **{envA}** • dist **{int(dA/1000)} km**")
    st.markdown(f"**Bearer:** {bearer} (prev: {st.session_state.bearer_prev})  "
                f"| **SNR:** {snr_use:.1f} dB  | **PER(single):** {per_single:.2%}  | **PER(x{laneA_reps}):** {per_reps:.2%}  "
                f"| **Lane A latency:** ~{int(lat_ms)} ms")

    # Bandwidth
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RAW (bps)", f"{raw_bps:,.0f}")
    c2.metric("Lane A (bps)", f"{alert_bps:,.0f}")
    c3.metric("Lane B (bps)", f"{laneB_bps:,.0f}")
    c4.metric("Total (bps)", f"{bps_total:,.0f}")

    # Alerts
    st.markdown("---")
    st.markdown("**Lane A (Safety Alerts)**")
    if laneA_alerts: st.json(laneA_alerts[:6])
    else: st.caption("No active Lane A alerts in this tick.")

    st.markdown("**Lane B (Ops/Maintenance)**")
    if laneB_msgs: st.json(laneB_msgs)
    else: st.caption("No Lane B messages in RAW-only mode.")

# ---------- Sankey Communication Flow ----------
if show_sankey:
    st.markdown("---")
    st.subheader("Communication Flow (This Tick)")
    # Build simple flow weights proportional to bytes/messages
    sensors_to_bs = max(1, raw_bps + alert_bps + laneB_bps)
    bs_to_net = sensors_to_bs
    net_to_tms = alert_bps + laneB_bps + (raw_bps if mode!="SEMANTIC" else 0)
    tms_to_train = max(1, len(laneA_alerts)*100)  # abstracted control advisories
    tms_to_maint = max(1, len(laneB_msgs)*100)

    nodes = ["Sensors","BS/Edge","Network ("+bearer+")","TMS","Train DAS","Maintenance"]
    idx = {n:i for i,n in enumerate(nodes)}
    link = dict(
        source=[idx["Sensors"], idx["BS/Edge"], idx["Network ("+bearer+")"], idx["TMS"], idx["TMS"]],
        target=[idx["BS/Edge"], idx["Network ("+bearer+")"], idx["TMS"], idx["Train DAS"], idx["Maintenance"]],
        value=[sensors_to_bs, bs_to_net, net_to_tms, tms_to_train, tms_to_maint],
        label=["telemetry/alerts","uplink","ctrl+data","advisories","work orders"]
    )
    sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=nodes),
        link=link
    )])
    sankey.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(sankey, use_container_width=True)

# ---------- Footnotes ----------
st.caption(
    "Channel model (proxy): FSPL + env offset (UMa/RMa), log-normal shadowing (σ≈7 dB), Rician (RMa) / Rayleigh (UMa) fading. "
    "Bearer selection with hysteresis + time-to-trigger to avoid ping-pong; 5G→LTE→3G→GSM-R. "
    "Lane A reliability via repetition combining; latency = bearer base + load jitter."
)
