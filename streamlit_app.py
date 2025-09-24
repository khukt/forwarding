# -------------------------------------------------------------
# ENSURE-6G • TMS Dashboard — PyDeck (deck.gl) Smooth Animation
# Sundsvall → Stockholm • RAW vs SEMANTIC vs HYBRID
# Dynamic bearer (5G→LTE→3G→GSM-R), channel model, Sankey flow
# -------------------------------------------------------------
import math, time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from shapely.geometry import LineString
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ENSURE-6G • TMS Rail Demo (PyDeck)", layout="wide")
st.title("🚆 ENSURE-6G: Raw vs Semantic vs Hybrid — Control Center (TMS) Dashboard")
st.caption("Sundsvall → Stockholm • Sensors + Base Stations • Lane A safety alerts vs Lane B ops • Channel model • Dynamic bearer (5G/LTE/3G/GSM-R) • Smooth Deck.gl animation")

# ---------- Helpers ----------
def haversine(p1, p2):
    R = 6371000.0
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * (2*math.atan2(math.sqrt(a), math.sqrt(1-a)))

def per_from_snr(snr_db):
    x0, k = 2.0, -1.1  # threshold-ish
    per = 1/(1+math.exp(k*(snr_db-x0)))
    return max(1e-5, min(0.99, per))

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

N_SENS = 22
sensor_pts = []
for i in range(N_SENS):
    frac = i/(N_SENS-1)
    if 0.35 < frac < 0.55:
        frac = 0.45 + (frac-0.45)*0.6
    d = frac*route_len
    sensor_pts.append(interp_point_by_dist(d))

# NOTE: 'tech' is a set (not JSON-serializable) — we will NOT pass it to pydeck layers
BS = [
    dict(name="Sundsvall",   lat=62.3913, lon=17.3063, tech={"5G","LTE","3G","GSMR"}),
    dict(name="Mid-North",   lat=61.80,   lon=17.10,   tech={"5G","LTE","3G","GSMR"}),
    dict(name="Gävle",       lat=60.6749, lon=17.1413, tech={"5G","LTE","3G","GSMR"}),
    dict(name="Mid-South",   lat=60.25,   lon=17.40,   tech={"LTE","3G","GSMR"}),  # no 5G
    dict(name="Uppsala",     lat=59.8586, lon=17.6389, tech={"5G","LTE","3G","GSMR"}),
    dict(name="Near-Stock",  lat=59.55,   lon=17.85,   tech={"5G","LTE","3G","GSMR"}),
    dict(name="Stockholm",   lat=59.3293, lon=18.0686, tech={"5G","LTE","3G","GSMR"}),
]

RISK_CENTER = (60.6749, 17.1413)
RISK_RADIUS_M = 15000

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Scenario Controls (TMS)")
    mode = st.radio("Communication Mode", ["RAW","SEMANTIC","HYBRID"], index=2)
    train_speed_kmh = st.slider("Train Speed (km/h)", 60, 200, 140, 10)
    anim_fps = st.slider("Animation FPS", 1, 20, 6, 1, help="UI refresh rate (frames per second).")
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
    colpp1, colpp2 = st.columns(2)
    if colpp1.button("⏯ Play/Pause"): st.session_state.playing = not st.session_state.get("playing", True)
    if colpp2.button("⏮ Reset"): st.session_state.t_sim = 0.0

# ---------- Animation State ----------
if "playing" not in st.session_state: st.session_state.playing = True
if "t_sim" not in st.session_state: st.session_state.t_sim = 0.0
if "last_tick" not in st.session_state: st.session_state.last_tick = time.time()
if "bearer" not in st.session_state: st.session_state.bearer = "5G"
if "bearer_prev" not in st.session_state: st.session_state.bearer_prev = "5G"
if "bearer_ttt" not in st.session_state: st.session_state.bearer_ttt = 0.0  # ms accumulator

# Smooth rerun timer
if st.session_state.playing and anim_fps > 0:
    st_autorefresh(interval=int(1000/anim_fps), key="tick", limit=None)

# Advance simulated time
now = time.time()
elapsed = now - st.session_state.last_tick
if st.session_state.playing and anim_fps > 0:
    st.session_state.t_sim += elapsed
st.session_state.last_tick = now
if st.session_state.t_sim > sim_minutes_total*60:
    st.session_state.t_sim = 0.0

# ---------- Train positions ----------
def route_position(dist_m):
    pt = route_ls.interpolate(dist_m % route_len)
    return (pt.y, pt.x)

v_mps = train_speed_kmh/3.6
dist_travelled = (v_mps * st.session_state.t_sim) % route_len
trainA = route_position(dist_travelled)             # southbound
trainB = route_position(route_len - dist_travelled) # northbound

# ---------- Channel & Bearer ----------
def env_class(lat, lon):
    cities = [(p[0],p[1]) for p in [ROUTE[0], ROUTE[2], ROUTE[3], ROUTE[4]]]
    for c in cities:
        if haversine((lat,lon), c) < 15000: return "UMa"
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
    dists = [haversine((lat,lon), (b["lat"],b["lon"])) for b in BS]
    i = int(np.argmin(dists))
    return BS[i], dists[i]

def distance_along(latlon):
    # sample route to find closest arc-length for correlated shadowing
    samples = 300; best_i=0; best_d=1e12
    for i in range(samples+1):
        f = i/samples; d = f*route_len
        p = route_ls.interpolate(d)
        hd = haversine((p.y,p.x), latlon)
        if hd < best_d: best_d, best_i = hd, i
    return (best_i/samples)*route_len

def snr_for(lat, lon, tech, env, d_bs_m, s_along_m):
    pl = pathloss_db(TECH[tech]["freq"], d_bs_m, env)
    sh = shadow.sample(s_along_m)
    fad = rician_db(8) if env=="RMa" else rayleigh_db()
    rx = P_TX - pl + sh + fad
    n = noise_dbm(TECH[tech]["bw"])
    return rx - n, rx, n

# compute SNR table at Train A
bsA, dA = serving_bs(*trainA)
envA = env_class(*trainA)
s_along = distance_along(trainA)
snr_table = {}
for b in ["5G","LTE","3G","GSMR"]:
    if b in bsA["tech"]:
        snr_table[b] = snr_for(trainA[0], trainA[1], b, envA, dA, s_along)[0]

def pick_bearer(snr_table, caps, curr_bearer):
    order = ["5G","LTE","3G","GSMR"]
    avail = [b for b in order if b in caps]
    # prefer highest tier meeting snr_ok
    for b in avail:
        if snr_table.get(b,-99) >= TECH[b]["snr_ok"]:
            return b, True
    # fallback: best SNR
    if avail:
        best = max(avail, key=lambda x: snr_table.get(x,-99))
        return best, True
    return curr_bearer, False

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
lat_ms = TECH[bearer]["base_lat"]

# ---------- Sensors & Messages ----------
def synth_sensors(t_sec):
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
                         rail_temp_C=round(temp,1), strain_kN=round(strain,1),
                         ballast_idx=round(ballast,2), risk_score=round(score,2),
                         risk_label=label, exceeded=exceeded))
    return pd.DataFrame(data)

sdf = synth_sensors(st.session_state.t_sim)

RAW_HZ, HYB_HZ = 2.0, 0.2
BYTES_RAW, BYTES_ALERT, BYTES_SUM = 24, 280, 180

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
alert_bps = len(laneA_alerts)*BYTES_ALERT
laneB_bps = len(laneB_msgs)*BYTES_SUM
bps_total = raw_bps + alert_bps + laneB_bps
lat_ms += (bps_total/1000)  # tiny jitter

# ---------- UI: Map (PyDeck) ----------
col1, col2 = st.columns([2.2, 1.8])

with col1:
    st.subheader("Live Map • Deck.gl (TMS view)")

    # 1) Route PathLayer
    path_data = [{"path": [[p[1], p[0]] for p in ROUTE]}]
    route_layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_width=4,
        width_min_pixels=2,
        get_color=[0, 102, 255, 200],
    )

    # 2) Risk zone PolygonLayer (circle approximation)
    def circle_polygon(center, radius_m, n=80):
        lat0, lon0 = center
        m2deg_lat = 1/111111.0
        m2deg_lon = 1/(111111.0*math.cos(math.radians(lat0)))
        verts = []
        for th in np.linspace(0, 2*math.pi, n):
            lat = lat0 + (radius_m*math.sin(th))*m2deg_lat
            lon = lon0 + (radius_m*math.cos(th))*m2deg_lon
            verts.append([lon, lat])
        return verts
    risk_poly = [{"polygon": circle_polygon(RISK_CENTER, RISK_RADIUS_M)}]
    risk_layer = pdk.Layer(
        "PolygonLayer",
        data=risk_poly,
        get_polygon="polygon",
        get_fill_color=[255, 51, 51, 40],
        get_line_color=[255, 51, 51],
        line_width_min_pixels=1,
        pickable=False,
    )

    # 3) Sensors ScatterplotLayer colored by risk (only serializable columns)
    colors = {"low":[0,170,0,200], "medium":[255,140,0,200], "high":[220,0,0,220]}
    sens_df = pd.DataFrame({
        "lat": sdf["lat"], "lon": sdf["lon"],
        "color": sdf["risk_label"].map(lambda x: colors[x]),
        "tooltip": [f"{r.id} • {r.rail_temp_C}°C • {r.strain_kN}kN • {r.risk_label}"
                    for r in sdf.itertuples()]
    })
    sensor_layer = pdk.Layer(
        "ScatterplotLayer",
        data=sens_df if show_sens else sens_df.iloc[0:0],
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=1200,
        pickable=True,
    )

    # 4) Base Stations ScatterplotLayer (DROP non-serializable 'tech')
    bs_df = pd.DataFrame(BS)
    bs_df_layer = bs_df[["lat","lon","name"]].copy()
    bs_df_layer["tooltip"] = [f"{row['name']} ({'/'.join(sorted(BS[i]['tech']))})" for i, row in bs_df_layer.reset_index().iterrows()]
    bs_layer = pdk.Layer(
        "ScatterplotLayer",
        data=bs_df_layer if show_bs else bs_df_layer.iloc[0:0],
        get_position="[lon, lat]",
        get_fill_color=[30, 144, 255, 220],
        get_radius=1800,
        pickable=True,
    )

    # 5) Trains as TextLayer with emoji (🚆)
    trains_df = pd.DataFrame([
        dict(lat=trainA[0], lon=trainA[1], label="🚆"),
        dict(lat=trainB[0], lon=trainB[1], label="🚆"),
    ])
    train_layer = pdk.Layer(
        "TextLayer",
        trains_df,
        get_position='[lon, lat]',
        get_text='label',
        get_size=24,
        get_text_anchor='middle',
        get_alignment_baseline='center',
    )

    view_state = pdk.ViewState(latitude=60.1, longitude=17.7, zoom=6, bearing=0, pitch=0)
    deck = pdk.Deck(
        map_provider="carto", map_style="light",
        initial_view_state=view_state,
        layers=[route_layer, risk_layer, sensor_layer, bs_layer, train_layer],
        tooltip={"html":"<b>{tooltip}</b>", "style":{"color":"white"}}
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption("Deck.gl updates smoothly. Trains (🚆) via TextLayer; sensors & BS ScatterplotLayers; route PathLayer; risk zone PolygonLayer.")

with col2:
    st.subheader("Comms, Channel & Safety Status")
    # Work out serving BS again for display
    bsA, dA = serving_bs(*trainA)
    envA = env_class(*trainA)
    st.markdown(f"**Serving BS:** {bsA['name']} • env **{envA}** • dist **{int(haversine(trainA,(bsA['lat'],bsA['lon']))/1000)} km**")
    st.markdown(
        f"**Bearer:** {bearer} (prev: {st.session_state.bearer_prev}, TTT: {int(st.session_state.bearer_ttt)} ms)  | "
        f"**SNR:** {snr_use:.1f} dB  | **PER(single):** {per_single:.2%}  | **PER(x{laneA_reps}):** {per_reps:.2%}  | "
        f"**Lane A latency:** ~{int(lat_ms)} ms"
    )
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RAW (bps)", f"{raw_bps:,.0f}")
    c2.metric("Lane A (bps)", f"{alert_bps:,.0f}")
    c3.metric("Lane B (bps)", f"{laneB_bps:,.0f}")
    c4.metric("Total (bps)", f"{(raw_bps+alert_bps+laneB_bps):,.0f}")

    st.markdown("---")
    st.markdown("**Lane A (Safety Alerts)**")
    st.json(laneA_alerts[:6] if laneA_alerts else [])
    st.markdown("**Lane B (Ops/Maintenance)**")
    st.json(laneB_msgs if laneB_msgs else [])

# ---------- Sankey Communication Flow (Plotly) ----------
if show_sankey:
    st.markdown("---")
    st.subheader("Communication Flow (This Tick)")
    sensors_to_bs = max(1, raw_bps + alert_bps + laneB_bps)
    bs_to_net = sensors_to_bs
    net_to_tms = alert_bps + laneB_bps + (raw_bps if mode!="SEMANTIC" else 0)
    tms_to_train = max(1, len(laneA_alerts)*100)
    tms_to_maint = max(1, len(laneB_msgs)*100)

    nodes = ["Sensors","BS/Edge",f"Network ({bearer})","TMS","Train DAS","Maintenance"]
    idx = {n:i for i,n in enumerate(nodes)}
    link = dict(
        source=[idx["Sensors"], idx["BS/Edge"], idx[f"Network ({bearer})"], idx["TMS"], idx["TMS"]],
        target=[idx["BS/Edge"], idx[f"Network ({bearer})"], idx["TMS"], idx["Train DAS"], idx["Maintenance"]],
        value=[sensors_to_bs, bs_to_net, net_to_tms, tms_to_train, tms_to_maint],
        label=["telemetry/alerts","uplink","ctrl+data","advisories","work orders"]
    )
    sankey = go.Figure(data=[go.Sankey(node=dict(label=nodes), link=link)])
    sankey.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(sankey, use_container_width=True, config={"displayModeBar": False})

# ---------- Footnote ----------
st.caption(
    "Fix: removed non-serializable columns (e.g., 'tech' sets) from pydeck layer data. "
    "Channel model: FSPL + env offset (UMa/RMa), log-normal shadowing (σ≈7 dB), Rician (RMa)/Rayleigh (UMa) fading. "
    "Dynamic bearer with hysteresis/time-to-trigger (5G→LTE→3G→GSM-R). "
    "Lane A repetition for reliability; latency = bearer base + light load jitter. HYBRID = Lane A alerts + sparse RAW."
)
