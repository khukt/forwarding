# ENSURE-6G • TMS Railway Demo — OSM Map + Coverage + PHY + Capacity + DC + Handover gaps + TSR
# Tabs: Map (playback) • Console (comms & metrics)
# Uses pydeck TileLayer (OSM), PathLayer, ScatterplotLayer, IconLayer, PolygonLayer

import math, time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from shapely.geometry import LineString
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ENSURE-6G • Rail Safety (TMS)", layout="wide")
st.title("🚆 ENSURE-6G: Rail Safety Demo — TMS Control Center")

# --------------------------- Geography & helpers ---------------------------
R_EARTH = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    p = math.pi/180.0
    dlat = (lat2-lat1)*p
    dlon = (lon2-lon1)*p
    a = math.sin(dlat/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2
    return 2*R_EARTH*math.asin(min(1.0, math.sqrt(a)))

# Rail polyline (lat, lon)
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
    n_pts = max(2, int(n_pts))
    lat = np.array([p[0] for p in points], dtype=float)
    lon = np.array([p[1] for p in points], dtype=float)
    cum = np.zeros(len(points), dtype=float)
    for i in range(1, len(points)):
        cum[i] = cum[i-1] + haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])
    total = float(cum[-1])
    if total <= 0:
        return pd.DataFrame({"lat":[lat[0]]*n_pts, "lon":[lon[0]]*n_pts, "s_m":np.linspace(0,0,n_pts)})
    tgt = np.linspace(0.0, total, n_pts)
    idx = np.searchsorted(cum, tgt, side="right")
    idx = np.clip(idx, 1, len(cum)-1)
    i0 = idx-1; i1 = idx
    d0 = cum[i0]; d1 = cum[i1]
    w = (tgt - d0) / np.maximum(d1 - d0, 1e-9)
    latp = lat[i0] + (lat[i1] - lat[i0]) * w
    lonp = lon[i0] + (lon[i1] - lon[i0]) * w
    return pd.DataFrame({"lat": latp, "lon": lonp, "s_m": tgt})

SEG_NAMES = ["Sundsvall→Hudiksvall","Hudiksvall→Söderhamn","Söderhamn→Gävle","Gävle→Uppsala","Uppsala→Stockholm"]
def segment_labels(n):
    bounds = np.linspace(0, n, len(SEG_NAMES)+1).astype(int)
    lab = np.empty(n, dtype=object)
    for i in range(len(SEG_NAMES)):
        lab[bounds[i]:bounds[i+1]] = SEG_NAMES[i]
    return lab

# --------------------------- PHY & Capacity models ---------------------------
def env_class(lat, lon):
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
def noise_dbm(bw_hz): return -174 + 10*np.log10(bw_hz) + 5

TECH = {
    "5G":  dict(freq=3.5,  bw=5e6,   base_lat=20,  snr_ok=3,  snr_hold=1),
    "LTE": dict(freq=1.8,  bw=3e6,   base_lat=35,  snr_ok=0,  snr_hold=-2),
    "3G":  dict(freq=2.1,  bw=1.5e6, base_lat=60,  snr_ok=-2, snr_hold=-4),
    "GSMR":dict(freq=0.9,  bw=200e3, base_lat=120, snr_ok=-4, snr_hold=-6),
}
P_TX = 43  # dBm

def nearest_sites(lat, lon, k=2):
    dists = [haversine_m(lat, lon, b[1], b[2]) for b in BASE_STATIONS]
    order = np.argsort(dists)
    return [(BASE_STATIONS[i], dists[i]) for i in order[:k]]

def bearer_snr(lat, lon, s_m, site, env):
    name, blat, blon, _ = site
    d = haversine_m(lat, lon, blat, blon)
    snr = {}
    for b in ["5G","LTE","3G","GSMR"]:
        pl = pathloss_db(TECH[b]["freq"], d, env)
        sh = shadow.sample(s_m)
        fad = rician_db(8) if env=="RMa" else rayleigh_db()
        rx = P_TX - pl + sh + fad
        snr[b] = rx - noise_dbm(TECH[b]["bw"])
    return d, snr

def pick_bearer(snr_table, curr):
    order = ["5G","LTE","3G","GSMR"]
    for b in order:
        if snr_table.get(b, -99) >= TECH[b]["snr_ok"]:
            return b
    return max(order, key=lambda b: snr_table.get(b, -99))

def per_from_snr(snr_db):
    x0, k = 2.0, -1.1
    per = 1/(1+math.exp(k*(snr_db-x0)))
    return max(1e-5, min(0.99, per))

def macro_quality(lat, lon):
    # GOOD (<=R), PATCHY (<=2.2R), POOR (>2.2R)
    best = None
    for name, blat, blon, R in BASE_STATIONS:
        d = haversine_m(lat, lon, blat, blon)
        q = "GOOD" if d<=R else ("PATCHY" if d<=2.2*R else "POOR")
        rank = {"GOOD":0, "PATCHY":1, "POOR":2}[q]
        if best is None or rank < best[3]:
            best = (name, d, q, rank, R)
    return best[0], best[1], best[2], best[4]

def cap_loss(quality, t_sec, base_capacity_kbps=800, burst_factor=1.4,
             good_loss_pct=0.5, bad_loss_pct=10.0):
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

# --------------------------- Sidebar controls ---------------------------
with st.sidebar:
    st.header("Scenario Controls")
    strategy = st.radio("Communication Strategy", ["RAW","SEMANTIC","HYBRID"], index=2)
    duration_min = st.slider("Scenario Duration (min)", 2, 60, 10, 2)
    laneA_reps = st.slider("Lane-A Repetitions", 1, 3, 2, 1)
    enable_dc = st.checkbox("Enable Dual Connectivity in PATCHY", True)
    enable_handover_gaps = st.checkbox("Enable Handover Gaps", True)
    tsr_threshold = st.slider("TSR Trigger Threshold (risk score)", 0.50, 0.95, 0.75, 0.05)
    st.caption("DC combines two best cells in PATCHY; TSR polygon appears when instantaneous risk exceeds threshold.")

# --------------------------- Build route & sensors ---------------------------
SECS = max(2, int(duration_min*60))
if ("route_df" not in st.session_state) or (st.session_state.get("route_secs") != SECS):
    route_df = interpolate_polyline(RAIL_WAYPOINTS, SECS)
    route_df["segment"] = segment_labels(SECS)
    # 24 sensors along the line (for map + risk)
    SENS_IDX = np.linspace(0, SECS-1, 24).astype(int)
    sensors_df = route_df.loc[SENS_IDX, ["lat","lon"]].copy()
    sensors_df["id"] = [f"S{i:02d}" for i in range(len(sensors_df))]
    # Precompute a static LineString for fast map path
    st.session_state.route_df = route_df.reset_index(drop=True)
    st.session_state.sensors_df = sensors_df.reset_index(drop=True)
    st.session_state.route_secs = SECS
route_df = st.session_state.route_df
sensors_df = st.session_state.sensors_df

# Playback state
if "t_idx" not in st.session_state: st.session_state.t_idx = 0
if "playing" not in st.session_state: st.session_state.playing = False

# --------------------------- Per-second simulation (vectorized precompute) ---------------------------
# Traffic model
RAW_HZ, HYB_HZ = 2.0, 0.2
BYTES_RAW, BYTES_ALERT, BYTES_SUM = 24, 280, 180

# Synthetic risk near Gävle hotspot; produce Lane-A alerts & TSR polygons if exceed threshold
def instant_risk(lat, lon, t_sec):
    d = haversine_m(lat, lon, 60.6749, 17.1413)
    base = 0.25 + 0.15*math.sin(2*math.pi*(t_sec%300)/300)
    hotspot = max(0, 1 - d/15000) * 0.6
    noise = np.random.default_rng(int(t_sec)).normal(0, 0.04)
    return max(0.0, min(1.0, base + hotspot + noise))

# Precompute arrays for the full scenario
def precompute_all():
    N = SECS
    res = dict(
        lat = route_df["lat"].to_numpy(),
        lon = route_df["lon"].to_numpy(),
        segment = route_df["segment"].to_numpy(),
        near_bs = np.empty(N, dtype=object),
        quality = np.empty(N, dtype=object),
        cap_bits = np.zeros(N, dtype=np.int64),
        snr = np.zeros(N, dtype=float),
        bearer = np.empty(N, dtype=object),
        laneA_bits = np.zeros(N, dtype=np.int64),
        laneB_bits = np.zeros(N, dtype=np.int64),
        raw_bits   = np.zeros(N, dtype=np.int64),
        risk = np.zeros(N, dtype=float),
        tsr_active = np.zeros(N, dtype=bool),
        gap = np.zeros(N, dtype=bool),
    )
    bearer_state = "5G"
    ttt_acc_ms = 0.0
    GAP_MS = 1200
    gap_timer_ms = 0.0

    for t in range(N):
        lat, lon = res["lat"][t], res["lon"][t]
        env = env_class(lat, lon)
        # nearest sites (for DC)
        sites = nearest_sites(lat, lon, k=2)
        s_m = float(route_df["s_m"].iloc[t])

        # SNR to primary site
        (site0, d0) = sites[0]
        d0, snr0 = bearer_snr(lat, lon, s_m, site0[0:4], env)
        # choose bearer from primary
        cand = pick_bearer(snr0, bearer_state)

        # TTT/handover gap
        if cand != bearer_state:
            ttt_acc_ms += 1000  # 1 s per step (precompute resolution is 1 s)
            if ttt_acc_ms >= 1200:  # 1.2s TTT
                if enable_handover_gaps:
                    gap_timer_ms = GAP_MS
                bearer_state = cand
                ttt_acc_ms = 0.0
        else:
            ttt_acc_ms = max(0.0, ttt_acc_ms - 500)

        # Gap countdown
        in_gap = False
        if gap_timer_ms > 0:
            in_gap = True
            gap_timer_ms = max(0.0, gap_timer_ms - 1000)

        res["gap"][t] = in_gap

        # PER (primary)
        per_single = per_from_snr(snr0[bearer_state])
        per_eff = per_single**laneA_reps

        # DC in PATCHY: combine with second site diversity
        bs_name_macro, dist_macro, q, R_cell = macro_quality(lat, lon)
        cap_bps, rand_loss = cap_loss(q, t)
        if enable_dc and q == "PATCHY" and len(sites) > 1:
            (site1, d1) = sites[1]
            d1, snr1 = bearer_snr(lat, lon, s_m, site1[0:4], env)
            per2 = per_from_snr(snr1[bearer_state])
            per_eff = per_eff * (per2**laneA_reps)  # independent paths (p1^n * p2^n)
            cap_bps = int(cap_bps * 1.8)  # DC throughput gain with overhead

        # Traffic this second
        n_sensors = len(sensors_df)
        raw_points = int((RAW_HZ if strategy=="RAW" else (HYB_HZ if strategy=="HYBRID" else 0)) * n_sensors)
        raw_bits   = raw_points * BYTES_RAW * 8
        # Lane B (semantic ops) only in SEMANTIC/HYBRID
        laneB_bits = (BYTES_SUM * 8) if strategy in ("SEMANTIC","HYBRID") else 0

        # Risk & Lane-A alerts
        rk = instant_risk(lat, lon, t)
        laneA_bits = (BYTES_ALERT * 8) if rk >= 0.55 else 0  # moderate threshold to generate events
        tsr_flag = rk >= tsr_threshold

        # Capacity loss + handover gap effect (apply to best-effort first)
        be_loss = rand_loss
        if in_gap:
            be_loss = min(0.95, be_loss + 0.6)   # most best-effort wiped during gap
            laneA_bits = int(laneA_bits * 0.8)  # Lane A buffered/repeated, slight penalty

        # Overload queueing & drops
        total_req = raw_bits + laneA_bits + laneB_bits
        if total_req > cap_bps:
            overload = total_req / max(cap_bps,1)
            be_loss = min(0.95, be_loss + 0.10*(overload-1))
        raw_bits_eff   = int(raw_bits   * (1.0 - be_loss))
        laneB_bits_eff = int(laneB_bits * (1.0 - be_loss))

        # Write result arrays
        res["near_bs"][t] = site0[0][0]
        res["quality"][t] = q
        res["cap_bits"][t] = cap_bps
        res["snr"][t] = float(snr0[bearer_state])
        res["bearer"][t] = bearer_state
        res["laneA_bits"][t] = laneA_bits
        res["laneB_bits"][t] = laneB_bits_eff
        res["raw_bits"][t]   = raw_bits_eff
        res["risk"][t] = rk
        res["tsr_active"][t] = bool(tsr_flag)

    return res

if "res_map" not in st.session_state or st.session_state.get("res_secs") != SECS or st.session_state.get("res_strategy") != strategy or st.session_state.get("res_dc") != enable_dc or st.session_state.get("res_gap") != enable_handover_gaps or st.session_state.get("res_tsr") != tsr_threshold or st.session_state.get("res_reps") != laneA_reps:
    st.session_state.res_map = precompute_all()
    st.session_state.res_secs = SECS
    st.session_state.res_strategy = strategy
    st.session_state.res_dc = enable_dc
    st.session_state.res_gap = enable_handover_gaps
    st.session_state.res_tsr = tsr_threshold
    st.session_state.res_reps = laneA_reps

res_map = st.session_state.res_map

# --------------------------- Tabs ---------------------------
tab_map, tab_console = st.tabs(["🗺️ Map", "📊 Console"])

# =================== MAP ===================
with tab_map:
    colL, colR = st.columns([2,1])

    # Right controls (Playback + KPIs)
    with colR:
        st.subheader("Playback")
        c1, c2 = st.columns(2)
        if c1.button("▶ Simulate", use_container_width=True): st.session_state.playing=True
        if c2.button("⏸ Pause", use_container_width=True):   st.session_state.playing=False

        if st.session_state.playing:
            st_autorefresh(interval=700, key="autoplay_tick_fixed")
            st.session_state.t_idx = min(st.session_state.t_idx + 1, SECS-1)
            if st.session_state.t_idx >= SECS-1: st.session_state.playing=False
            st.slider("Time (s)", 0, SECS-1, value=st.session_state.t_idx, key="time_slider", disabled=True)
        else:
            t_idx = st.slider("Time (s)", 0, SECS-1, value=st.session_state.t_idx, key="time_slider", disabled=False)
            st.session_state.t_idx = t_idx

        t_idx = st.session_state.t_idx
        live_strategy = strategy
        st.metric("Strategy (map)", live_strategy)
        st.metric("Segment", str(route_df.loc[t_idx,"segment"]))
        st.metric("Nearest BS", str(res_map["near_bs"][t_idx]))
        st.metric("Link quality", str(res_map["quality"][t_idx]))
        st.metric("Lane A bits this second", int(res_map["laneA_bits"][t_idx]))
        st.metric("Lane B bits this second", int(res_map["laneB_bits"][t_idx]))
        st.metric("Raw bits this second",   int(res_map["raw_bits"][t_idx]))
        st.metric("Capacity (kbps)", int(res_map["cap_bits"][t_idx]/1000))
        if res_map["gap"][t_idx]:
            st.warning("Handover gap ongoing (best-effort throttled)")

    # Left: Map (OSM tiles + layers)
    with colL:
        tile_layer = pdk.Layer(
            "TileLayer",
            data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            min_zoom=0, max_zoom=19, tile_size=256
        )

        # Route path (downsample for performance)
        step = max(1, SECS//300)
        path_coords = [[route_df.loc[i,"lon"], route_df.loc[i,"lat"]] for i in range(0, SECS, step)]
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": path_coords, "name":"Sundsvall→Stockholm"}],
            get_color=[60,60,160], width_scale=4, width_min_pixels=2
        )

        # BS coverage discs (cell size)
        bs_df = pd.DataFrame(BASE_STATIONS, columns=["name","lat","lon","r_m"])
        bs_layer = pdk.Layer(
            "ScatterplotLayer",
            data=bs_df,
            get_position="[lon, lat]",
            get_radius="r_m",
            get_fill_color="[0,150,0,40]",
            stroked=True,
            get_line_color=[0,150,0],
            line_width_min_pixels=1,
            pickable=True,
        )

        # Sensors (simple markers like BS, smaller radius)
        sens_plot = sensors_df.copy()
        sens_plot["r_m"] = 1200
        sens_layer = pdk.Layer(
            "ScatterplotLayer",
            data=sens_plot,
            get_position="[lon, lat]",
            get_radius="r_m",
            get_fill_color=[255,140,0,120],
            stroked=True,
            get_line_color=[180,90,0],
            line_width_min_pixels=1,
            pickable=True
        )

        # Train icon + quality halo
        qcol = {"GOOD":[0,170,0], "PATCHY":[255,165,0], "POOR":[200,0,0]}
        cur = pd.DataFrame([{
            "lat": route_df.loc[st.session_state.t_idx,"lat"],
            "lon": route_df.loc[st.session_state.t_idx,"lon"],
            "icon_data": {"url":"https://img.icons8.com/emoji/48/train-emoji.png",
                          "width":128,"height":128,"anchorY":128},
            "color": qcol[res_map["quality"][st.session_state.t_idx]],
        }])
        train_icon_layer = pdk.Layer(
            "IconLayer", data=cur, get_position='[lon, lat]',
            get_icon='icon_data', get_size=4, size_scale=15
        )
        halo_layer = pdk.Layer(
            "ScatterplotLayer", data=cur, get_position='[lon, lat]',
            get_fill_color='color', get_radius=5000,
            stroked=True, get_line_color=[0,0,0], line_width_min_pixels=1
        )

        # TSR polygons (as red circles centered where risk > threshold)
        tsr_pts = []
        for i in range(max(0, st.session_state.t_idx-10), st.session_state.t_idx+1):
            if res_map["tsr_active"][i]:
                tsr_pts.append([route_df.loc[i,"lon"], route_df.loc[i,"lat"]])
        def make_circle(lon, lat, radius_m, n=60):
            m2deg_lat = 1/111111.0
            m2deg_lon = 1/(111111.0*math.cos(math.radians(lat)))
            return [[lon + (radius_m*math.cos(th))*m2deg_lon,
                     lat + (radius_m*math.sin(th))*m2deg_lat] for th in np.linspace(0,2*math.pi,n)]
        tsr_polys = [{"polygon": make_circle(lon, lat, 5000)} for lon,lat in tsr_pts]
        tsr_layer = pdk.Layer(
            "PolygonLayer",
            data=tsr_polys,
            get_polygon="polygon",
            get_fill_color=[255,0,0,60],
            get_line_color=[180,0,0],
            line_width_min_pixels=1,
        )

        view_state = pdk.ViewState(latitude=60.7, longitude=17.5, zoom=6.2, pitch=0)
        deck = pdk.Deck(
            layers=[tile_layer, path_layer, bs_layer, sens_layer, tsr_layer, halo_layer, train_icon_layer],
            initial_view_state=view_state,
            map_style=None,
            tooltip={"text":"{name}"}
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption("OSM tiles. Green discs show BS coverage. Orange marks sensors. Train halo color is link quality. TSRs in translucent red.")

# =================== CONSOLE ===================
with tab_console:
    st.subheader("Comms Console")
    t_idx = st.session_state.t_idx
    lat, lon = route_df.loc[t_idx,"lat"], route_df.loc[t_idx,"lon"]
    near = res_map["near_bs"][t_idx]
    q = res_map["quality"][t_idx]
    cap = res_map["cap_bits"][t_idx]
    b_raw, b_la, b_lb = res_map["raw_bits"][t_idx], res_map["laneA_bits"][t_idx], res_map["laneB_bits"][t_idx]
    gap = res_map["gap"][t_idx]
    risk = res_map["risk"][t_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nearest BS", str(near))
    c2.metric("Quality", str(q))
    c3.metric("Cap (kbps)", int(cap/1000))
    c4.metric("Risk", f"{risk:.2f}")
    if gap: st.warning("Handover gap: best-effort suppressed; Lane-A slightly buffered")

    st.markdown("**Per-second Traffic (effective delivered):**")
    st.write({
        "RAW bits": int(b_raw),
        "Lane-A bits (alerts)": int(b_la),
        "Lane-B bits (semantic)": int(b_lb),
        "Total (kb)": round((b_raw + b_la + b_lb)/8/1000, 2)
    })

    st.markdown("---")
    st.markdown("**Notes**")
    st.markdown("""
- **PHY:** path loss (UMa/RMa) + lognormal shadowing + Rayleigh/Rician fading → SNR → PER.  
- **Handover:** TTT=1.2s; optional **handover gap** throttles best-effort for ~1.2s.  
- **DC:** When **PATCHY**, second-best site is used: diversity reduces effective PER and boosts capacity.  
- **Capacity:** GOOD/PATCHY/POOR → capacity budget + random loss applied to best-effort (RAW, Lane-B).  
- **Lane-A:** protected via **repetition**; only slightly penalized in gaps; TSR polygon is raised if risk ≥ threshold.
""")
