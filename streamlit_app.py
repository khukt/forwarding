# ENSURE-6G • TMS Rail Demo — Advanced + Playback + BS Discs + Train Halo + Tile Fallback
# Features:
#  • PHY radio: FSPL + shadowing + Rayleigh/Rician → SNR → PER, bearer handover (5G/LTE/3G/GSM-R)
#  • Handover gaps: short outage after confirmed switch (TTT-based)
#  • Dual Connectivity (Lane-A): optional duplication over secondary bearer
#  • Macro capacity model: GOOD/PATCHY/POOR → capacity (bps) + random loss → overload queueing
#  • TSR overlays: polygons along track when high-confidence alerts fire
#  • Playback UI: ▶/⏸ with scrub slider (frame-accurate), decimated path rendering
#  • Deck.gl map: optional OSM tiles, with safe no-tiles fallback

import math, time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from shapely.geometry import LineString, Point
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ENSURE-6G • TMS Rail Demo (Merged)", layout="wide")
st.title("🚆 ENSURE-6G: Raw vs Semantic vs Hybrid — Control Center (TMS) Dashboard")
st.caption("Sundsvall → Stockholm • PHY (SNR→PER, handover gaps) + Macro capacity (GOOD/PATCHY/POOR) + DC + TSR • Playback UI • OSM tile fallback")

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
ROUTE_LS = LineString([(lon, lat) for lat, lon in RAIL_WAYPOINTS])  # for TSR geometry ops

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
    """Resample a polyline to n_pts evenly spaced by arc length. Returns DataFrame(lat, lon, s_m)."""
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
    i0, i1 = idx-1, idx
    d0, d1 = cum[i0], cum[i1]
    w = (tgt - d0) / np.maximum(d1 - d0, 1e-9)
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
    """Return (bs_name, distance_m, quality) via 3-bin metric: GOOD (<=R), PATCHY (<=2.2R), POOR (>2.2R)."""
    best = None
    for name, blat, blon, R in BASE_STATIONS:
        d = haversine_m(lat, lon, blat, blon)
        if d <= R: q = "GOOD"
        elif d <= 2.2*R: q = "PATCHY"
        else: q = "POOR"
        rank = {"GOOD":0, "PATCHY":1, "POOR":2}[q]
        if best is None or rank < best[3]: best = (name, d, q, rank)
    return best[0], best[1], best[2]

def cap_loss(quality, t_sec, base_capacity_kbps=800, burst_factor=1.4, good_loss_pct=0.5, bad_loss_pct=10.0):
    """Map quality to capacity (bps) & random loss (best-effort)."""
    cap_bps = int(base_capacity_kbps * 1000)
    if quality == "GOOD":
        cap_bps = int(cap_bps * burst_factor); loss = good_loss_pct / 100.0
    elif quality == "PATCHY":
        wobble = 0.6 + 0.2 * math.sin(2*math.pi * (t_sec % 30) / 30.0)
        cap_bps = max(int(cap_bps * wobble * 0.9), 1); loss = min(0.4, (bad_loss_pct * 0.5) / 100.0)
    else:
        cap_bps = int(cap_bps * 0.25); loss = bad_loss_pct / 100.0
    return max(1, cap_bps), float(loss)

# =================== Sidebar (controls) ===================
with st.sidebar:
    st.header("Scenario Controls (TMS)")
    mode = st.radio("Communication Mode", ["RAW","SEMANTIC","HYBRID"], index=2)
    sim_minutes_total = st.number_input("Sim Length (minutes)", 5, 180, 30, 5)
    # Playback options (tile toggle here for convenience)
    use_tiles = st.toggle("Use OSM tiles", value=False, help="If tiles fail to load in your environment, disable this.")
    st.markdown("---")
    st.subheader("Safety Lane (Lane-A)")
    laneA_reps = st.slider("Repetitions (reliability)", 1, 3, 2, 1)
    enable_dc = st.checkbox("Dual Connectivity (duplicate Lane-A over secondary bearer)", True)
    dc_min_snr_delta = st.slider("DC: min SNR delta to use secondary (dB)", 0.0, 10.0, 2.0, 0.5)
    st.markdown("---")
    st.subheader("Handover")
    TTT_MS = st.slider("Time-To-Trigger (ms)", 200, 3000, 1200, 100)
    HO_GAP_MS = st.slider("Handover outage (ms)", 0, 1500, 350, 50)
    st.markdown("---")
    st.subheader("TSR policy")
    tsr_conf_threshold = st.slider("Trigger TSR if alert confidence ≥", 0.50, 0.95, 0.80, 0.01)
    tsr_speed_kmh = st.slider("TSR speed (km/h)", 30, 100, 60, 5)
    tsr_len_m = st.slider("TSR length along track (m)", 500, 5000, 1500, 100)

# =================== Session state (route, time, handover) ===================
if "route_df" not in st.session_state or st.session_state.get("route_secs") != int(sim_minutes_total*60):
    SECS = max(2, int(sim_minutes_total * 60))
    st.session_state.route_df = interpolate_polyline(RAIL_WAYPOINTS, SECS)
    st.session_state.seg_labels = label_segments(SECS)
    st.session_state.route_secs = SECS
if "t_idx" not in st.session_state: st.session_state.t_idx = 0
if "playing" not in st.session_state: st.session_state.playing = False
if "bearer" not in st.session_state: st.session_state.bearer = "5G"
if "bearer_prev" not in st.session_state: st.session_state.bearer_prev = "5G"
if "bearer_ttt" not in st.session_state: st.session_state.bearer_ttt = 0.0
if "handover_gap_until" not in st.session_state: st.session_state.handover_gap_until = 0.0
if "tsr_polys" not in st.session_state: st.session_state.tsr_polys = []

route_df = st.session_state.route_df
seg_labels = st.session_state.seg_labels
SECS = st.session_state.route_secs

# =================== Top layout: Map & KPIs ===================
tab_map, tab_flow = st.tabs(["Map & KPIs", "Communication Flow"])

with tab_map:
    colL, colR = st.columns([2.1, 1.4])

    # ---------- Right: Playback & KPIs ----------
    with colR:
        st.subheader("Playback")
        c1, c2 = st.columns(2)
        if c1.button("▶ Simulate", use_container_width=True): st.session_state.playing=True
        if c2.button("⏸ Pause",    use_container_width=True): st.session_state.playing=False

        if st.session_state.playing:
            st_autorefresh(interval=700, key="autoplay_tick_fixed")
            st.session_state.t_idx = min(st.session_state.t_idx + 1, SECS-1)
            if st.session_state.t_idx >= SECS-1: st.session_state.playing=False
            st.slider("Time (s)", 0, SECS-1, value=st.session_state.t_idx, key="time_slider", disabled=True)
        else:
            t_idx_tmp = st.slider("Time (s)", 0, SECS-1, value=st.session_state.t_idx, key="time_slider", disabled=False)
            st.session_state.t_idx = t_idx_tmp

        t_idx = st.session_state.t_idx

        # ---------- Build per-frame state ----------
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
        if "shadow" not in st.session_state: st.session_state.shadow = ShadowingTrack()

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

        # Positions (frame-accurate)
        trainA = (float(route_df.lat.iloc[t_idx]), float(route_df.lon.iloc[t_idx]))
        t_idx_nb = (len(route_df) - 1 - t_idx) % len(route_df)
        trainB = (float(route_df.lat.iloc[t_idx_nb]), float(route_df.lon.iloc[t_idx_nb]))
        segment_name = seg_labels[t_idx]
        s_along = float(route_df.s_m.iloc[t_idx])

        # Serving macro quality
        bs_macro_name, bs_macro_dist, quality = nearest_bs_quality(*trainA)
        cap_bps, rand_loss = cap_loss(quality, t_idx)
        badge = {"GOOD":"🟢","PATCHY":"🟠","POOR":"🔴"}[quality]

        # PHY bearer selection
        def serving_bs(lat, lon):
            dists = [haversine_m(lat, lon, b[1], b[2]) for b in BASE_STATIONS]
            i = int(np.argmin(dists))
            caps = {"5G","LTE","3G","GSMR"}
            return dict(name=BASE_STATIONS[i][0], lat=BASE_STATIONS[i][1], lon=BASE_STATIONS[i][2], tech=caps), dists[i]

        bsA, dA = serving_bs(*trainA)
        envA = env_class(*trainA)
        shadow = st.session_state.shadow

        snr_table, per_table = {}, {}
        for b in ["5G","LTE","3G","GSMR"]:
            if b in bsA["tech"]:
                pl = pathloss_db(TECH[b]["freq"], dA, envA)
                sh = shadow.sample(s_along)
                fad = rician_db(8) if envA=="RMa" else rayleigh_db()
                rx = P_TX - pl + sh + fad
                snr = rx - noise_dbm(TECH[b]["bw"])
                snr_table[b] = snr
                per_table[b] = 1/(1+math.exp(-1.1*(snr-2.0)))
                per_table[b] = max(1e-5, min(0.99, per_table[b]))

        def pick_bearer(snr_table, caps, curr_bearer):
            order = ["5G","LTE","3G","GSMR"]
            avail = [x for x in order if x in caps]
            for b in avail:
                if snr_table.get(b,-99) >= TECH[b]["snr_ok"]:
                    return b, True
            return (max(avail, key=lambda x: snr_table.get(x,-99)) if avail else curr_bearer), bool(avail)

        cand, valid = pick_bearer(snr_table, bsA["tech"], st.session_state.bearer)
        # emulate TTT by checking previous frame's choice vs cand
        if valid and cand != st.session_state.bearer:
            st.session_state.bearer_ttt += 700  # ms per tick (matches st_autorefresh interval)
            if st.session_state.bearer_ttt >= TTT_MS:
                st.session_state.bearer_prev = st.session_state.bearer
                st.session_state.bearer = cand
                st.session_state.bearer_ttt = 0
                st.session_state.handover_gap_until = t_idx + math.ceil(HO_GAP_MS/700.0)  # frame index when gap ends
        else:
            st.session_state.bearer_ttt = 0

        bearer = st.session_state.bearer
        snr_use = snr_table.get(bearer, -20.0)
        per_single = per_table.get(bearer, 0.5)
        lat_ms_phy = TECH[bearer]["base_lat"]

        # DC secondary
        def pick_secondary(primary, snr_table, min_delta_db=2.0):
            alts = [(b, s) for b, s in snr_table.items() if b != primary]
            if not alts: return None
            b2, s2 = max(alts, key=lambda x: x[1])
            return b2 if s2 + 1e-9 >= snr_table[primary] - min_delta_db else None

        secondary = pick_secondary(bearer, snr_table, dc_min_snr_delta) if enable_dc else None
        per_secondary = per_table.get(secondary, None) if secondary else None

        if secondary and per_secondary is not None:
            p1 = (1 - per_single)**laneA_reps
            p2 = (1 - per_secondary)**laneA_reps
            laneA_success_phy = 1 - (1 - p1)*(1 - p2)
        else:
            laneA_success_phy = (1 - per_single)**laneA_reps

        # Synthetic sensors & messages (per frame)
        def synth_sensors():
            base = 24 + 10*math.sin(2*math.pi*((t_idx/60)%1440)/1440)
            data, N_SENS = [], 22
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
                data.append(dict(id=f"S{i:02d}", lat=la, lon=lo, rail_temp_C=round(temp,1),
                                 strain_kN=round(strain,1), ballast_idx=round(ballast,2),
                                 risk_score=round(score,2), risk_label=label, exceeded=exceeded))
            return pd.DataFrame(data)
        sdf = synth_sensors()

        RAW_HZ, HYB_HZ = 2.0, 0.2
        BYTES_RAW, BYTES_ALERT, BYTES_SUM = 24, 280, 180

        raw_points = 0
        laneA_alerts, laneB_msgs = [], []
        for _, row in sdf.iterrows():
            if mode in ("RAW","HYBRID"):
                raw_points += int(RAW_HZ if mode=="RAW" else HYB_HZ)
            if row["risk_label"] in ("medium","high") and (("temp>38" in row["exceeded"]) or ("strain>10" in row["exceeded"])):
                conf = round(0.6+0.4*row["risk_score"],2)
                laneA_alerts.append(dict(
                    event="buckling_risk", sensor=row["id"],
                    location=dict(lat=row["lat"], lon=row["lon"]),
                    severity=row["risk_label"], confidence=conf,
                    evidence=dict(rail_temp_C=row["rail_temp_C"], strain_kN=row["strain_kN"],
                                  ballast_idx=row["ballast_idx"], exceeded=row["exceeded"]),
                    recommendation=dict(tsr_kmh=tsr_speed_kmh), ttl_s=900
                ))
        if mode in ("SEMANTIC","HYBRID"):
            laneB_msgs.append(dict(type="maintenance_summary",
                                   ballast_hotspots=int((sdf.ballast_idx>0.6).sum()),
                                   window=f"t={t_idx}s"))

        # TSR overlays (create/expire)
        def polyline_offset_polygon(center_lat, center_lon, length_m=1500, half_width_m=18, samples=40):
            lat0, lon0 = center_lat, center_lon
            m2deg_lat = 1/111111.0
            m2deg_lon = 1/(111111.0*math.cos(math.radians(lat0)))
            nearest = ROUTE_LS.interpolate(ROUTE_LS.project(Point(lon0, lat0)))
            seg_pts = [nearest]
            step_m = length_m / (samples//2)
            for dir_sign in (+1, -1):
                acc = 0.0
                while acc < length_m/2:
                    acc += step_m
                    s = ROUTE_LS.project(nearest) + dir_sign * acc
                    s = max(0, min(s, ROUTE_LS.length))
                    seg_pts.append(ROUTE_LS.interpolate(s))
            seg_pts = sorted(seg_pts, key=lambda p: ROUTE_LS.project(p))
            p0, p1 = seg_pts[0], seg_pts[-1]
            lat0, lon0 = p0.y, p0.x; lat1, lon1 = p1.y, p1.x
            dx, dy = lon1-lon0, lat1-lat0
            L = math.hypot(dx, dy) + 1e-12
            nx, ny = -dy/L, dx/L
            off_lon = (half_width_m * m2deg_lon) * nx
            off_lat = (half_width_m * m2deg_lat) * ny
            return [[lon0 - off_lon, lat0 - off_lat], [lon0 + off_lon, lat0 + off_lat],
                    [lon1 + off_lon, lat1 + off_lat], [lon1 - off_lon, lat1 - off_lat]]

        for a in laneA_alerts:
            if a["confidence"] >= tsr_conf_threshold:
                poly = polyline_offset_polygon(a["location"]["lat"], a["location"]["lon"], length_m=tsr_len_m, half_width_m=18)
                st.session_state.tsr_polys.append(dict(polygon=poly, speed=tsr_speed_kmh, ttl_s=900, created_idx=t_idx))
        st.session_state.tsr_polys = [p for p in st.session_state.tsr_polys if (t_idx - p["created_idx"]) < p["ttl_s"]]

        # Streams (per frame)
        raw_bps = raw_points * BYTES_RAW
        laneA_bps = len(laneA_alerts) * BYTES_ALERT
        if enable_dc and secondary: laneA_bps *= 2  # DC duplication cost
        laneB_bps = len(laneB_msgs) * BYTES_SUM
        bps_total = raw_bps + laneA_bps + laneB_bps

        # Fuse PHY + Macro + HO gap
        lat_ms = lat_ms_phy + (bps_total/1000)
        if bps_total > cap_bps:
            overload = bps_total / cap_bps
            lat_ms *= min(4.0, 1.0 + 0.35*(overload-1))
            overflow_drop = min(0.5, 0.10*(overload-1))
        else:
            overflow_drop = 0.0

        in_gap = t_idx < st.session_state.handover_gap_until
        gap_loss = 0.30 if in_gap else 0.0
        if in_gap: lat_ms += 80

        loss_be = max(0.0, min(0.9, rand_loss + overflow_drop + gap_loss))
        effective_raw_bps   = int(raw_bps   * (1.0 - loss_be))
        effective_laneB_bps = int(laneB_bps * (1.0 - loss_be))
        laneA_success = laneA_success_phy if (not in_gap or secondary) else max(0.0, laneA_success_phy * 0.85)

        # ---------- KPIs (frame-synchronous) ----------
        st.metric("Segment", segment_name)
        st.metric("Serving BS (PHY)", bsA["name"])
        st.metric("Link quality (macro)", quality)
        st.metric("Capacity (kbps)", int(cap_bps/1000))
        st.metric("Lane A bits (this s)", laneA_bps)
        st.metric("Lane B bits (this s)", laneB_bps)
        st.metric("Raw bits (this s)", raw_bps)
        st.metric("E2E Lane-A success (%)", f"{laneA_success*100:.1f}")
        st.metric("Latency E2E (ms)", int(lat_ms))

        # expose for map panel
        frame = dict(
            t_idx=t_idx, trainA=trainA, trainB=trainB, segment=segment_name,
            quality=quality, in_gap=in_gap, bearer=bearer, bs_macro_name=bs_macro_name,
            cap_bps=cap_bps, rand_loss=rand_loss, laneA_alerts=laneA_alerts,
        )
        st.session_state.__dict__["_frame"] = frame

    # ---------- Left: Map ----------
    with colL:
        st.subheader("Live Map")

        # pull per-frame items
        f = st.session_state._frame
        trainA, trainB, quality, in_gap = f["trainA"], f["trainB"], f["quality"], f["in_gap"]

        # Path decimation for rendering
        step = max(1, SECS // 300)
        path_coords = [[route_df.lon.iloc[i], route_df.lat.iloc[i]] for i in range(0, SECS, step)]
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": path_coords, "name": "Sundsvall→Stockholm"}],
            get_color=[0, 102, 255, 255], width_scale=4, width_min_pixels=2,
        )

        # BS coverage discs + centers
        bs_df = pd.DataFrame(BASE_STATIONS, columns=["name","lat","lon","r_m"])
        bs_cov_layer = pdk.Layer(
            "ScatterplotLayer", data=bs_df, get_position="[lon, lat]",
            get_radius="r_m", get_fill_color="[0,150,0,40]",
            stroked=True, get_line_color=[0,150,0], line_width_min_pixels=1, pickable=True
        )
        bs_dot_layer = pdk.Layer(
            "ScatterplotLayer", data=bs_df, get_position="[lon, lat]",
            get_radius=1200, get_fill_color=[30,144,255,255]
        )

        # Risk circle near Gävle
        def circle_polygon(center, radius_m, n=100):
            lat0, lon0 = center
            m2deg_lat = 1/111111.0
            m2deg_lon = 1/(111111.0*math.cos(math.radians(lat0)))
            return [[lon0 + (radius_m*math.cos(th))*m2deg_lon,
                     lat0 + (radius_m*math.sin(th))*m2deg_lat] for th in np.linspace(0, 2*math.pi, n)]
        risk_layer = pdk.Layer(
            "PolygonLayer",
            data=[{"polygon": circle_polygon((60.6749, 17.1413), 15000)}],
            get_polygon="polygon", get_fill_color=[255,80,80,70],
            get_line_color=[255,80,80], line_width_min_pixels=1,
        )

        # TSR overlays
        tsr_layer = pdk.Layer(
            "PolygonLayer",
            data=[{"polygon": p["polygon"], "tooltip": f"TSR {p['speed']} km/h"} for p in st.session_state.tsr_polys],
            get_polygon="polygon", get_fill_color=[255, 215, 0, 80],
            get_line_color=[255, 215, 0], line_width_min_pixels=1, pickable=True
        )

        # Sensors (simple markers along route)
        # (We reuse the risk zone for visualization simplicity; see right panel for alert JSON)
        N_SENS = 22
        idxs = np.linspace(0, len(route_df)-1, N_SENS).astype(int)
        sensors_df = pd.DataFrame([{"lon": float(route_df.lon.iloc[j]), "lat": float(route_df.lat.iloc[j])} for j in idxs])
        sensors_layer = pdk.Layer(
            "ScatterplotLayer", data=sensors_df, get_position='[lon, lat]',
            get_fill_color=[0,170,0,230], get_radius=2000
        )

        # Train halo (color by macro quality) + icon
        qcol = {"GOOD":[0,170,0,200], "PATCHY":[255,165,0,200], "POOR":[200,0,0,220]}
        cur = pd.DataFrame([{
            "lat": trainA[0], "lon": trainA[1],
            "color": qcol[quality],
            "icon_data": {"url":"https://img.icons8.com/emoji/48/train-emoji.png","width":128,"height":128,"anchorY":128}
        }])
        halo_layer = pdk.Layer(
            "ScatterplotLayer", data=cur, get_position='[lon, lat]',
            get_fill_color='color', get_radius=5000,
            stroked=True, get_line_color=[0,0,0], line_width_min_pixels=1
        )
        train_icon_layer = pdk.Layer(
            "IconLayer", data=cur, get_position='[lon, lat]', get_icon='icon_data',
            get_size=4, size_scale=15
        )

        # Optional OSM tiles with safe fallback
        view_state = pdk.ViewState(latitude=60.7, longitude=17.5, zoom=6.2)
        layers = [path_layer, bs_cov_layer, bs_dot_layer, risk_layer, tsr_layer, sensors_layer, halo_layer, train_icon_layer]
        if use_tiles:
            tile_layer = pdk.Layer("TileLayer", data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                                   min_zoom=0, max_zoom=19, tile_size=256)
            layers = [tile_layer] + layers

        deck = pdk.Deck(
            initial_view_state=view_state,
            map_provider=None if not use_tiles else "carto",
            map_style=None if not use_tiles else "light",
            layers=layers,
            tooltip={"text":"{name}"}
        )
        st.pydeck_chart(deck, use_container_width=True, height=560)
        st.caption("Tiles optional. Train halo color shows macro link quality. ▶ to animate; slider scrubs when paused.")

with tab_flow:
    # Sankey (per frame)
    f = st.session_state._frame
    laneA_alerts = f["laneA_alerts"]
    # Recompute simple per-frame stream sizes to feed Sankey
    RAW_HZ, HYB_HZ = 2.0, 0.2
    BYTES_RAW, BYTES_ALERT, BYTES_SUM = 24, 280, 180
    # Rough counts to visualize flow
    raw_bps = (22 * (RAW_HZ if st.session_state.get("mode","HYBRID")=="RAW" else HYB_HZ)) * BYTES_RAW if st.session_state.get("mode") else 22*HYB_HZ*BYTES_RAW
    laneA_bps = len(laneA_alerts) * BYTES_ALERT
    laneB_bps = BYTES_SUM if len(laneA_alerts)>0 else 0

    sensors_to_bs = max(1, raw_bps + laneA_bps + laneB_bps)
    bs_to_net = sensors_to_bs
    net_to_tms = laneA_bps + laneB_bps + raw_bps
    tms_to_train = max(1, len(laneA_alerts)*100)
    tms_to_maint = max(1, (1 if laneB_bps>0 else 0)*100)

    nodes = ["Sensors","BS/Edge",f"Network ({st.session_state.bearer})","TMS","Train DAS","Maintenance"]
    idx = {n:i for i,n in enumerate(nodes)}
    sankey = go.Figure(data=[go.Sankey(
        node=dict(label=nodes),
        link=dict(
            source=[idx["Sensors"], idx["BS/Edge"], idx[f"Network ({st.session_state.bearer})"], idx["TMS"], idx["TMS"]],
            target=[idx["BS/Edge"], idx[f"Network ({st.session_state.bearer})"], idx["TMS"], idx["Train DAS"], idx["Maintenance"]],
            value=[sensors_to_bs, bs_to_net, net_to_tms, tms_to_train, tms_to_maint],
            label=["telemetry/alerts","uplink","ctrl+data","advisories","work orders"],
        )
    )])
    sankey.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(sankey, use_container_width=True, config={"displayModeBar": False})
