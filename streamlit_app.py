# ENSURE-6G • TMS Rail Demo — Robust per-frame builder + Sensor uplink QoS + Track heat + TSR/STOP/CRASH + Maintenance
# Fixes: centralized _frame build BEFORE columns (no KeyError on first render), graceful fallbacks.
# Map: deck.gl (pydeck) with optional OSM tiles; playback with frame-accurate slider.
# Radio: PHY (SNR→PER, TTT, HO gap) + Macro capacity/loss at train *and* per-sensor uplink.

import math, time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from shapely.geometry import LineString, Point
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ENSURE-6G • TMS Rail (Ops-Realistic, Robust)", layout="wide")
st.title("🚆 ENSURE-6G: Raw vs Semantic vs Hybrid — Control Center (TMS)")

# -------------------- Geography --------------------
R_EARTH = 6371000.0
def haversine_m(lat1, lon1, lat2, lon2):
    p = math.pi/180.0
    dlat = (lat2-lat1)*p; dlon = (lon2-lon1)*p
    a = math.sin(dlat/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2
    return 2*R_EARTH*math.asin(min(1.0, math.sqrt(a)))

RAIL_WAYPOINTS = [
    (62.3930,17.3070),(62.1200,17.1500),(61.8600,17.1400),(61.7300,17.1100),
    (61.5600,17.0800),(61.3900,17.0700),(61.3000,17.0600),(61.0700,17.1000),
    (60.8500,17.1600),(60.6749,17.1413),(60.3800,17.3300),(60.2000,17.4500),
    (60.0500,17.5200),(59.9300,17.6100),(59.8586,17.6389),(59.7500,17.8200),
    (59.6600,17.9400),(59.6100,17.9900),(59.5500,18.0300),(59.4800,18.0400),
    (59.4200,18.0600),(59.3700,18.0700),(59.3293,18.0686),
]
ROUTE_LS = LineString([(lon, lat) for lat, lon in RAIL_WAYPOINTS])

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
    cum = np.zeros(len(points))
    for i in range(1,len(points)):
        cum[i] = cum[i-1] + haversine_m(lat[i-1],lon[i-1],lat[i],lon[i])
    total = float(cum[-1])
    tgt = np.linspace(0.0, total, n_pts)
    idx = np.searchsorted(cum, tgt, side="right")
    idx = np.clip(idx, 1, len(cum)-1)
    i0, i1 = idx-1, idx
    d0, d1 = cum[i0], cum[i1]
    w = (tgt-d0)/np.maximum(d1-d0, 1e-9)
    latp = lat[i0] + (lat[i1]-lat[i0])*w
    lonp = lon[i0] + (lon[i1]-lon[i0])*w
    return pd.DataFrame({"lat":latp,"lon":lonp,"s_m":tgt})

def label_segments(n):
    names = ["Sundsvall→Hudiksvall","Hudiksvall→Söderhamn","Söderhamn→Gävle","Gävle→Uppsala","Uppsala→Stockholm"]
    bounds = np.linspace(0,n,len(names)+1).astype(int)
    lab = np.empty(n, dtype=object)
    for i in range(len(names)): lab[bounds[i]:bounds[i+1]] = names[i]
    return lab

def nearest_bs_quality(lat, lon):
    best=None
    for name, blat, blon, R in BASE_STATIONS:
        d=haversine_m(lat,lon,blat,blon)
        if d<=R: q="GOOD"
        elif d<=2.2*R: q="PATCHY"
        else: q="POOR"
        rank={"GOOD":0,"PATCHY":1,"POOR":2}[q]
        if best is None or rank<best[3]: best=(name,d,q,rank)
    return best[0], best[1], best[2]

def cap_loss(qual, t_sec, base_capacity_kbps=800, burst_factor=1.4, good_loss_pct=0.5, bad_loss_pct=10.0):
    cap = int(base_capacity_kbps*1000)
    if qual=="GOOD":
        return int(cap*burst_factor), good_loss_pct/100.0
    elif qual=="PATCHY":
        wobble = 0.6+0.2*math.sin(2*math.pi*(t_sec%30)/30.0)
        return max(int(cap*wobble*0.9),1), min(0.4,(bad_loss_pct*0.5)/100.0)
    else:
        return int(cap*0.25), bad_loss_pct/100.0

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Scenario Controls (TMS)")
    sim_minutes_total = st.number_input("Sim Length (minutes)", 5, 120, 20, 5)
    use_tiles = st.toggle("Use OSM tiles", False)
    mode = st.radio("Comm Mode", ["RAW","SEMANTIC","HYBRID"], index=2)
    st.markdown("---")
    st.subheader("Lane A (Safety)")
    laneA_reps = st.slider("Repetitions", 1, 3, 2, 1)
    enable_dc = st.checkbox("Dual Connectivity", True)
    dc_min_snr_delta = st.slider("DC min SNR delta (dB)", 0.0, 10.0, 2.0, 0.5)
    st.markdown("---")
    st.subheader("Handover")
    TTT_MS = st.slider("Time-To-Trigger (ms)", 200, 3000, 1200, 100)
    HO_GAP_MS = st.slider("Handover outage (ms)", 0, 1500, 350, 50)
    st.markdown("---")
    st.subheader("TSR / STOP")
    tsr_conf_critical = st.slider("Critical buckling threshold", 0.60, 0.95, 0.85, 0.01)
    tsr_speed_kmh = st.slider("TSR speed (km/h)", 30, 120, 60, 5)
    stop_on_critical = st.checkbox("Issue STOP on very high risk (≥0.92)", True)
    st.markdown("---")
    st.subheader("Maintenance")
    repair_time_s = st.slider("On-site repair time (s)", 30, 900, 180, 10)
    crew_capacity = st.slider("Max concurrent crews", 1, 4, 2, 1)

# -------------------- Session init --------------------
SECS = max(2, int(sim_minutes_total*60))
if "route_secs" not in st.session_state or st.session_state.route_secs != SECS:
    st.session_state.route_df = interpolate_polyline(RAIL_WAYPOINTS, SECS)
    st.session_state.seg_labels = label_segments(SECS)
    st.session_state.route_secs = SECS
if "t_idx" not in st.session_state: st.session_state.t_idx = 0
if "playing" not in st.session_state: st.session_state.playing = False

# persistent ops state
if "bearer" not in st.session_state: st.session_state.bearer="5G"
if "bearer_prev" not in st.session_state: st.session_state.bearer_prev="5G"
if "bearer_ttt" not in st.session_state: st.session_state.bearer_ttt=0
if "handover_gap_until" not in st.session_state: st.session_state.handover_gap_until=-1
if "tsr_polys" not in st.session_state: st.session_state.tsr_polys=[]  # {poly, speed, created_idx, critical:bool, ack_train:bool, stop:bool}
if "work_orders" not in st.session_state: st.session_state.work_orders=[] # {polygon, created_idx, status, eta_done_idx}

route_df = st.session_state.route_df
seg_labels = st.session_state.seg_labels

# -------------------- PHY model helpers --------------------
def env_class(lat, lon):
    for c in [(62.3913,17.3063),(60.6749,17.1413),(59.8586,17.6389),(59.3293,18.0686)]:
        if haversine_m(lat,lon,c[0],c[1])<15000: return "UMa"
    return "RMa"
def pathloss_db(freq_GHz, d_m, env):
    if d_m<1: d_m=1
    fspl = 32.4 + 20*np.log10(freq_GHz*1000) + 20*np.log10(d_m/1000)
    return fspl + (7 if env=="UMa" else 3)
class ShadowingTrack:
    def __init__(self,sigma=7,decor=100,seed=7):
        self.sigma=sigma; self.decor=decor; self.rng=np.random.default_rng(seed)
        self.last_s=0.0; self.curr=0.0
    def sample(self,s):
        rho=np.exp(-abs(s-self.last_s)/self.decor)
        self.curr = rho*self.curr + math.sqrt(max(1e-9,1-rho**2))*self.rng.normal(0,self.sigma)
        self.last_s=s; return self.curr
if "shadow" not in st.session_state: st.session_state.shadow=ShadowingTrack()
def rician_db(K_dB=8):
    K=10**(K_dB/10); h = math.sqrt(K/(K+1)) + (np.random.normal(0,1/np.sqrt(2))+1j*np.random.normal(0,1/np.sqrt(2)))
    p=(abs(h)**2)/(K+1); return 10*np.log10(max(p,1e-6))
def rayleigh_db():
    h = np.random.normal(0,1/np.sqrt(2))+1j*np.random.normal(0,1/np.sqrt(2))
    p=abs(h)**2; return 10*np.log10(max(p,1e-6))
def noise_dbm(bw_hz): return -174 + 10*np.log10(bw_hz) + 5
TECH = {
    "5G":  dict(freq=3.5,  bw=5e6,   base_lat=20,  snr_ok=3,  snr_hold=1),
    "LTE": dict(freq=1.8,  bw=3e6,   base_lat=35,  snr_ok=0,  snr_hold=-2),
    "3G":  dict(freq=2.1,  bw=1.5e6, base_lat=60,  snr_ok=-2, snr_hold=-4),
    "GSMR":dict(freq=0.9,  bw=200e3, base_lat=120, snr_ok=-4, snr_hold=-6),
}
P_TX=43

def serving_bs(lat,lon):
    d=[haversine_m(lat,lon,b[1],b[2]) for b in BASE_STATIONS]
    i=int(np.argmin(d))
    return dict(name=BASE_STATIONS[i][0],lat=BASE_STATIONS[i][1],lon=BASE_STATIONS[i][2],
                tech={"5G","LTE","3G","GSMR"}), d[i]

def per_from_snr(snr_db):
    x0,k=2.0,-1.1
    per = 1/(1+math.exp(k*(snr_db-x0)))
    return max(1e-5, min(0.99, per))

def pick_bearer(snr_table, caps, curr):
    order=["5G","LTE","3G","GSMR"]
    avail=[b for b in order if b in caps]
    for b in avail:
        if snr_table.get(b,-99)>=TECH[b]["snr_ok"]: return b, True
    if avail: return max(avail,key=lambda x: snr_table.get(x,-99)), True
    return curr, False

def pick_secondary(primary, snr_table, min_delta_db=2.0):
    alts=[(b,s) for b,s in snr_table.items() if b!=primary]
    if not alts: return None
    b2,s2=max(alts,key=lambda x:x[1])
    return b2 if s2 + 1e-9 >= snr_table[primary] - min_delta_db else None

# -------------------- Tabs --------------------
tab_map, tab_flow, tab_ops = st.tabs(["Map & KPIs", "Comm Flow", "Ops (Maintenance/Incidents)"])

with tab_map:
    # ========= Centralized per-frame builder (runs BEFORE columns) =========
    def build_frame(t_idx):
        trainA = (float(route_df.lat.iloc[t_idx]), float(route_df.lon.iloc[t_idx]))
        t_idx_nb = (len(route_df)-1 - t_idx) % len(route_df)
        trainB = (float(route_df.lat.iloc[t_idx_nb]), float(route_df.lon.iloc[t_idx_nb]))
        seg = seg_labels[t_idx]
        # pre-place 22 sensors along route
        N_SENS=22
        sidx = np.linspace(0,len(route_df)-1,N_SENS).astype(int)
        sensors = pd.DataFrame([{"sid":f"S{i:02d}","lat":float(route_df.lat.iloc[j]),"lon":float(route_df.lon.iloc[j])}
                                for i,j in enumerate(sidx)])
        # macro at train
        bs_name, bs_dist, quality = nearest_bs_quality(*trainA)
        cap_bps, rand_loss = cap_loss(quality, t_idx)
        return dict(t=t_idx, trainA=trainA, trainB=trainB, segment=seg,
                    quality=quality, cap_bps=cap_bps, rand_loss=rand_loss,
                    enforce_stop=False, crash=False, sensors=sensors)

    # Make sure _frame exists before columns render
    t0 = st.session_state.get("t_idx", 0)
    if "_frame" not in st.session_state:
        st.session_state._frame = build_frame(t0)
    else:
        # keep it roughly in sync even before right column updates it
        if st.session_state._frame.get("t") != t0:
            st.session_state._frame = build_frame(t0)

    # ---------------- Layout columns ----------------
    colR, colL = st.columns([1.0,2.2])

    # ---------- Right: Playback & KPIs ----------
    with colR:
        st.subheader("Playback")
        c1,c2=st.columns(2)
        if c1.button("▶ Simulate", use_container_width=True): st.session_state.playing=True
        if c2.button("⏸ Pause", use_container_width=True): st.session_state.playing=False
        if st.session_state.playing:
            st_autorefresh(interval=700, key="tick")
            st.session_state.t_idx = min(st.session_state.t_idx+1, SECS-1)
            if st.session_state.t_idx>=SECS-1: st.session_state.playing=False
            st.slider("Time (s)", 0, SECS-1, value=st.session_state.t_idx, disabled=True)
        else:
            t = st.slider("Time (s)", 0, SECS-1, value=st.session_state.t_idx)
            st.session_state.t_idx = t
        t = st.session_state.t_idx

        # ------------- Per-frame computation (full) -------------
        # Train position
        trainA=(float(route_df.lat.iloc[t]), float(route_df.lon.iloc[t]))
        t_nb=(len(route_df)-1-t)%len(route_df); trainB=(float(route_df.lat.iloc[t_nb]), float(route_df.lon.iloc[t_nb]))
        seg = seg_labels[t]; s_along=float(route_df.s_m.iloc[t])

        # Sensor grid (22 along route)
        sensors = st.session_state._frame["sensors"].copy()

        # PHY at train
        bsA, dA = serving_bs(*trainA)
        envA = env_class(*trainA)
        shadow = st.session_state.shadow
        snr_table={}
        for b in ["5G","LTE","3G","GSMR"]:
            if b in bsA["tech"]:
                pl = pathloss_db(TECH[b]["freq"], dA, envA)
                sh = shadow.sample(s_along)
                fad = rician_db(8) if envA=="RMa" else rayleigh_db()
                rx = P_TX - pl + sh + fad
                snr_table[b] = rx - noise_dbm(TECH[b]["bw"])

        cand, valid = pick_bearer(snr_table, bsA["tech"], st.session_state.bearer)
        if valid and cand != st.session_state.bearer:
            st.session_state.bearer_ttt += 700
            if st.session_state.bearer_ttt >= TTT_MS:
                st.session_state.bearer_prev = st.session_state.bearer
                st.session_state.bearer = cand
                st.session_state.bearer_ttt=0
                st.session_state.handover_gap_until = t + math.ceil(HO_GAP_MS/700.0)
        else:
            st.session_state.bearer_ttt=0

        bearer = st.session_state.bearer
        snr_use = snr_table.get(bearer,-20.0)
        per_single = per_from_snr(snr_use)
        lat_ms_phy = TECH[bearer]["base_lat"]

        secondary = pick_secondary(bearer, snr_table, dc_min_snr_delta) if enable_dc else None
        per_secondary = per_from_snr(snr_table.get(secondary,-20.0)) if secondary else None
        laneA_success_phy = ((1-per_single)**laneA_reps) if not secondary else 1-((1-(1-per_single)**laneA_reps)*(1-(1-per_secondary)**laneA_reps))

        # Macro at train
        bs_macro_name, bs_macro_dist, quality = nearest_bs_quality(*trainA)
        cap_bps_train, rand_loss = cap_loss(quality, t)
        in_gap = (t < st.session_state.handover_gap_until)
        gap_loss = 0.30 if in_gap else 0.0

        # Sensors: risk + uplink QoS
        def sensor_row(r):
            d_risk = haversine_m(r.lat, r.lon, 60.6749, 17.1413)
            boost = max(0,1-d_risk/15000)*14
            base = 24 + 10*math.sin(2*math.pi*((t/60)%1440)/1440)
            temp = base + np.random.normal(0,0.6) + boost
            strain = max(0.0, (temp-35)*0.8 + np.random.normal(0,0.5))
            ballast = max(0.0, np.random.normal(0.3,0.1) + 0.02*boost)
            score = min(1.0, 0.01*(temp-30)**2 + 0.04*max(0, strain-8))
            label = "high" if score>0.75 else ("medium" if score>0.4 else "low")
            exceeded=[]
            if temp>=38: exceeded.append("temp>38")
            if strain>=10: exceeded.append("strain>10")
            _,_,qualS = nearest_bs_quality(r.lat, r.lon)
            capS, lossS = cap_loss(qualS, t)
            return dict(lat=r.lat, lon=r.lon, score=score, label=label, exceeded=exceeded,
                        temp=round(temp,1), strain=round(strain,1), ballast=round(ballast,2),
                        qualS=qualS, capS=capS, lossS=lossS)
        S = sensors.apply(sensor_row, axis=1, result_type="expand")
        sensors = pd.concat([sensors, S], axis=1)

        # Streams per frame (sensor uplink first)
        RAW_HZ, HYB_HZ = 2.0, 0.2
        BYTES_RAW, BYTES_ALERT, BYTES_SUM = 24, 280, 180
        raw_points = (RAW_HZ if mode=="RAW" else (HYB_HZ if mode=="HYBRID" else 0.0))
        raw_points_total = int(len(sensors)*raw_points)
        delivery_ratio = float(np.mean(1.0 - sensors.lossS.values)) if raw_points_total>0 else 1.0
        raw_bps_uplink = raw_points_total * BYTES_RAW
        raw_bps_delivered = int(raw_bps_uplink * delivery_ratio)

        # Lane-A alerts generated on sensors, filtered by uplink success
        rng = np.random.default_rng(42+t)
        laneA_alerts_all=[]
        for r in sensors.itertuples():
            if r.label in ("medium","high") and (("temp>38" in r.exceeded) or ("strain>10" in r.exceeded)):
                conf = round(0.6+0.4*r.score,2)
                laneA_alerts_all.append(dict(sid=r.sid, location=dict(lat=r.lat, lon=r.lon),
                                             severity=r.label, confidence=conf,
                                             evidence=dict(rail_temp_C=r.temp, strain_kN=r.strain, ballast_idx=r.ballast)))
        laneA_alerts=[]
        for a in laneA_alerts_all:
            lossS = float(sensors.loc[sensors.sid==a["sid"], "lossS"].iloc[0])
            if rng.uniform() < (1.0 - lossS): laneA_alerts.append(a)

        laneB_msgs=[]
        if mode in ("SEMANTIC","HYBRID"):
            laneB_msgs.append(dict(type="maintenance_summary",
                                   ballast_hotspots=int((sensors.ballast>0.6).sum()),
                                   alerts=len(laneA_alerts), window=f"t={t}s"))

        laneA_bps_uplink = len(laneA_alerts)*BYTES_ALERT
        laneB_bps_uplink = len(laneB_msgs)*BYTES_SUM

        # TMS decisions: TSR / STOP & downlink
        def tsr_poly(center_lat,center_lon,length_m=1500,half_w=18):
            lat0,lon0=center_lat,center_lon
            m2deg_lat=1/111111.0; m2deg_lon=1/(111111.0*math.cos(math.radians(lat0)))
            nearest = ROUTE_LS.interpolate(ROUTE_LS.project(Point(lon0, lat0)))
            step_m=length_m/10; pts=[nearest]
            for sgn in (+1,-1):
                acc=0
                while acc<length_m/2:
                    acc+=step_m
                    s = ROUTE_LS.project(nearest)+sgn*acc
                    s=max(0,min(s,ROUTE_LS.length)); pts.append(ROUTE_LS.interpolate(s))
            pts=sorted(pts, key=lambda p: ROUTE_LS.project(p))
            p0,p1=pts[0],pts[-1]
            dx,dy = p1.x-p0.x, p1.y-p0.y; L=math.hypot(dx,dy)+1e-12
            nx,ny = -dy/L, dx/L
            off_lon=(half_w*m2deg_lon)*nx; off_lat=(half_w*m2deg_lat)*ny
            return [[p0.x-off_lon,p0.y-off_lat],[p0.x+off_lon,p0.y+off_lat],
                    [p1.x+off_lon,p1.y+off_lat],[p1.x-off_lon,p1.y-off_lat]]

        new_tsr=[]
        for a in laneA_alerts:
            if a["confidence"] >= tsr_conf_critical:
                poly = tsr_poly(a["location"]["lat"], a["location"]["lon"])
                very_high = a["confidence"]>=0.92 and stop_on_critical
                new_tsr.append(dict(polygon=poly, speed=tsr_speed_kmh, created_idx=t,
                                    critical=True, ack_train=False, stop=very_high))
        for p in new_tsr:
            st.session_state.tsr_polys.append(p)
            st.session_state.work_orders.append(dict(polygon=p["polygon"], created_idx=t,
                                                     status="Dispatched", eta_done_idx=t+repair_time_s))

        # Downlink to Train
        _,_,qual_down = nearest_bs_quality(*trainA)
        _, rand_down = cap_loss(qual_down, t)
        loss_down = min(0.95, rand_down + (0.30 if in_gap else 0.0))
        down_ok = (np.random.random() < (1.0 - loss_down))
        if down_ok:
            for p in st.session_state.tsr_polys: p["ack_train"]=True

        # STOP enforcement
        enforce_stop = any(p.get("stop",False) and p.get("ack_train",False) for p in st.session_state.tsr_polys)
        if enforce_stop and st.session_state.playing:
            st.session_state.t_idx = max(0, st.session_state.t_idx-1)
            t = st.session_state.t_idx
            trainA=(float(route_df.lat.iloc[t]), float(route_df.lon.iloc[t]))

        # Crash detection: entering un-acked critical TSR
        def point_in_poly(lat,lon,poly):
            xs=[pt[0] for pt in poly]; ys=[pt[1] for pt in poly]
            return (min(ys)<=lat<=max(ys)) and (min(xs)<=lon<=max(xs))
        crash=False
        for p in st.session_state.tsr_polys:
            if p["critical"] and (not p["ack_train"]) and point_in_poly(trainA[0],trainA[1],p["polygon"]):
                crash=True; break

        # Throughputs and latency (display)
        laneA_bps = len(laneA_alerts)*BYTES_ALERT * (2 if (enable_dc and secondary) else 1)
        laneB_bps = len(laneB_msgs)*BYTES_SUM
        raw_bps = raw_bps_delivered
        bps_total = laneA_bps + laneB_bps + raw_bps
        lat_ms = TECH[bearer]["base_lat"] + (bps_total/1000.0)
        if bps_total>cap_bps_train: lat_ms *= min(4.0, 1.0 + 0.35*(bps_total/cap_bps_train - 1))
        if in_gap: lat_ms += 80
        laneA_success = ( (1-per_single)**laneA_reps if not secondary else laneA_success_phy )
        if in_gap and not secondary: laneA_success = max(0.0, laneA_success*0.85)

        # Per-frame arrays
        if "_times" not in st.session_state:
            st.session_state._times = np.full(SECS, np.nan)
            st.session_state.arr = {k:np.full(SECS, np.nan) for k in
                    ["raw_bits","laneA_bits","laneB_bits","cap_bits","snr_db","laneA_succ","lat_ms"]}
            st.session_state.qual = np.array([""]*SECS, dtype=object)
            st.session_state.seg = np.array([""]*SECS, dtype=object)
        if math.isnan(st.session_state._times[t]):
            st.session_state._times[t]=t
            st.session_state.arr["raw_bits"][t]=raw_bps
            st.session_state.arr["laneA_bits"][t]=laneA_bps
            st.session_state.arr["laneB_bits"][t]=laneB_bps
            st.session_state.arr["cap_bits"][t]=cap_bps_train
            st.session_state.arr["snr_db"][t]=snr_use
            st.session_state.arr["laneA_succ"][t]=laneA_success
            st.session_state.arr["lat_ms"][t]=lat_ms
            st.session_state.qual[t]=quality
            st.session_state.seg[t]=seg

        # KPI cards
        badge={"GOOD":"🟢","PATCHY":"🟠","POOR":"🔴"}[quality]
        st.metric("Segment", seg)
        st.metric("Bearer", f"{bearer} (TTT {int(st.session_state.bearer_ttt)} ms)")
        st.metric("Macro quality", f"{badge} {quality}")
        st.metric("Capacity (kbps)", int(cap_bps_train/1000))
        st.metric("LaneA bits (this s)", int(laneA_bps))
        st.metric("LaneB bits (this s)", int(laneB_bps))
        st.metric("RAW bits (this s)", int(raw_bps))
        st.metric("LaneA success (PHY/DC) %", f"{(laneA_success*100):.1f}")
        st.metric("E2E latency (ms)", int(lat_ms))
        if enforce_stop: st.error("STOP enforced")
        if crash: st.error("🚨 CRASH: critical TSR entered without alert delivery!")

        # ---- Update shared frame so the map ALWAYS has complete data ----
        st.session_state._frame.update({
            "t": t, "trainA": trainA, "trainB": trainB, "segment": seg,
            "quality": quality, "cap_bps": cap_bps_train, "rand_loss": rand_loss,
            "enforce_stop": enforce_stop, "crash": crash, "sensors": sensors
        })

    # ---------------- Left: Map panel ----------------
    with colL:
        st.subheader("Live Map (heat, TSR, QoS)")
        f = st.session_state.get("_frame")
        if not f or "t" not in f:
            # final fallback (shouldn't trigger now, but keeps things bulletproof)
            t = st.session_state.get("t_idx", 0)
            trainA = (float(route_df.lat.iloc[t]), float(route_df.lon.iloc[t]))
            sensors = pd.DataFrame([{"lat":trainA[0], "lon":trainA[1], "label":"low", "qualS":"GOOD"}])
            enforce_stop=False; crash=False; quality="GOOD"
        else:
            t = f["t"]; trainA = f["trainA"]; sensors = f["sensors"]
            enforce_stop=f["enforce_stop"]; crash=f["crash"]; quality=f["quality"]

        # Track heat by risk (nearest sensor)
        step = max(1, SECS//300)
        path_coords = [[route_df.lon.iloc[i], route_df.lat.iloc[i]] for i in range(0, SECS, step)]
        def near_sensor_risk(lat,lon):
            d = ((sensors.lat-lat)**2 + (sensors.lon-lon)**2)**0.5
            j = int(np.argmin(d))
            label = sensors.iloc[j].label if "label" in sensors.columns else "low"
            return {"low":[0,170,0,180], "medium":[255,165,0,200], "high":[220,0,0,220]}[label]
        heat = [near_sensor_risk(lat,lon) for lon,lat in path_coords]
        heat_df = pd.DataFrame([{"path":path_coords, "colors":heat}])
        path_layer = pdk.Layer("PathLayer", data=heat_df, get_path="path",
                               get_color="colors", width_scale=4, width_min_pixels=3)

        # BS discs + centers
        bs_df = pd.DataFrame(BASE_STATIONS, columns=["name","lat","lon","r_m"])
        bs_cov_layer = pdk.Layer("ScatterplotLayer", data=bs_df, get_position="[lon, lat]",
                                 get_radius="r_m", get_fill_color="[0,150,0,40]",
                                 stroked=True, get_line_color=[0,150,0], line_width_min_pixels=1)
        bs_dot_layer = pdk.Layer("ScatterplotLayer", data=bs_df, get_position="[lon, lat]",
                                 get_radius=1200, get_fill_color=[30,144,255,255])

        # Sensors pins colored by their uplink quality
        qcol = {"GOOD":[0,170,0,230], "PATCHY":[255,165,0,230], "POOR":[200,0,0,230]}
        sens_df = []
        if "qualS" in sensors.columns:
            for r in sensors.itertuples():
                sens_df.append({"lon":r.lon, "lat":r.lat,
                                "color": qcol.get(r.qualS, [150,150,150,200]),
                                "tooltip": f"{getattr(r,'sid','S?')} • {getattr(r,'label','low')} • uplink {getattr(r,'qualS','?')}"})
        else:
            sens_df.append({"lon":trainA[1], "lat":trainA[0], "color":[0,170,0,230], "tooltip":"sensor"})
        sens_df = pd.DataFrame(sens_df)
        sens_layer = pdk.Layer("ScatterplotLayer", data=sens_df, get_position='[lon, lat]',
                               get_fill_color='color', get_radius=2000, pickable=True)

        # TSR polygons (yellow)
        tsr_layer = pdk.Layer("PolygonLayer",
                              data=[{"polygon":p["polygon"], "tooltip":f"TSR {p['speed']} km/h"} for p in st.session_state.tsr_polys],
                              get_polygon="polygon", get_fill_color=[255,215,0,70],
                              get_line_color=[255,215,0], line_width_min_pixels=1, pickable=True)

        # Train halo & icon (STOP/crash styling)
        qcol_macro = {"GOOD":[0,170,0,200], "PATCHY":[255,165,0,200], "POOR":[200,0,0,220]}
        halo_color = [150,150,150,240] if enforce_stop else ([200,0,0,240] if crash else qcol_macro.get(quality,[0,170,0,200]))
        cur = pd.DataFrame([{
            "lat": trainA[0], "lon": trainA[1],
            "icon_data":{"url":"https://img.icons8.com/emoji/48/train-emoji.png","width":128,"height":128,"anchorY":128}
        }])
        halo_layer = pdk.Layer("ScatterplotLayer", data=cur, get_position='[lon, lat]',
                               get_fill_color=halo_color, get_radius=5200,
                               stroked=True, get_line_color=[0,0,0], line_width_min_pixels=1)
        train_icon_layer = pdk.Layer("IconLayer", data=cur, get_position='[lon, lat]',
                                     get_icon='icon_data', get_size=4, size_scale=15)

        # OSM tiles (optional)
        layers=[path_layer, bs_cov_layer, bs_dot_layer, sens_layer, tsr_layer, halo_layer, train_icon_layer]
        if use_tiles:
            tile_layer = pdk.Layer("TileLayer", data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                                   min_zoom=0, max_zoom=19, tile_size=256)
            layers=[tile_layer]+layers
        view_state = pdk.ViewState(latitude=60.7, longitude=17.5, zoom=6.2)
        deck = pdk.Deck(initial_view_state=view_state, map_provider=None if not use_tiles else "carto",
                        map_style=None if not use_tiles else "light", layers=layers,
                        tooltip={"html":"<b>{tooltip}</b>","style":{"color":"white"}})
        st.pydeck_chart(deck, use_container_width=True, height=560)
        st.caption("Track heat shows risk (green→OK, orange/red→summer hotspots). Sensor pins show **uplink** quality. Yellow=TSR. Halo gray=STOP, red=CRASH risk.")

# ---------------- Comm Flow tab ----------------
with tab_flow:
    st.subheader("Communication Flow (frame-synchronous)")
    t = st.session_state.t_idx
    arr = st.session_state.get("arr", None); bearer = st.session_state.bearer
    if not arr:
        st.info("No data yet — press ▶ Simulate once.")
    else:
        getv=lambda k: (int(arr[k][t]) if (not math.isnan(arr[k][t])) else 0)
        raw_bps  = getv("raw_bits"); laneA_bps = getv("laneA_bits"); laneB_bps = getv("laneB_bits"); cap_bps=getv("cap_bits")
        sensors_to_bs = max(1, raw_bps + laneA_bps + laneB_bps)
        bs_to_core = sensors_to_bs
        core_to_tms = sensors_to_bs
        tms_to_train = max(1, laneA_bps + laneB_bps)
        tms_to_maint = max(1, 100 if laneB_bps>0 else 0)
        nodes = ["Sensors","BS/Edge",f"Network ({bearer})","TMS","Train DAS","Maintenance"]
        idx = {n:i for i,n in enumerate(nodes)}
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=nodes),
            link=dict(
                source=[idx["Sensors"], idx["BS/Edge"], idx[f"Network ({bearer})"], idx["TMS"], idx["TMS"]],
                target=[idx["BS/Edge"], idx[f"Network ({bearer})"], idx["TMS"], idx["Train DAS"], idx["Maintenance"]],
                value=[sensors_to_bs, bs_to_core, core_to_tms, tms_to_train, tms_to_maint],
                label=["uplink data+alerts","backhaul","to TMS","advisories/TSR/STOP","work orders"],
            )
        )])
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("RAW (bps)", f"{raw_bps:,}")
        c2.metric("Lane A (bps)", f"{laneA_bps:,}")
        c3.metric("Lane B (bps)", f"{laneB_bps:,}")
        c4.metric("Capacity (kbps)", f"{cap_bps//1000:,}")

# ---------------- Ops tab ----------------
with tab_ops:
    st.subheader("Maintenance & Incidents")
    # Update work orders
    t = st.session_state.t_idx
    for w in st.session_state.work_orders:
        if w["status"]=="Dispatched" and t >= w["eta_done_idx"]:
            w["status"]="Resolved"
    # Clear TSRs resolved by maintenance
    resolved_polys = {tuple(map(tuple, w["polygon"])) for w in st.session_state.work_orders if w["status"]=="Resolved"}
    st.session_state.tsr_polys = [p for p in st.session_state.tsr_polys if tuple(map(tuple, p["polygon"])) not in resolved_polys]

    # Table
    if st.session_state.work_orders:
        rows=[]
        for i,w in enumerate(st.session_state.work_orders):
            rows.append(dict(id=f"WO-{i+1:03d}", status=w["status"], created_s=w["created_idx"], eta_done_s=w["eta_done_idx"]))
        df=pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No active work orders yet. High-confidence buckling alerts will create them automatically.")

    # Incidents
    f = st.session_state.get("_frame", {})
    if f.get("crash", False):
        st.error("🚨 Incident: Train entered critical TSR region without receiving alert. **CRASH** triggered in simulation.")
    elif f.get("enforce_stop", False):
        st.warning("STOP order in effect: Train halted awaiting clearance / maintenance.")
