import json, zlib, math, time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import cbor2
import pydeck as pdk

# ================= Page / Session =================
st.set_page_config(page_title="ENSURE-6G • Semantic Rail Demo (Live Map)", layout="wide")
st.title("ENSURE-6G: Semantic Communication — Live Rail Demo (Map + Animation)")
st.caption("Real map • Approximated railway path Sundsvall→Stockholm • Base stations with coverage • Train animation • Safety vs Ops semantics")

if "t_idx" not in st.session_state:
    st.session_state.t_idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

# ================= Sidebar Controls =================
with st.sidebar:
    st.header("Simulation")
    duration_min = st.slider("Duration (minutes)", 5, 30, 12, 1)
    seed = st.number_input("Random seed", value=11, step=1)
    ensure_events = st.checkbox("Guarantee visible events", value=True)

    st.markdown("---")
    st.subheader("Winter conditions")
    ambient_c = st.slider("Ambient temp (°C)", -30, 10, -12)
    snowfall = st.select_slider("Snowfall", ["none","light","moderate","heavy"], "moderate")
    icing = st.select_slider("Icing risk", ["low","medium","high"], "high")

    st.markdown("---")
    st.subheader("Link & Semantics")
    base_capacity_kbps = st.slider("Base capacity (kbps)", 128, 2048, 512, 64)
    burst_factor = st.slider("Burst factor when GOOD", 1.0, 3.0, 2.0, 0.1)
    good_loss_pct = st.slider("Loss in GOOD (%)", 0.0, 5.0, 0.5, 0.1)
    bad_loss_pct  = st.slider("Loss in POOR (%)", 5.0, 60.0, 20.0, 1.0)
    strategy = st.radio("Transmit strategy", ["Raw", "Semantic", "Adaptive (prefer Semantic)"], index=2)
    k_codebook = st.select_slider("Codebook size (k-means z)", [32, 64, 128, 256], 128)
    use_cbor = st.checkbox("Encode Lane B events as CBOR", value=True)

# ================= Geography: Real Map + Approximated Railway =================
# Handcrafted polyline hugging the coast / mainline (no external fetch)
# Points include Sundsvall, Iggesund/Hudiksvall area, Söderhamn, Gävle, Tierp, Uppsala, Märsta, Stockholm
RAIL_WAYPOINTS = [
    (62.3930, 17.3070),  # Sundsvall C
    (62.12, 17.15),
    (61.86, 17.14),
    (61.73, 17.11),      # Hudiksvall
    (61.56, 17.08),
    (61.39, 17.07),
    (61.30, 17.06),      # Söderhamn
    (61.07, 17.10),
    (60.85, 17.16),
    (60.67, 17.14),      # Gävle C
    (60.38, 17.33),
    (60.20, 17.45),
    (60.05, 17.52),
    (59.93, 17.61),
    (59.86, 17.64),      # Uppsala
    (59.75, 17.82),
    (59.66, 17.94),
    (59.61, 17.99),
    (59.55, 18.03),
    (59.48, 18.04),
    (59.42, 18.06),
    (59.37, 18.07),
    (59.3293, 18.0686),  # Stockholm C
]

# Base stations (mock but plausible locations along route) w/ coverage radius (m)
BASE_STATIONS = [
    ("BS-Sundsvall", 62.386, 17.325, 16000),
    ("BS-Njurunda",  62.275, 17.354, 14000),
    ("BS-Harmånger", 61.897, 17.170, 14000),
    ("BS-Hudiksvall",61.728, 17.103, 15000),
    ("BS-Söderhamn", 61.303, 17.058, 15000),
    ("BS-Axmar",     61.004, 17.190, 14000),
    ("BS-Gävle",     60.675, 17.141, 16000),
    ("BS-Tierp",     60.345, 17.513, 14000),
    ("BS-Skyttorp",  60.03,  17.58,  14000),
    ("BS-Uppsala",   59.858, 17.639, 16000),
    ("BS-Märsta",    59.62,  17.86,  15000),
    ("BS-Stockholm", 59.33,  18.07,  18000),
]

CMD_CENTER = ("Trafikledning Stockholm", 59.3326, 18.0649)

# ================= Utilities =================
R_EARTH = 6371000.0
def haversine_m(lat1, lon1, lat2, lon2):
    p = math.pi/180.0
    dlat = (lat2-lat1)*p; dlon = (lon2-lon1)*p
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2)
    return 2*R_EARTH*math.asin(math.sqrt(a))

def interpolate_polyline(waypoints: List[Tuple[float,float]], n_pts: int) -> pd.DataFrame:
    # Build cumulative distances then resample to n_pts
    lats = np.array([p[0] for p in waypoints], dtype=float)
    lons = np.array([p[1] for p in waypoints], dtype=float)
    seg_d = [0.0]
    for i in range(1, len(waypoints)):
        seg_d.append(haversine_m(lats[i-1], lons[i-1], lats[i], lons[i]))
    seg_d = np.array(seg_d)
    cum_d = np.cumsum(seg_d)
    total = cum_d[-1]
    tgt = np.linspace(0, total, n_pts)
    # piecewise linear on cumulative distance
    latp = []
    lonp = []
    idx = 0
    acc = cum_d[0]
    for d in tgt:
        while idx < len(cum_d)-1 and cum_d[idx] < d:
            idx += 1
        i0 = max(0, idx-1)
        i1 = idx
        d0 = cum_d[i0]
        d1 = cum_d[i1] if i1 < len(cum_d) else cum_d[-1]
        if d1 == d0:
            w = 0
        else:
            w = (d - d0) / (d1 - d0)
        latp.append(lats[i0] + (lats[i1] - lats[i0]) * w)
        lonp.append(lons[i0] + (lons[i1] - lons[i0]) * w)
    return pd.DataFrame({"lat": latp, "lon": lonp})

def nearest_bs_quality(lat, lon):
    best = None
    for name, blat, blon, r_m in BASE_STATIONS:
        d = haversine_m(lat, lon, blat, blon)
        if d <= r_m:
            q = "GOOD"
        elif d <= 2.2*r_m:
            q = "PATCHY"
        else:
            q = "POOR"
        rank = {"GOOD":0, "PATCHY":1, "POOR":2}[q]
        if best is None or rank < best[3]:
            best = (name, d, q, rank)
    return best[0], best[1], best[2]

def quality_to_capacity_loss(quality: str, t: int, base_kbps: int, burst: float):
    cap = base_kbps * 1000
    if quality == "GOOD":
        cap = int(cap * burst)
        loss = good_loss_pct / 100.0
    elif quality == "PATCHY":
        cap = int(cap * (0.6 + 0.2*math.sin(2*math.pi*t/30)))
        loss = min(0.4, (bad_loss_pct*0.5)/100.0)
    else:
        cap = int(cap * 0.25)
        loss = bad_loss_pct / 100.0
    return max(0, cap), loss

# ================= Simulation core (per second) =================
SECS = duration_min * 60
route_df = interpolate_polyline(RAIL_WAYPOINTS, SECS)

# per-second segment labels (rough)
SEG_NAMES = ["Sundsvall→Hudiksvall","Hudiksvall→Söderhamn","Söderhamn→Gävle","Gävle→Uppsala","Uppsala→Stockholm"]
seg_boundaries = np.linspace(0, SECS, len(SEG_NAMES)+1).astype(int)
seg_labels = np.empty(SECS, dtype=object)
for i in range(len(SEG_NAMES)):
    seg_labels[seg_boundaries[i]:seg_boundaries[i+1]] = SEG_NAMES[i]
route_df["segment"] = seg_labels

rng = np.random.default_rng(int(seed))
snow_factor = {"none":0.3, "light":0.7, "moderate":1.0, "heavy":1.5}[snowfall]
ice_factor  = {"low":0.6, "medium":1.0, "high":1.4}[icing]

@dataclass
class Feat:
    t: int; segment: str; rms: float; crest: float; band_20_50: float; band_50_120: float
    jerk: float; temp_peak: float; temp_ewma: float; temp_slope_cpm: float
    slip_ratio: float; wsp_count: int; panto_varN: float

def synth_feature_row(t: int) -> Feat:
    seg = route_df.loc[t, "segment"]
    base_rms = 0.18 + 0.05*np.sin(2*np.pi*t/180.0)
    rms = abs(base_rms*(1+0.6*(snow_factor-0.3)) + rng.normal(0, 0.02))
    band_20_50 = max(0.0, 0.12*(snow_factor) + rng.normal(0, 0.015))
    band_50_120 = max(0.0, 0.08*(snow_factor) + rng.normal(0, 0.015))
    crest = 2.2 + 0.2*(snow_factor) + rng.normal(0, 0.05)
    jerk = 0.6 + 0.3*(snow_factor) + rng.normal(0, 0.05)

    temp_base = 45 + 0.02*t + (0 if ambient_c < -15 else 2)
    temp_inst = temp_base + rng.normal(0, 0.3) + (3.0 if snow_factor>1.2 and (t%600>300) else 0.0)
    ewma = 0.8*temp_base + 0.2*temp_inst
    slope_cpm = (0.8 + 0.4*(ice_factor-1.0))
    temp_peak = temp_inst + max(0, rng.normal(0.5, 0.2))

    slip_ratio = max(0.0, rng.normal(0.05, 0.01) + 0.08*(snow_factor-0.3))
    wsp_count = int(max(0, rng.normal(5, 2) + 12*(snow_factor-0.3)))
    panto_varN = max(0.0, rng.normal(40, 8) + 25*(ice_factor-1.0))

    return Feat(t, seg, float(rms), float(crest), float(band_20_50), float(band_50_120),
                float(jerk), float(temp_peak), float(ewma), float(slope_cpm),
                float(slip_ratio), int(wsp_count), float(panto_varN))

features: List[Feat] = [synth_feature_row(t) for t in range(SECS)]

# Events with dwell/hysteresis
@dataclass
class Event: t: int; intent: str; slots: Dict[str, object]
RMS_HIGH=0.35; SLIP_HIGH=0.18; TEMP_HIGH=85.0; PANTO_VAR_HIGH=80.0; DWELL_S=120
events: List[Event] = []
rms_since=slip_since=temp_since=panto_since=None
for f in features:
    # ride degradation
    rms_since = f.t if (f.rms>=RMS_HIGH and rms_since is None) else (rms_since if f.rms>=RMS_HIGH else None)
    if rms_since is not None and f.t - rms_since >= DWELL_S:
        events.append(Event(f.t,"ride_degradation",{"segment":f.segment,"rms":round(f.rms,3),"dwell_s":f.t-rms_since}))
        rms_since=None
    # low adhesion
    slip_since = f.t if (f.slip_ratio>=SLIP_HIGH and slip_since is None) else (slip_since if f.slip_ratio>=SLIP_HIGH else None)
    if slip_since is not None and f.t - slip_since >= 60:
        km_total = 415
        km = round(km_total * (f.t/SECS), 1)
        events.append(Event(f.t,"low_adhesion_event",{"km":km,"slip_ratio":round(f.slip_ratio,3),"duration_s":f.t-slip_since}))
        slip_since=None
    # bearing overtemp
    temp_since = f.t if (f.temp_peak>=TEMP_HIGH and temp_since is None) else (temp_since if f.temp_peak>=TEMP_HIGH else None)
    if temp_since is not None and f.t - temp_since >= 180:
        events.append(Event(f.t,"bearing_overtemp",{"axle":"2L","peak_c":round(f.temp_peak,1),"dwell_s":f.t-temp_since}))
        temp_since=None
    # pantograph ice
    panto_since = f.t if (f.panto_varN>=PANTO_VAR_HIGH and ambient_c<=-8 and panto_since is None) else (panto_since if (f.panto_varN>=PANTO_VAR_HIGH and ambient_c<=-8) else None)
    if panto_since is not None and f.t - panto_since >= 90:
        events.append(Event(f.t,"pantograph_ice",{"varN":int(f.panto_varN),"temp_c":ambient_c}))
        panto_since=None

if ensure_events and len(events)==0:
    t0=min(SECS-10,300)
    events += [
        Event(t0,"ride_degradation",{"segment":features[t0].segment,"rms":0.42,"dwell_s":180}),
        Event(t0+20,"low_adhesion_event",{"km":200.3,"slip_ratio":0.22,"duration_s":90}),
        Event(t0+40,"pantograph_ice",{"varN":95,"temp_c":ambient_c}),
    ]

# Codebook tokens
X = np.array([[f.rms, f.crest, f.band_20_50, f.band_50_120, f.jerk] for f in features], dtype=np.float32)
kmeans = KMeans(n_clusters=int(k_codebook), n_init=5, random_state=int(seed)).fit(X)
tokens = kmeans.predict(X)
centroids = kmeans.cluster_centers_
labels = ["smooth"]*len(centroids)
order = np.argsort(centroids[:,0])
if len(centroids)>=4:
    labels[order[-1]]="rough-snow"; labels[order[-2]]="curve-rough"; labels[order[0]]="very-smooth"

# Packets
def laneA_adhesion_state(f: Feat) -> Dict[str,int]:
    mu_est = max(0.0, min(0.6, 0.6 - 0.5*f.slip_ratio))
    return {"mu_q7_9":int(mu_est*(1<<9)),
            "conf_pct":int(max(0,min(100,100-120*abs(0.2-f.slip_ratio)))),
            "slip":int(f.slip_ratio>=SLIP_HIGH),
            "wsp":int(min(255,f.wsp_count)),
            "valid_ms":500}
def enc_laneA(pkt: Dict[str,int], seq: int) -> bytes:
    body = dict(pkt); body.update({"seq":seq})
    b = cbor2.dumps(body); crc = zlib.crc32(b).to_bytes(4,"big")
    return b + crc
def enc_laneB(ev, z:int, zl:str, use_cbor: bool) -> bytes:
    body = {"i":ev.intent,"ts":int(ev.t),"s":ev.slots,"z":int(z),"zl":zl}
    b = cbor2.dumps(body) if use_cbor else json.dumps(body, separators=(",",":")).encode()
    crc = zlib.crc32(b).to_bytes(4,"big")
    return b + crc

# Channel model per second (capacity & loss by proximity)
FS_ACCEL=200; FS_FORCE=500; BITS_PER_SAMPLE=16
raw_bits_per_sec = (FS_ACCEL*3*BITS_PER_SAMPLE + FS_FORCE*BITS_PER_SAMPLE + 4*BITS_PER_SAMPLE + 400)

laneA_bits = np.zeros(SECS, dtype=np.int64)
laneB_bits = np.zeros(SECS, dtype=np.int64)
cap_bits   = np.zeros(SECS, dtype=np.int64)
quality    = np.empty(SECS, dtype=object)
near_bs    = np.empty(SECS, dtype=object)

events_by_t: Dict[int, List[Event]] = {}
for e in events:
    events_by_t.setdefault(e.t, []).append(e)

seqA=0; seqB=0
laneB_table = []
for t in range(SECS):
    lat, lon = route_df.loc[t,"lat"], route_df.loc[t,"lon"]
    bs_name, dist_m, q = nearest_bs_quality(lat, lon)
    near_bs[t] = bs_name; quality[t] = q
    cap, loss = quality_to_capacity_loss(q, t, base_capacity_kbps, burst_factor)
    cap_bits[t] = cap

    # Lane A every second
    pa = laneA_adhesion_state(features[t]); ba = enc_laneA(pa, seqA); seqA += 1
    szA = len(ba)*8
    if cap >= szA and (rng.random() > loss):
        cap -= szA
        laneA_bits[t] += szA
    # Raw background
    if strategy in ["Raw","Adaptive (prefer Semantic)"]:
        add = raw_bits_per_sec
        if strategy=="Adaptive (prefer Semantic)" and q!="GOOD":
            add = int(add*0.35)
        cap = max(0, cap - add)
    # Lane B events (if any)
    if t in events_by_t:
        for ev in events_by_t[t]:
            bb = enc_laneB(ev, tokens[t], labels[tokens[t]], use_cbor)
            szB = len(bb)*8
            if cap >= szB and (rng.random() > loss):
                cap -= szB
                laneB_bits[t] += szB
                laneB_table.append({
                    "t": t, "segment": features[t].segment, "intent": ev.intent,
                    "bytes": len(bb), "encoding": "CBOR" if use_cbor else "JSON",
                    "token_z": int(tokens[t]), "token_label": labels[tokens[t]],
                    "near_bs": bs_name, "link_quality": q
                })

# ================= UI Layout: Map + Controls =================
tab_map, tab_packets, tab_metrics = st.tabs(["🗺️ Map (Animate)", "📦 Packets", "📈 Metrics"])

with tab_map:
    colL, colR = st.columns([2,1])
    with colR:
        st.subheader("Playback")
        # Play / Pause buttons
        cols = st.columns(2)
        if cols[0].button("▶ Simulate", use_container_width=True):
            st.session_state.playing = True
        if cols[1].button("⏸ Pause", use_container_width=True):
            st.session_state.playing = False

        # Time slider
        t_idx = st.slider("Time (s)", 0, SECS-1, value=st.session_state.t_idx, key="time_slider", disabled=st.session_state.playing)
        st.session_state.t_idx = t_idx

        # KPIs at current time
        st.metric("Segment", route_df.loc[t_idx,"segment"])
        st.metric("Nearest BS", str(near_bs[t_idx]))
        st.metric("Link quality", str(quality[t_idx]))
        st.metric("Lane A bits this second", int(laneA_bits[t_idx]))
        st.metric("Lane B bits this second", int(laneB_bits[t_idx]))
        st.metric("Capacity (kbps)", int(cap_bits[t_idx]/1000))

    with colL:
        # Build layers
        # Railway path (downsample for rendering if long)
        step = max(1, SECS//300)
        path_coords = [[route_df.loc[i,"lon"], route_df.loc[i,"lat"]] for i in range(0, SECS, step)]
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": path_coords, "name":"Sundsvall→Stockholm"}],
            get_color=[60,60,120],
            width_scale=4, width_min_pixels=2
        )

        # Base stations as coverage discs
        bs_df = pd.DataFrame(BASE_STATIONS, columns=["name","lat","lon","r_m"])
        bs_layer = pdk.Layer(
            "ScatterplotLayer",
            data=bs_df,
            get_position="[lon, lat]",
            get_radius="r_m",
            get_fill_color="[0, 150, 0, 40]",
            stroked=True, get_line_color=[0,150,0], line_width_min_pixels=1,
            pickable=True,
        )

        # Train dot with color by quality
        qcol = {"GOOD":[0,170,0], "PATCHY":[255,165,0], "POOR":[200,0,0]}
        cur = pd.DataFrame([{"lat":route_df.loc[st.session_state.t_idx,"lat"],
                             "lon":route_df.loc[st.session_state.t_idx,"lon"],
                             "color": qcol[quality[st.session_state.t_idx]]}])
        train_layer = pdk.Layer(
            "ScatterplotLayer",
            data=cur,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=1200,
            stroked=True, get_line_color=[0,0,0], line_width_min_pixels=1,
            pickable=False
        )

        # Command center marker
        cc_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": CMD_CENTER[1], "lon": CMD_CENTER[2]}]),
            get_position="[lon, lat]",
            get_fill_color=[30,30,30],
            get_radius=800,
            stroked=True, get_line_color=[255,255,255], line_width_min_pixels=1,
            pickable=False
        )

        view_state = pdk.ViewState(latitude=60.7, longitude=17.5, zoom=6.2, pitch=0)
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[path_layer, bs_layer, train_layer, cc_layer],
            tooltip={"text":"{name}"}
        ))
        st.caption("Click ▶ Simulate to animate. The train dot changes color with link quality to the nearest base station.")

    # Animation loop (simple; advances and reruns)
    if st.session_state.playing:
        next_t = st.session_state.t_idx + 1
        if next_t >= SECS:
            st.session_state.playing = False
        else:
            st.session_state.t_idx = next_t
            time.sleep(0.08)  # ~12.5 fps feel
            st.experimental_rerun()

with tab_packets:
    st.subheader("Packet Inspector")
    example_t = min(SECS-1, max(0, SECS//3))
    # Lane A
    pktA = laneA_adhesion_state(features[example_t])
    bA = enc_laneA(pktA, 123)
    st.markdown("**Lane A (safety) adhesion_state**")
    st.code(json.dumps(pktA, indent=2))
    st.write(f"Encoded size: {len(bA)} bytes (CBOR + CRC32)")

    # Lane B
    ev = None
    for e in events:
        if e.t >= example_t: ev=e; break
    if ev is None:
        ev = Event(example_t,"ride_degradation",{"segment":features[example_t].segment,"rms":0.41,"dwell_s":160})
    bB_json = enc_laneB(ev, tokens[example_t], labels[tokens[example_t]], use_cbor=False)
    bB_cbor = enc_laneB(ev, tokens[example_t], labels[tokens[example_t]], use_cbor=True)
    st.markdown("**Lane B (ops) example event**")
    st.code(json.dumps({"i":ev.intent,"ts":ev.t,"s":ev.slots,"z":int(tokens[example_t]),"zl":labels[tokens[example_t]]}, indent=2))
    st.write(f"JSON size: {len(bB_json)} bytes • CBOR size: {len(bB_cbor)} bytes (both include CRC32)")

    st.subheader("Delivered Lane-B events (table snapshot)")
    if len(laneB_table):
        st.dataframe(pd.DataFrame(laneB_table))
    else:
        st.info("No Lane-B events delivered under current settings. Increase snowfall/icing, or toggle Guarantee events.")

with tab_metrics:
    st.subheader("Link & Semantics Metrics")
    df_time = pd.DataFrame({
        "t": np.arange(SECS),
        "LaneA_bits": laneA_bits,
        "LaneB_bits": laneB_bits,
        "Capacity_bits": cap_bits,
        "Quality": quality
    })
    st.line_chart(df_time.set_index("t")[["Capacity_bits","LaneA_bits","LaneB_bits"]])
    raw_total_bits = raw_bits_per_sec * SECS
    actual_total_bits = int(laneA_bits.sum() + laneB_bits.sum())
    saved = 1.0 - (actual_total_bits / max(raw_total_bits,1))
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Raw if streamed (MB)", f"{raw_total_bits/(8*1024*1024):.2f}")
    with c2: st.metric("Actual sent (MB)", f"{actual_total_bits/(8*1024*1024):.2f}")
    with c3: st.metric("Saved vs Raw", f"{100*saved:.1f}%")

st.markdown("---")
st.caption("This demo uses a real base map (Mapbox) and an approximated railway polyline. Base stations are placed along the corridor with plausible coverage radii. Train animation runs locally for smooth presentations.")
