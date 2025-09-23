import json, zlib, math, time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans

# ---------- Page setup ----------
st.set_page_config(page_title="ENSURE-6G • Semantic Rail Demo (Map)", layout="wide")
st.title("ENSURE-6G: Semantic Communication • Rail Winter Demo (with Map)")
st.caption("Map view of train, base stations, and command center • Live KPIs • Safety (Lane A) vs Ops (Lane B) semantics • Coverage-aware link")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Route & Run")
    duration_min = st.slider("Simulated duration (minutes)", 5, 20, 10, 1)
    seed = st.number_input("Random seed", value=7, step=1)
    autoplay = st.checkbox("Auto-play on the Map", value=False)
    st.markdown("---")

    st.subheader("Winter conditions")
    ambient_c = st.slider("Ambient temp (°C)", -30, 10, -12)
    snowfall = st.select_slider("Snowfall", options=["none","light","moderate","heavy"], value="moderate")
    icing = st.select_slider("Icing risk", options=["low","medium","high"], value="high")
    st.markdown("---")

    st.subheader("Link profile")
    base_capacity_kbps = st.slider("Base capacity (kbps)", 128, 2048, 512, 64)
    burst_factor = st.slider("Capacity burst factor", 1.0, 3.0, 2.0, 0.1)
    good_loss_pct = st.slider("Loss in GOOD (%)", 0.0, 5.0, 0.5, 0.1)
    bad_loss_pct  = st.slider("Loss in BAD (%)", 5.0, 60.0, 20.0, 1.0)
    st.markdown("---")

    st.subheader("Semantics & Packets")
    strategy = st.radio("Transmit strategy", ["Raw", "Semantic", "Adaptive (prefer Semantic)"], index=2)
    k_codebook = st.select_slider("Codebook size (k-means z tokens)", options=[32, 64, 128, 256], value=128)
    use_cbor = st.checkbox("Encode Lane B events as CBOR (smaller than JSON)", value=True)
    ensure_events = st.checkbox("Guarantee visible events", value=True)

st.info("Scrub the time slider on the **Map** tab (or enable autoplay). Colors show link quality to the nearest base station.")

# ---------- Route & geography ----------
# Major points (approx lat/lon)
CITIES = [
    ("Sundsvall",   62.3908, 17.3069),
    ("Hudiksvall",  61.7280, 17.1040),
    ("Söderhamn",   61.3030, 17.0580),
    ("Gävle",       60.6749, 17.1413),
    ("Uppsala",     59.8586, 17.6389),
    ("Stockholm",   59.3293, 18.0686),
]
CMD_CENTER = ("Trafikledning Stockholm", 59.3326, 18.0649)

# Simple set of base stations placed along the corridor (mock but plausible)
BASE_STATIONS = [
    ("BS-Sundsvall", 62.38, 17.33, 16_000),  # radius (approx meters) for good coverage
    ("BS-Iggesund",  61.71, 17.11, 14_000),
    ("BS-Söderh",    61.31, 17.07, 14_000),
    ("BS-GävleN",    60.72, 17.17, 14_000),
    ("BS-Uppsala",   59.86, 17.64, 14_000),
    ("BS-Stockholm", 59.33, 18.07, 18_000),
]

# Interpolate the path into N waypoints
def interpolate_route(points: List[Tuple[str, float, float]], total_secs: int) -> pd.DataFrame:
    # Piecewise linear interpolation by equal time per leg (for simplicity)
    legs = []
    seg_secs = total_secs // (len(points)-1)
    for i in range(len(points)-1):
        n = seg_secs if i < len(points)-2 else total_secs - seg_secs*(len(points)-2)
        a, lat1, lon1 = points[i]
        b, lat2, lon2 = points[i+1]
        t = np.linspace(0, 1, max(n, 2))
        lat = lat1 + (lat2-lat1)*t
        lon = lon1 + (lon2-lon1)*t
        legs.append(pd.DataFrame({
            "t": np.arange(len(t)) + (i*seg_secs),
            "lat": lat,
            "lon": lon,
            "segment": [f"{a}→{b}"]*len(t)
        }))
    df = pd.concat(legs, ignore_index=True)
    df = df.iloc[:total_secs]  # ensure exactly total_secs rows
    return df

SECS = duration_min * 60
route_df = interpolate_route(CITIES, SECS)

# ---------- Helpers: distances / coverage ----------
R_EARTH = 6371000.0
def haversine_m(lat1, lon1, lat2, lon2):
    p = math.pi/180.0
    dlat = (lat2-lat1)*p; dlon = (lon2-lon1)*p
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2)
    return 2*R_EARTH*math.asin(math.sqrt(a))

def nearest_bs_quality(lat, lon):
    # Compute distance to all BS and select best
    best = None
    for name, blat, blon, r_good in BASE_STATIONS:
        d = haversine_m(lat, lon, blat, blon)
        # Piecewise: Good inside r_good; Patchy until 2.2x; Poor beyond
        if d <= r_good:
            q = "GOOD"
        elif d <= 2.2*r_good:
            q = "PATCHY"
        else:
            q = "POOR"
        if (best is None) or (["GOOD","PATCHY","POOR"].index(q) < ["GOOD","PATCHY","POOR"].index(best[2])):
            best = (name, d, q)
    return best  # (name, distance_m, quality)

# Capacity & loss from quality (plus bursts)
def link_params(quality: str, t: int) -> Tuple[int, float]:
    # capacity bits/s
    cap = base_capacity_kbps * 1000
    if quality == "GOOD":
        cap = int(cap * burst_factor)  # bursts when good
        loss = good_loss_pct / 100.0
    elif quality == "PATCHY":
        cap = int(cap * (0.6 + 0.2*math.sin(2*math.pi*t/30)))
        loss = min(0.4, (bad_loss_pct*0.5)/100.0)
    else:  # POOR
        cap = int(cap * 0.25)
        loss = bad_loss_pct / 100.0
    return max(0, cap), loss

# ---------- Sensor features & events (per-second) ----------
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

# Event logic with dwell/hysteresis
@dataclass
class Event: t: int; intent: str; slots: Dict[str, object]
events: List[Event] = []
RMS_HIGH=0.35; SLIP_HIGH=0.18; TEMP_HIGH=85.0; PANTO_VAR_HIGH=80.0; DWELL_S=120
rms_since=slip_since=temp_since=panto_since=None
for f in features:
    # ride
    rms_since = f.t if (f.rms>=RMS_HIGH and rms_since is None) else (rms_since if f.rms>=RMS_HIGH else None)
    if rms_since is not None and f.t - rms_since >= DWELL_S:
        events.append(Event(f.t,"ride_degradation",{"segment":f.segment,"rms":round(f.rms,3),"dwell_s":f.t-rms_since}))
        rms_since=None
    # adhesion
    slip_since = f.t if (f.slip_ratio>=SLIP_HIGH and slip_since is None) else (slip_since if f.slip_ratio>=SLIP_HIGH else None)
    if slip_since is not None and f.t - slip_since >= 60:
        # rough km est by path fraction
        km_total = 415  # Sundsvall->Stockholm rough
        km = round(km_total * (f.t/SECS), 1)
        events.append(Event(f.t,"low_adhesion_event",{"km":km,"slip_ratio":round(f.slip_ratio,3),"duration_s":f.t-slip_since}))
        slip_since=None
    # temp
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

# Codebook tokens on features (k-means)
X = np.array([[f.rms, f.crest, f.band_20_50, f.band_50_120, f.jerk] for f in features], dtype=np.float32)
kmeans = KMeans(n_clusters=int(k_codebook), n_init=5, random_state=int(seed)).fit(X)
tokens = kmeans.predict(X)
centroids = kmeans.cluster_centers_
lab = ["smooth"]*len(centroids)
order = np.argsort(centroids[:,0])
if len(centroids)>=4:
    lab[order[-1]]="rough-snow"; lab[order[-2]]="curve-rough"; lab[order[0]]="very-smooth"

# ---------- Packets ----------
import cbor2

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

def enc_laneB(ev: Event, z:int, zl:str, use_cbor: bool) -> bytes:
    body = {"i":ev.intent,"ts":int(ev.t),"s":ev.slots,"z":int(z),"zl":zl}
    b = cbor2.dumps(body) if use_cbor else json.dumps(body, separators=(",",":")).encode()
    crc = zlib.crc32(b).to_bytes(4,"big")
    return b + crc

# ---------- Per-second link + scheduler (simple, map-friendly) ----------
FS_ACCEL=200; FS_FORCE=500; BITS_PER_SAMPLE=16
raw_bits_per_sec = (FS_ACCEL*3*BITS_PER_SAMPLE + FS_FORCE*BITS_PER_SAMPLE + 4*BITS_PER_SAMPLE + 400)

laneA_bits = np.zeros(SECS, dtype=np.int64)
laneB_bits = np.zeros(SECS, dtype=np.int64)
cap_bits   = np.zeros(SECS, dtype=np.int64)
loss_pct   = np.zeros(SECS, dtype=np.float32)
quality    = ["GOOD"]*SECS
nearest_bs = [""]*SECS

laneA_sent = 0; laneA_drop = 0
laneB_sent = 0; laneB_drop = 0

seqA=0; seqB=0
laneB_table = []

# Pre-index events for faster lookup
events_by_t = {}
for e in events:
    events_by_t.setdefault(e.t, []).append(e)

for t in range(SECS):
    lat = route_df.loc[t, "lat"]; lon = route_df.loc[t, "lon"]
    bs_name, dist_m, q = nearest_bs_quality(lat, lon)
    nearest_bs[t] = bs_name; quality[t] = q
    cap, loss = link_params(q, t)
    cap_bits[t] = cap; loss_pct[t] = loss

    # Always try to send one Lane A telegram
    pa = laneA_adhesion_state(features[t]); ba = enc_laneA(pa, seqA); seqA += 1
    sizeA = len(ba)*8
    if cap >= sizeA and (rng.random() > loss):
        cap -= sizeA
        laneA_bits[t] += sizeA; laneA_sent += 1
    else:
        laneA_drop += 1

    # Strategy for raw background (consumes remaining cap)
    if strategy in ["Raw", "Adaptive (prefer Semantic)"]:
        add = raw_bits_per_sec
        if strategy == "Adaptive (prefer Semantic)":
            # near border quality, adaptively reduce
            if q != "GOOD": add = int(add*0.35)
        raw_use = min(add, cap)
        cap -= raw_use
        # we show raw_use in neither lane; it just reduces capacity

    # Lane B events at this second
    if t in events_by_t:
        for ev in events_by_t[t]:
            bb = enc_laneB(ev, tokens[t], lab[tokens[t]], use_cbor)
            sizeB = len(bb)*8
            if cap >= sizeB and (rng.random() > loss):
                cap -= sizeB
                laneB_bits[t] += sizeB; laneB_sent += 1
                # record delivered row for the table
                body = {"i":ev.intent,"ts":int(ev.t),"s":ev.slots,"z":int(tokens[t]),"zl":lab[tokens[t]]}
                laneB_table.append({
                    "t": t, "segment": features[t].segment, "intent": ev.intent,
                    "bytes": len(bb), "encoding": "CBOR" if use_cbor else "JSON",
                    "token_z": int(tokens[t]), "token_label": lab[tokens[t]],
                    "near_bs": bs_name, "link_quality": q
                })
            else:
                laneB_drop += 1

# ---------- Tabs ----------
tab_map, tab_packets, tab_plots, tab_about = st.tabs(["🗺️ Map", "📦 Packets", "📈 Plots & KPIs", "ℹ️ About"])

# ========================= Map TAB =========================
with tab_map:
    st.subheader("Map: Train • Base Stations • Command Center")
    colA, colB = st.columns([2,1])
    with colB:
        # Timeline control
        if "t_idx" not in st.session_state: st.session_state.t_idx = 0
        t_idx = st.slider("Time (s)", 0, SECS-1, value=st.session_state.t_idx, key="time_slider")
        st.session_state.t_idx = t_idx

        if autoplay:
            # Advance a few steps and rerun (gentle animation)
            new_t = min(SECS-1, t_idx+3)
            st.session_state.t_idx = new_t
            st.experimental_rerun()

        # Point-in-time KPIs
        st.metric("Segment", features[t_idx].segment)
        st.metric("Nearest base station", nearest_bs[t_idx])
        st.metric("Link quality", quality[t_idx])
        st.metric("Lane A sent (total)", laneA_sent)
        st.metric("Lane B sent (total)", laneB_sent)

        st.markdown("---")
        st.caption("Colors: Base station circles show nominal good coverage. Train dot color indicates link quality.")

    with colA:
        # PyDeck layers
        import pydeck as pdk

        # Route line
        route_path = [{"position":[route_df.loc[i, "lon"], route_df.loc[i, "lat"]], "t":i} for i in range(SECS)]
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path":[d["position"] for d in route_path[::max(1, SECS//200)]], "name":"Sundsvall→Stockholm"}],
            get_color=[60, 60, 120], width_scale=4, width_min_pixels=2,
        )

        # Base stations as discs
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

        # Train marker at t_idx
        qcol = {"GOOD":[0,170,0], "PATCHY":[255,165,0], "POOR":[200,0,0]}
        train_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{
                "lat": route_df.loc[t_idx,"lat"],
                "lon": route_df.loc[t_idx,"lon"],
                "color": qcol[quality[t_idx]]
            }]),
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

        view_state = pdk.ViewState(
            latitude=60.7, longitude=17.5, zoom=6.2, pitch=0
        )
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[path_layer, bs_layer, train_layer, cc_layer],
            tooltip={"text":"{name}"}
        ))
        st.caption("Move the time slider (or enable autoplay). The train dot changes color with link quality (GOOD/PATCHY/POOR).")

    st.markdown("---")
    st.subheader("Delivered Lane-B Events (table)")
    if laneB_table:
        st.dataframe(pd.DataFrame(laneB_table))
    else:
        st.info("No Lane-B events delivered under current conditions. Increase snowfall/icing or enable 'Guarantee visible events'.")

# ========================= Packets TAB =========================
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
    bB_json = enc_laneB(ev, tokens[example_t], lab[tokens[example_t]], use_cbor=False)
    bB_cbor = enc_laneB(ev, tokens[example_t], lab[tokens[example_t]], use_cbor=True)
    st.markdown("**Lane B (ops) example event**")
    st.code(json.dumps({"i":ev.intent,"ts":ev.t,"s":ev.slots,"z":int(tokens[example_t]),"zl":lab[tokens[example_t]]}, indent=2))
    st.write(f"JSON size: {len(bB_json)} bytes • CBOR size: {len(bB_cbor)} bytes (both include CRC32)")

# ========================= Plots & KPIs TAB =========================
with tab_plots:
    st.subheader("Throughput & Quality over Time")
    df_time = pd.DataFrame({
        "t": np.arange(SECS),
        "LaneA_bits": laneA_bits,
        "LaneB_bits": laneB_bits,
        "Capacity_bits": cap_bits,
        "Quality": quality
    })
    st.line_chart(df_time.set_index("t")[["Capacity_bits","LaneA_bits","LaneB_bits"]])
    q_counts = pd.Series(quality).value_counts()
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Lane A delivered", int((laneA_bits>0).sum()))
    with col2: st.metric("Lane B delivered", int((laneB_bits>0).sum()))
    with col3: st.metric("Avg capacity (kbps)", f"{cap_bits.mean()/1000:.1f}")
    with col4: st.metric("Quality time (GOOD/PATCHY/POOR)", f"{int(q_counts.get('GOOD',0))}/{int(q_counts.get('PATCHY',0))}/{int(q_counts.get('POOR',0))}")

    st.subheader("Bandwidth vs Raw (cumulative)")
    raw_total_bits = raw_bits_per_sec * SECS
    actual_total_bits = int(laneA_bits.sum() + laneB_bits.sum())
    saved = 1.0 - (actual_total_bits / max(raw_total_bits,1))
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Raw if streamed (MB)", f"{raw_total_bits/(8*1024*1024):.2f}")
    with c2: st.metric("Actual sent (MB)", f"{actual_total_bits/(8*1024*1024):.2f}")
    with c3: st.metric("Saved vs Raw", f"{100*saved:.1f}%")

# ========================= About TAB =========================
with tab_about:
    st.markdown("""
### What you’re seeing
- **Map**: The train moves Sundsvall→Stockholm; colored dot = link quality to the **nearest base station**.
- **Lane A (safety)**: Tiny, fixed-field adhesion telegram every second (CBOR+CRC). Always prioritized.
- **Lane B (ops)**: Event-driven semantics (e.g., low_adhesion, ride_degradation) + **codebook token `z`** (k-means).
- **Capacity & loss**: Derived from proximity-based quality (GOOD/PATCHY/POOR) + bursts; **Raw** mode reduces available capacity.

### Why it’s realistic & presentation-ready
- Uses **per-second sensor features** (RMS, band powers, crest, jerk, slip, pantograph variance, temp slope).
- **Events with dwell** mirror maintenance logic; **codebook tokens** are tiny semantic IDs.
- **Coverage explains everything visually**: when the dot turns **orange/red**, Lane B drops first; Lane A still squeezes through.

Tip: Start with *moderate snow*, *high icing*, *Adaptive* strategy, *base 512 kbps*, *burst×2*. Scrub the time slider and watch events deliver even as quality degrades.
""")
