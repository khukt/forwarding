import json, zlib, math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import cbor2
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh
import altair as alt

# ========= Page / session =========
st.set_page_config(page_title="ENSURE-6G • Raw vs Semantic vs Hybrid", layout="wide")
st.title("ENSURE-6G: Raw vs Semantic vs Hybrid — Live Rail Demo (OpenStreetMap)")
st.caption("OpenStreetMap • Sundsvall→Stockholm • Base stations + coverage • Moving train • Lane A (safety) vs Lane B (ops) • Strategy comparison")

if "t_idx" not in st.session_state: st.session_state.t_idx = 0
if "playing" not in st.session_state: st.session_state.playing = False

# ========= Sidebar =========
with st.sidebar:
    st.header("Scenario")
    duration_min = st.slider("Duration (minutes)", 5, 20, 10, 1)
    seed = st.number_input("Random seed", value=11, step=1)
    ensure_events = st.checkbox("Guarantee visible events", value=True)

    st.markdown("---")
    st.subheader("Winter conditions")
    ambient_c = st.slider("Ambient temp (°C)", -30, 10, -12)
    snowfall = st.select_slider("Snowfall", ["none","light","moderate","heavy"], "moderate")
    icing = st.select_slider("Icing risk", ["low","medium","high"], "high")

    st.markdown("---")
    st.subheader("Link model")
    base_capacity_kbps = st.slider("Base capacity (kbps)", 128, 2048, 512, 64)
    burst_factor = st.slider("Burst when GOOD", 1.0, 3.0, 2.0, 0.1)
    good_loss_pct = st.slider("Loss in GOOD (%)", 0.0, 5.0, 0.5, 0.1)
    bad_loss_pct  = st.slider("Loss in POOR (%)", 5.0, 60.0, 20.0, 1.0)

    st.markdown("---")
    st.subheader("Semantics")
    k_codebook = st.select_slider("Codebook size (k-means z)", [32, 64, 128, 256], 128)
    use_cbor = st.checkbox("Encode Lane B as CBOR (smaller than JSON)", value=True)

    st.markdown("---")
    st.subheader("Live map strategy")
    live_strategy = st.radio("Map playback uses…", ["Raw only", "Hybrid (Adaptive)", "Semantic only"], index=1)

# ========= Geography =========
RAIL_WAYPOINTS = [
    (62.3930, 17.3070), (62.12, 17.15), (61.86, 17.14), (61.73, 17.11),
    (61.56, 17.08), (61.39, 17.07), (61.30, 17.06), (61.07, 17.10),
    (60.85, 17.16), (60.67, 17.14), (60.38, 17.33), (60.20, 17.45),
    (60.05, 17.52), (59.93, 17.61), (59.86, 17.64), (59.75, 17.82),
    (59.66, 17.94), (59.61, 17.99), (59.55, 18.03), (59.48, 18.04),
    (59.42, 18.06), (59.37, 18.07), (59.3293, 18.0686),
]
BASE_STATIONS = [
    ("BS-Sundsvall", 62.386, 17.325, 16000), ("BS-Njurunda",62.275,17.354,14000),
    ("BS-Harmånger",61.897,17.170,14000), ("BS-Hudiksvall",61.728,17.103,15000),
    ("BS-Söderhamn",61.303,17.058,15000), ("BS-Axmar",61.004,17.190,14000),
    ("BS-Gävle",60.675,17.141,16000), ("BS-Tierp",60.345,17.513,14000),
    ("BS-Skyttorp",60.03,17.58,14000), ("BS-Uppsala",59.858,17.639,16000),
    ("BS-Märsta",59.62,17.86,15000), ("BS-Stockholm",59.33,18.07,18000),
]
R_EARTH = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    p = math.pi/180.0
    dlat = (lat2-lat1)*p; dlon = (lon2-lon1)*p
    a = (math.sin(dlat/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2)
    return 2*R_EARTH*math.asin(math.sqrt(a))

def interpolate_polyline(waypoints, n_pts):
    lats = np.array([p[0] for p in waypoints]); lons = np.array([p[1] for p in waypoints])
    seg = [0.0]
    for i in range(1,len(waypoints)): seg.append(haversine_m(lats[i-1],lons[i-1],lats[i],lons[i]))
    seg = np.array(seg); cum = np.cumsum(seg); total=cum[-1]
    tgt = np.linspace(0,total,n_pts)
    latp=[]; lonp=[]; idx=0
    for d in tgt:
        while idx < len(cum)-1 and cum[idx] < d: idx+=1
        i0=max(0,idx-1); i1=idx; d0=cum[i0]; d1=cum[i1] if i1<len(cum) else cum[-1]
        w=0 if d1==d0 else (d-d0)/(d1-d0)
        latp.append(lats[i0]+(lats[i1]-lats[i0])*w)
        lonp.append(lons[i0]+(lons[i1]-lons[i0])*w)
    return pd.DataFrame({"lat":latp,"lon":lonp})

SECS = duration_min*60
route_df = interpolate_polyline(RAIL_WAYPOINTS, SECS)
SEG_NAMES = ["Sundsvall→Hudiksvall","Hudiksvall→Söderhamn","Söderhamn→Gävle","Gävle→Uppsala","Uppsala→Stockholm"]
seg_bounds = np.linspace(0, SECS, len(SEG_NAMES)+1).astype(int)
seg_labels = np.empty(SECS, dtype=object)
for i in range(len(SEG_NAMES)): seg_labels[seg_bounds[i]:seg_bounds[i+1]] = SEG_NAMES[i]
route_df["segment"]=seg_labels

def nearest_bs_quality(lat, lon):
    best=None
    for name, blat, blon, r in BASE_STATIONS:
        d=haversine_m(lat,lon,blat,blon)
        q="GOOD" if d<=r else ("PATCHY" if d<=2.2*r else "POOR")
        rank={"GOOD":0,"PATCHY":1,"POOR":2}[q]
        if best is None or rank<best[3]: best=(name,d,q,rank)
    return best[0], best[1], best[2]

def cap_loss(quality, t):
    cap = base_capacity_kbps*1000
    if quality=="GOOD":
        cap=int(cap*burst_factor); loss=good_loss_pct/100.0
    elif quality=="PATCHY":
        cap=int(cap*(0.6+0.2*math.sin(2*math.pi*t/30))); loss=min(0.4,(bad_loss_pct*0.5)/100.0)
        cap=max(int(cap*0.9), 1)
    else:
        cap=int(cap*0.25); loss=bad_loss_pct/100.0
    return max(0,cap), loss

# ========= Features & events =========
rng = np.random.default_rng(int(seed))
snow_factor = {"none":0.3,"light":0.7,"moderate":1.0,"heavy":1.5}[snowfall]
ice_factor  = {"low":0.6,"medium":1.0,"high":1.4}[icing]

@dataclass
class Feat:
    t:int; segment:str; rms:float; crest:float; band_20_50:float; band_50_120:float
    jerk:float; temp_peak:float; temp_ewma:float; temp_slope_cpm:float
    slip_ratio:float; wsp_count:int; panto_varN:float

def synth_feature(t:int)->Feat:
    seg=route_df.loc[t,"segment"]
    base_rms=0.18+0.05*np.sin(2*np.pi*t/180)
    rms=abs(base_rms*(1+0.6*(snow_factor-0.3))+rng.normal(0,0.02))
    band_20_50=max(0.0,0.12*snow_factor+rng.normal(0,0.015))
    band_50_120=max(0.0,0.08*snow_factor+rng.normal(0,0.015))
    crest=2.2+0.2*snow_factor+rng.normal(0,0.05); jerk=0.6+0.3*snow_factor+rng.normal(0,0.05)
    temp_base=45+0.02*t+(0 if ambient_c<-15 else 2)
    temp_inst=temp_base+rng.normal(0,0.3)+(3.0 if snow_factor>1.2 and (t%600>300) else 0.0)
    ewma=0.8*temp_base+0.2*temp_inst; slope_cpm=(0.8+0.4*(ice_factor-1.0))
    temp_peak=temp_inst+max(0,rng.normal(0.5,0.2))
    slip_ratio=max(0.0,rng.normal(0.05,0.01)+0.08*(snow_factor-0.3))
    wsp_count=int(max(0,rng.normal(5,2)+12*(snow_factor-0.3)))
    panto_varN=max(0.0,rng.normal(40,8)+25*(ice_factor-1.0))
    return Feat(t,seg,float(rms),float(crest),float(band_20_50),float(band_50_120),
                float(jerk),float(temp_peak),float(ewma),float(slope_cpm),
                float(slip_ratio),int(wsp_count),float(panto_varN))

features=[synth_feature(t) for t in range(SECS)]

@dataclass
class Event: t:int; intent:str; slots:Dict[str,object]
RMS_HIGH=0.35; SLIP_HIGH=0.18; TEMP_HIGH=85.0; PANTO_VAR_HIGH=80.0; DWELL_S=120
events=[]; rms_since=slip_since=temp_since=panto_since=None
for f in features:
    rms_since = f.t if (f.rms>=RMS_HIGH and rms_since is None) else (rms_since if f.rms>=RMS_HIGH else None)
    if rms_since is not None and f.t-rms_since>=DWELL_S:
        events.append(Event(f.t,"ride_degradation",{"segment":f.segment,"rms":round(f.rms,3),"dwell_s":f.t-rms_since})); rms_since=None
    slip_since = f.t if (f.slip_ratio>=SLIP_HIGH and slip_since is None) else (slip_since if f.slip_ratio>=SLIP_HIGH else None)
    if slip_since is not None and f.t-slip_since>=60:
        km=round(415*(f.t/SECS),1)
        events.append(Event(f.t,"low_adhesion_event",{"km":km,"slip_ratio":round(f.slip_ratio,3),"duration_s":f.t-slip_since})); slip_since=None
    temp_since = f.t if (f.temp_peak>=TEMP_HIGH and temp_since is None) else (temp_since if f.temp_peak>=TEMP_HIGH else None)
    if temp_since is not None and f.t-temp_since>=180:
        events.append(Event(f.t,"bearing_overtemp",{"axle":"2L","peak_c":round(f.temp_peak,1),"dwell_s":f.t-temp_since})); temp_since=None
    panto_since = f.t if (f.panto_varN>=PANTO_VAR_HIGH and ambient_c<=-8 and panto_since is None) else (panto_since if (f.panto_varN>=PANTO_VAR_HIGH and ambient_c<=-8) else None)
    if panto_since is not None and f.t-panto_since>=90:
        events.append(Event(f.t,"pantograph_ice",{"varN":int(f.panto_varN),"temp_c":ambient_c})); panto_since=None

if ensure_events and len(events)==0:
    t0=min(SECS-10,300)
    events += [Event(t0,"ride_degradation",{"segment":features[t0].segment,"rms":0.42,"dwell_s":180}),
               Event(t0+20,"low_adhesion_event",{"km":200.3,"slip_ratio":0.22,"duration_s":90}),
               Event(t0+40,"pantograph_ice",{"varN":95,"temp_c":ambient_c})]

# ========= Semantics: codebook =========
X=np.array([[f.rms,f.crest,f.band_20_50,f.band_50_120,f.jerk] for f in features],dtype=np.float32)
kmeans=KMeans(n_clusters=int(k_codebook),n_init=5,random_state=int(seed)).fit(X)
tokens=kmeans.predict(X); C=kmeans.cluster_centers_; labels=["smooth"]*len(C)
order=np.argsort(C[:,0])
if len(C)>=4: labels[order[-1]]="rough-snow"; labels[order[-2]]="curve-rough"; labels[order[0]]="very-smooth"

# ========= Packets =========
def laneA_adhesion_state(f:Feat)->Dict[str,int]:
    mu=max(0.0,min(0.6,0.6-0.5*f.slip_ratio))
    return {"mu_q7_9":int(mu*(1<<9)),"conf_pct":int(max(0,min(100,100-120*abs(0.2-f.slip_ratio)))),
            "slip":int(f.slip_ratio>=SLIP_HIGH),"wsp":int(min(255,f.wsp_count)),"valid_ms":500}
def enc_laneA(pkt,seq): b=cbor2.dumps({**pkt,"seq":seq}); return b+zlib.crc32(b).to_bytes(4,"big")
def enc_laneB(ev,z,zl,use_cbor): 
    body={"i":ev.intent,"ts":int(ev.t),"s":ev.slots,"z":int(z),"zl":zl}
    b=cbor2.dumps(body) if use_cbor else json.dumps(body,separators=(",",":")).encode()
    return b+zlib.crc32(b).to_bytes(4,"big")

# ========= Channel + per-second simulator =========
FS_ACCEL=200; FS_FORCE=500; BITS_PER_SAMPLE=16
RAW_BITS_PER_SEC = (FS_ACCEL*3*BITS_PER_SAMPLE + FS_FORCE*BITS_PER_SAMPLE + 4*BITS_PER_SAMPLE + 400)

def run_strategy(strategy_name:str):
    """Run whole trip and return arrays + KPIs. Includes raw_bits in totals."""
    np.random.seed(int(seed))
    laneA_bits=np.zeros(SECS,dtype=np.int64)
    laneB_bits=np.zeros(SECS,dtype=np.int64)
    raw_bits  =np.zeros(SECS,dtype=np.int64)  # NEW: track raw stream load
    cap_bits=np.zeros(SECS,dtype=np.int64)
    quality=np.empty(SECS,dtype=object)
    near_bs=np.empty(SECS,dtype=object)
    ev_by_t={}
    for e in events: ev_by_t.setdefault(e.t,[]).append(e)
    seqA=0
    for t in range(SECS):
        lat,lon=route_df.loc[t,"lat"],route_df.loc[t,"lon"]
        bs_name,_,q=nearest_bs_quality(lat,lon); near_bs[t]=bs_name; quality[t]=q
        cap,loss=cap_loss(q,t); cap_bits[t]=cap

        # Lane A (always)
        pa=laneA_adhesion_state(features[t]); A=enc_laneA(pa,seqA); seqA+=1
        szA=len(A)*8
        if cap>=szA and (np.random.random()>loss):
            cap-=szA; laneA_bits[t]+=szA

        # Raw background (now counted)
        if strategy_name=="Raw only":
            raw_use=min(RAW_BITS_PER_SEC, int(cap*0.95))
            cap=max(0,cap-raw_use); raw_bits[t]=raw_use
        elif strategy_name=="Hybrid (Adaptive)":
            desired = RAW_BITS_PER_SEC if q=="GOOD" else int(RAW_BITS_PER_SEC*0.35)
            raw_use = min(desired, int(cap*0.60))  # leave headroom for events
            cap=max(0,cap-raw_use); raw_bits[t]=raw_use
        else:  # "Semantic only"
            raw_bits[t]=0

        # Lane B events
        if t in ev_by_t:
            for ev in ev_by_t[t]:
                B=enc_laneB(ev,tokens[t],labels[tokens[t]],use_cbor); szB=len(B)*8
                if cap>=szB and (np.random.random()>loss):
                    cap-=szB; laneB_bits[t]+=szB

    # KPIs
    laneA_pkts_delivered = int((laneA_bits > 0).sum())
    laneB_pkts_delivered = int((laneB_bits > 0).sum())
    laneA_pkts_attempted = SECS
    laneB_pkts_attempted = len(events)
    total_bits = int(laneA_bits.sum() + laneB_bits.sum() + raw_bits.sum())

    return {
        "laneA_bits": laneA_bits,
        "laneB_bits": laneB_bits,
        "raw_bits": raw_bits,               # return raw
        "cap_bits": cap_bits,
        "quality": quality,
        "near_bs": near_bs,
        "kpis": {
            "laneA_deliv": laneA_pkts_delivered,
            "laneA_attempt": laneA_pkts_attempted,
            "laneB_deliv": laneB_pkts_delivered,
            "laneB_attempt": laneB_pkts_attempted,
            "total_bits": total_bits,
        }
    }

# Run strategies & choose live map set
res_raw     = run_strategy("Raw only")
res_hybrid  = run_strategy("Hybrid (Adaptive)")
res_sem     = run_strategy("Semantic only")
res_map = {"Raw only":res_raw, "Hybrid (Adaptive)":res_hybrid, "Semantic only":res_sem}[live_strategy]

# Baseline for "Saved vs Raw": use the actual Raw-only run (includes Lane A + Raw + delivered events)
BASELINE_RAW_BITS = max(1, res_raw["kpis"]["total_bits"])

# ========= Tabs =========
tab_map, tab_packets, tab_metrics, tab_compare, tab_why = st.tabs(
    ["🗺️ Map (Animate, OSM)", "📦 Packets", "📈 Per-lane Metrics", "📊 Strategy Comparison", "❓ When to use Raw vs Semantic"]
)

# ===================== MAP =====================
with tab_map:
    colL, colR = st.columns([2,1])
    with colR:
        st.subheader("Playback")

        c1, c2 = st.columns(2)
        if c1.button("▶ Simulate", use_container_width=True): st.session_state.playing = True
        if c2.button("⏸ Pause", use_container_width=True):   st.session_state.playing = False

        # --- AUTOPLAY: fixed interval, +1 second per tick ---
        if st.session_state.playing:
            st_autorefresh(interval=700, key="autoplay_tick_fixed")
            st.session_state.t_idx = min(st.session_state.t_idx + 1, SECS - 1)
            if st.session_state.t_idx >= SECS - 1:
                st.session_state.playing = False
            st.slider("Time (s)", 0, SECS - 1, value=st.session_state.t_idx, key="time_slider", disabled=True)
        else:
            t_idx = st.slider("Time (s)", 0, SECS - 1, value=st.session_state.t_idx, key="time_slider", disabled=False)
            st.session_state.t_idx = t_idx

        t_idx = st.session_state.t_idx
        st.metric("Strategy (map)", live_strategy)
        st.metric("Segment", route_df.loc[t_idx,"segment"])
        st.metric("Nearest BS", str(res_map["near_bs"][t_idx]))
        st.metric("Link quality", str(res_map["quality"][t_idx]))
        st.metric("Lane A bits this second", int(res_map["laneA_bits"][t_idx]))
        st.metric("Lane B bits this second", int(res_map["laneB_bits"][t_idx]))
        st.metric("Raw bits this second", int(res_map["raw_bits"][t_idx]))
        st.metric("Capacity (kbps)", int(res_map["cap_bits"][t_idx]/1000))

    with colL:
        tile_layer = pdk.Layer("TileLayer",
                               data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                               min_zoom=0, max_zoom=19, tile_size=256)
        step = max(1, SECS//300)
        path_coords = [[route_df.loc[i,"lon"], route_df.loc[i,"lat"]] for i in range(0, SECS, step)]
        path_layer = pdk.Layer("PathLayer",
                               data=[{"path": path_coords, "name":"Sundsvall→Stockholm"}],
                               get_color=[60,60,160], width_scale=4, width_min_pixels=2)
        bs_df = pd.DataFrame(BASE_STATIONS, columns=["name","lat","lon","r_m"])
        bs_layer = pdk.Layer("ScatterplotLayer", data=bs_df, get_position="[lon, lat]",
                             get_radius="r_m", get_fill_color="[0,150,0,40]",
                             stroked=True, get_line_color=[0,150,0], line_width_min_pixels=1, pickable=True)
        qcol = {"GOOD":[0,170,0], "PATCHY":[255,165,0], "POOR":[200,0,0]}
        cur = pd.DataFrame([{
            "lat": route_df.loc[t_idx,"lat"],
            "lon": route_df.loc[t_idx,"lon"],
            "icon_data": {"url":"https://img.icons8.com/emoji/48/train-emoji.png","width":128,"height":128,"anchorY":128},
            "color": qcol[res_map["quality"][t_idx]],
        }])
        train_icon_layer = pdk.Layer("IconLayer", data=cur, get_position='[lon, lat]', get_icon='icon_data', get_size=4, size_scale=15)
        halo_layer = pdk.Layer("ScatterplotLayer", data=cur, get_position='[lon, lat]', get_fill_color='color',
                               get_radius=5000, stroked=True, get_line_color=[0,0,0], line_width_min_pixels=1)
        view_state = pdk.ViewState(latitude=60.7, longitude=17.5, zoom=6.2, pitch=0)
        st.pydeck_chart(pdk.Deck(layers=[tile_layer, path_layer, bs_layer, halo_layer, train_icon_layer],
                                 initial_view_state=view_state, map_style=None, tooltip={"text":"{name}"}))
        st.caption("OSM tiles. Train halo color shows link quality to nearest BS. Click ▶ to animate; slider scrubs when paused.")

# ===================== PACKETS =====================
with tab_packets:
    st.subheader("Packet examples")
    example_t = min(SECS-1, max(0, SECS//3))
    # Lane A
    pktA = laneA_adhesion_state(features[example_t]); bA = enc_laneA(pktA, 123)
    st.markdown("**Lane A (safety) adhesion_state**"); st.code(json.dumps(pktA, indent=2))
    st.write(f"Encoded size: {len(bA)} bytes (CBOR + CRC32)")
    # Lane B
    ev = next((e for e in events if e.t >= example_t), None)
    if ev is None: ev = Event(example_t,"ride_degradation",{"segment":features[example_t].segment,"rms":0.41,"dwell_s":160})
    bB_json = enc_laneB(ev, tokens[example_t], labels[tokens[example_t]], use_cbor=False)
    bB_cbor = enc_laneB(ev, tokens[example_t], labels[tokens[example_t]], use_cbor=True)
    st.markdown("**Lane B (ops) example event**")
    st.code(json.dumps({"i":ev.intent,"ts":ev.t,"s":ev.slots,"z":int(tokens[example_t]),"zl":labels[tokens[example_t]]}, indent=2))
    st.write(f"JSON size: {len(bB_json)} bytes • CBOR size: {len(bB_cbor)} bytes (both include CRC32)")

# ===================== PER-LANE METRICS (for live strategy) =====================
with tab_metrics:
    st.subheader(f"Per-lane metrics — {live_strategy}")
    df = pd.DataFrame({"t":np.arange(SECS),
                       "Capacity_bps": res_map["cap_bits"],
                       "Raw_bps": res_map["raw_bits"],
                       "LaneA_bps": res_map["laneA_bits"],
                       "LaneB_bps": res_map["laneB_bits"]})
    base = alt.Chart(df).encode(x="t:Q")
    cap_line  = base.mark_line(color="#2ca02c").encode(y=alt.Y("Capacity_bps:Q", title="Capacity / Raw (bps)"))
    raw_line  = base.mark_line(color="#9467bd").encode(y=alt.Y("Raw_bps:Q"))
    # Right axis for lanes
    right_scale = alt.Scale(domain=[0, max(1, int(max(df["LaneA_bps"].max(), df["LaneB_bps"].max())*1.4))])
    laneA_line = base.mark_line(color="#1f77b4").encode(y=alt.Y("LaneA_bps:Q", axis=alt.Axis(title="Lane A/B (bps)", orient="right"), scale=right_scale))
    laneB_line = base.mark_line(color="#d62728").encode(y=alt.Y("LaneB_bps:Q", axis=alt.Axis(title=None, orient="right"), scale=right_scale))
    st.altair_chart(alt.layer(cap_line, raw_line, laneA_line, laneB_line).resolve_scale(y="independent").properties(height=360), use_container_width=True)

    # KPIs for the selected strategy
    total_bits = res_map["kpis"]["total_bits"]
    saved = 1.0 - (total_bits / BASELINE_RAW_BITS)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Lane A delivered / attempted", f'{res_map["kpis"]["laneA_deliv"]}/{res_map["kpis"]["laneA_attempt"]}')
    with c2: st.metric("Lane B delivered / attempted", f'{res_map["kpis"]["laneB_deliv"]}/{res_map["kpis"]["laneB_attempt"]}')
    with c3: st.metric("Actual sent (MB)", f"{total_bits/(8*1024*1024):.2f}")
    with c4: st.metric("Saved vs Raw-only", f"{100*saved:.1f}%")

# ===================== STRATEGY COMPARISON =====================
with tab_compare:
    st.subheader("Raw-only vs Hybrid (Adaptive) vs Semantic-only — whole-run summary")

    # --- NEW: live/whole-run switch ---
    live_mode = st.toggle("Live (use current time / slider)", value=True, help="When ON, all stats are computed up to the current time (t). When OFF, they are computed for the whole simulation.")

    # Helper: how many Lane B events have occurred up to t (inclusive)
    def laneB_attempted_upto(t):
        if t is None:   # whole run
            return len(events)
        return sum(1 for e in events if e.t <= t)

    # Helper: build KPIs for a result up to t (inclusive) or for whole run
    def kpis_upto(res, t):
        if t is None:
            laneA_bits = res["laneA_bits"]
            laneB_bits = res["laneB_bits"]
            raw_bits   = res["raw_bits"]
            laneA_attempt = SECS
            laneB_attempt = len(events)
        else:
            idx = max(0, min(int(t), SECS-1))
            slc = slice(0, idx+1)
            laneA_bits = res["laneA_bits"][slc]
            laneB_bits = res["laneB_bits"][slc]
            raw_bits   = res["raw_bits"][slc]
            laneA_attempt = idx + 1
            laneB_attempt = laneB_attempted_upto(idx)

        laneA_deliv = int((laneA_bits > 0).sum())
        laneB_deliv = int((laneB_bits > 0).sum())
        total_bits  = int(laneA_bits.sum() + laneB_bits.sum() + raw_bits.sum())

        return {
            "laneA_deliv": laneA_deliv,
            "laneA_attempt": laneA_attempt,
            "laneB_deliv": laneB_deliv,
            "laneB_attempt": laneB_attempt,
            "total_bits": total_bits,
        }

    # choose the cutoff time
    cutoff_t = st.session_state.t_idx if live_mode else None

    # compute KPIs for each strategy in the same window
    kpi_raw    = kpis_upto(res_raw, cutoff_t)
    kpi_hybrid = kpis_upto(res_hybrid, cutoff_t)
    kpi_sem    = kpis_upto(res_sem, cutoff_t)

    # Baseline for "Saved vs Raw-only": use Raw-only in the SAME window
    baseline_bits = max(1, kpi_raw["total_bits"])

    # formatting helper
    def as_row(name, k):
        la_succ = 100.0 * (k["laneA_deliv"] / max(1, k["laneA_attempt"]))
        lb_succ = 100.0 * (k["laneB_deliv"] / max(1, k["laneB_attempt"]))
        sent_MB = k["total_bits"] / (8*1024*1024)
        saved   = 100.0 * (1.0 - (k["total_bits"] / baseline_bits))
        return {
            "Strategy": name,
            "Lane A success %": la_succ,
            "Lane B success %": lb_succ,
            "Total sent (MB)": sent_MB,
            "Saved vs Raw-only (%)": saved,
        }

    table = pd.DataFrame([
        as_row("Raw only", kpi_raw),
        as_row("Hybrid (Adaptive)", kpi_hybrid),
        as_row("Semantic only", kpi_sem),
    ])

    st.dataframe(
        table.style.format({
            "Lane A success %": "{:.1f}",
            "Lane B success %": "{:.1f}",
            "Total sent (MB)":  "{:.2f}",
            "Saved vs Raw-only (%)": "{:.2f}",
        }),
        use_container_width=True,
    )

    # small hint so partners know what they're seeing
    if live_mode:
        st.caption(f"Live view at t = {st.session_state.t_idx}s (values are cumulative up to the current time).")
    else:
        st.caption("Whole-run summary (entire simulation).")


# ===================== WHEN TO USE RAW VS SEMANTIC =====================
with tab_why:
    st.subheader("When do we use RAW vs SEMANTIC?")
    st.markdown("""
- **Lane A (Safety)** — always **structured & tiny** (telegram each second) regardless of strategy.
- **Lane B (Operational)**:
  - **Raw only**: stream raw sensors continuously; best for lab/offline analytics; highest bandwidth.
  - **Semantic only**: send compact events/tokens; best under poor coverage; lowest bandwidth.
  - **Hybrid (Adaptive)** *(recommended)*: allow some raw when **GOOD**, throttle raw when **PATCHY/POOR** to protect event delivery and save bandwidth.
""")

st.markdown("---")
st.caption("Tiles © OpenStreetMap contributors. Railway path is an approximation for demo purposes.")
