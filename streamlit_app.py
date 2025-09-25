# ENSURE-6G • TMS Rail Demo — Integrated Telemetry + Sensor Inspector + Optimized UI
# (No CSVs; per-sensor time series generated live)

import math, numpy as np, pandas as pd, streamlit as st, pydeck as pdk
from shapely.geometry import LineString, Point
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ENSURE-6G • TMS Rail (Live Telemetry)", layout="wide")

# -------------------- Color-blind palette & CSS --------------------
CBL = {
    "green":[0,153,136],"orange":[230,159,0],"red":[213,94,0],
    "blue":[86,180,233],"cyan":[0,158,115],"purple":[204,121,167],
    "gold":[240,228,66],"gray":[120,120,120],
}
st.markdown("""
<style>
.kpi-chip{display:inline-block;padding:6px 10px;margin:2px 6px;border-radius:12px;background:#F6F6F9;font-weight:600;}
.chip-good{color:rgb(0,153,136)} .chip-mid{color:rgb(230,159,0)} .chip-bad{color:rgb(213,94,0)}
.chip-raw{color:rgb(86,180,233)} .chip-sem{color:rgb(204,121,167)} .chip-hyb{color:rgb(0,158,115)}
.ribbon{background:#0F172A;color:#fff;padding:6px 10px;border-radius:8px;margin:6px 0;display:inline-block}
.legend-dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px}
</style>
""", unsafe_allow_html=True)

# -------------------- Geo helpers --------------------
R_EARTH = 6371000.0
def haversine_m(lat1, lon1, lat2, lon2):
    p=np.pi/180.0; dlat=(lat2-lat1)*p; dlon=(lon2-lon1)*p
    a=np.sin(dlat/2)**2+np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 2*R_EARTH*np.arcsin(min(1.0, np.sqrt(a)))

def haversine_vec(lat1, lon1, lat2, lon2):
    lat1=np.asarray(lat1,float); lon1=np.asarray(lon1,float)
    p=np.pi/180.0; dlat=(lat2-lat1)*p; dlon=(lon2-lon1)*p
    a=np.sin(dlat/2)**2+np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 2*R_EARTH*np.arcsin(np.minimum(1.0, np.sqrt(a)))

RAIL_WAYPOINTS=[
    (62.3930,17.3070),(62.1200,17.1500),(61.8600,17.1400),(61.7300,17.1100),
    (61.5600,17.0800),(61.3900,17.0700),(61.3000,17.0600),(61.0700,17.1000),
    (60.8500,17.1600),(60.6749,17.1413),(60.3800,17.3300),(60.2000,17.4500),
    (60.0500,17.5200),(59.9300,17.6100),(59.8586,17.6389),(59.7500,17.8200),
    (59.6600,17.9400),(59.6100,17.9900),(59.5500,18.0300),(59.4800,18.0400),
    (59.4200,18.0600),(59.3700,18.0700),(59.3293,18.0686),
]
ROUTE_LS=LineString([(lon,lat) for lat,lon in RAIL_WAYPOINTS])

BASE_STATIONS=[
    ("BS-Sundsvall",62.386,17.325,16000),("BS-Njurunda",62.275,17.354,14000),
    ("BS-Harmånger",61.897,17.170,14000),("BS-Hudiksvall",61.728,17.103,15000),
    ("BS-Söderhamn",61.303,17.058,15000),("BS-Axmar",61.004,17.190,14000),
    ("BS-Gävle",60.675,17.141,16000),("BS-Tierp",60.345,17.513,14000),
    ("BS-Skyttorp",60.030,17.580,14000),("BS-Uppsala",59.858,17.639,16000),
    ("BS-Märsta",59.620,17.860,15000),("BS-Stockholm",59.330,18.070,18000),
]
HOTSPOTS=[
    dict(name="Hudiksvall cut", lat=61.728,  lon=17.103,  radius_m=12000),
    dict(name="Gävle marsh",    lat=60.675,  lon=17.141,  radius_m=15000),
    dict(name="Uppsala bend",   lat=59.8586, lon=17.6389, radius_m=12000),
]

def interpolate_polyline(points, n_pts):
    n_pts=max(2,int(n_pts)); lat=np.array([p[0] for p in points],float); lon=np.array([p[1] for p in points],float)
    cum=np.zeros(len(points))
    for i in range(1,len(points)): cum[i]=cum[i-1]+haversine_m(lat[i-1],lon[i-1],lat[i],lon[i])
    tgt=np.linspace(0.0,float(cum[-1]),n_pts); idx=np.searchsorted(cum,tgt,side="right"); idx=np.clip(idx,1,len(cum)-1)
    i0, i1=idx-1, idx; d0, d1=cum[i0], cum[i1]; w=(tgt-d0)/np.maximum(d1-d0,1e-9)
    return pd.DataFrame({"lat": lat[i0]+(lat[i1]-lat[i0])*w, "lon": lon[i0]+(lon[i1]-lon[i0])*w, "s_m": tgt})

def label_segments(n):
    names=["Sundsvall→Hudiksvall","Hudiksvall→Söderhamn","Söderhamn→Gävle","Gävle→Uppsala","Uppsala→Stockholm"]
    b=np.linspace(0,n,len(names)+1).astype(int); lab=np.empty(n,dtype=object)
    for i in range(len(names)): lab[b[i]:b[i+1]]=names[i]
    return lab

def point_in_poly(lat,lon,poly):
    xs=[pt[0] for pt in poly]; ys=[pt[1] for pt in poly]
    return (min(ys)<=lat<=max(ys)) and (min(xs)<=lon<=max(xs))

def segment_of_point(lat,lon,route_df,seg_labels):
    d=((route_df.lat-lat)**2+(route_df.lon-lon)**2)**0.5; idx=int(np.argmin(d.values))
    return seg_labels[idx], idx

def nearest_bs_quality(lat,lon):
    best=None
    for name,blat,blon,R in BASE_STATIONS:
        d=haversine_m(lat,lon,blat,blon)
        q="GOOD" if d<=R else ("PATCHY" if d<=2.2*R else "POOR")
        rank={"GOOD":0,"PATCHY":1,"POOR":2}[q]
        if best is None or rank<best[3]: best=(name,d,q,rank)
    return best[0],best[1],best[2]

def cap_loss(qual,t_sec,base_capacity_kbps=800,burst_factor=1.4,good_loss_pct=0.5,bad_loss_pct=10.0):
    cap=int(base_capacity_kbps*1000)
    if qual=="GOOD": return int(cap*burst_factor), good_loss_pct/100.0
    if qual=="PATCHY":
        wobble=0.6+0.2*math.sin(2*math.pi*(t_sec%30)/30.0)
        return max(int(cap*wobble*0.9),1), min(0.4,(bad_loss_pct*0.5)/100.0)
    return int(cap*0.25), bad_loss_pct/100.0

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Scenario Controls")
    preset = st.segmented_control("Presets", ["Good","Mixed","Adverse"], key="preset_sel")
    if preset=="Good": default_minutes, default_TTT, default_HO, default_dc = 20, 1000, 200, True
    elif preset=="Mixed": default_minutes, default_TTT, default_HO, default_dc = 20, 1200, 350, True
    else: default_minutes, default_TTT, default_HO, default_dc = 20, 1600, 600, False

    sim_minutes_total = st.number_input("Sim Length (minutes)",5,120,default_minutes,5)
    use_tiles = st.toggle("Use OSM tiles", False)
    mode = st.radio("Comm Mode (uplink)", ["RAW","SEMANTIC","HYBRID"], index=2)
    st.markdown("---")
    st.subheader("Lane A (Safety)")
    laneA_reps = st.slider("Repetitions",1,3,2,1)
    enable_dc = st.checkbox("Dual Connectivity", default_dc)
    dc_min_snr_delta = st.slider("DC min SNR delta (dB)",0.0,10.0,2.0,0.5)
    st.markdown("---")
    st.subheader("Handover")
    TTT_MS = st.slider("Time-To-Trigger (ms)",200,3000,default_TTT,100)
    HO_GAP_MS = st.slider("Handover outage (ms)",0,1500,default_HO,50)
    st.markdown("---")
    st.subheader("TSR / STOP")
    tsr_conf_critical = st.slider("Critical buckling threshold",0.60,0.95,0.85,0.01)
    tsr_speed_kmh = st.slider("TSR speed (km/h)",30,120,60,5)
    stop_on_critical = st.checkbox("Issue STOP when confidence ≥0.92", True)
    st.markdown("---")
    st.subheader("Maintenance")
    repair_time_s = st.slider("On-site repair time (s)",30,900,180,10)
    crew_capacity = st.slider("Max concurrent crews",1,4,2,1)
    st.markdown("---")
    st.subheader("Demo Issues")
    demo_force_issues = st.checkbox("Inject summer hotspots", True)
    summer_severity = st.slider("Summer severity (°C boost)",0.0,20.0,12.0,1.0)
    always_show_tsr = st.checkbox("Always show TSR polygons for injected issues", True)
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("⏮ -10s", use_container_width=True):
            st.session_state.t_idx = max(0, st.session_state.get("t_idx",0)-10)
    with c2:
        rate = st.segmented_control("Rate",["0.5×","1×","2×","4×"], key="play_rate")
    with c3:
        if st.button("⏭ +10s", use_container_width=True):
            st.session_state.t_idx = min(int(sim_minutes_total*60)-1, st.session_state.get("t_idx",0)+10)
    st.markdown("---")
    if st.button("🔄 Reset simulation state", use_container_width=True):
        for k in ["_frame","sensor_hist","arr","_times","qual","seg","tsr_polys_real","tsr_polys_tms","work_orders",
                  "env_ts","sensor_params"]:
            st.session_state.pop(k, None)
        for k,v in [("t_idx",0),("train_s_m",0.0),("train_v_ms",0.0),("bearer","5G"),
                    ("bearer_prev","5G"),("bearer_ttt",0),("handover_gap_until",-1)]:
            st.session_state[k]=v
        st.success("State cleared. Press ▶ Play.")

# -------------------- Session init --------------------
SECS = max(2, int(sim_minutes_total*60))
if "route_secs" not in st.session_state or st.session_state.route_secs != SECS:
    st.session_state.route_df = interpolate_polyline(RAIL_WAYPOINTS, SECS)
    st.session_state.seg_labels = label_segments(SECS)
    st.session_state.route_secs = SECS
    st.session_state.pop("_frame", None)
    st.session_state.pop("sensor_hist", None)

if "t_idx" not in st.session_state: st.session_state.t_idx = 0
if "playing" not in st.session_state: st.session_state.playing = False
if "alerts_feed" not in st.session_state: st.session_state.alerts_feed = []

# Train kinematics
if "train_s_m" not in st.session_state: st.session_state.train_s_m = 0.0
if "train_v_ms" not in st.session_state: st.session_state.train_v_ms = 0.0
DT_S=1.0; V_MAX_MS=200/3.6; A_MAX=0.6; B_MAX=0.9

# Ops state
for k,v in [("bearer","5G"),("bearer_prev","5G"),("bearer_ttt",0),("handover_gap_until",-1),
            ("tsr_polys_real",[]),("tsr_polys_tms",[]),("work_orders",[])]:
    if k not in st.session_state: st.session_state[k]=v

route_df=st.session_state.route_df; seg_labels=st.session_state.seg_labels

# -------------------- PHY / channel --------------------
def env_class(lat,lon):
    for c in [(62.3913,17.3063),(60.6749,17.1413),(59.8586,17.6389),(59.3293,18.0686)]:
        if haversine_m(lat,lon,c[0],c[1])<15000: return "UMa"
    return "RMa"

def pathloss_db(freq_GHz,d_m,env):
    if d_m<1: d_m=1
    fspl=32.4+20*np.log10(freq_GHz*1000)+20*np.log10(d_m/1000)
    return fspl+(7 if env=="UMa" else 3)

class ShadowingTrack:
    def __init__(self,sigma=7,decor=100,seed=7):
        rng=np.random.default_rng(seed); self.sigma=sigma; self.decor=decor; self.rng=rng; self.last_s=0.0; self.curr=0.0
    def sample(self,s):
        rho=np.exp(-abs(s-self.last_s)/self.decor)
        self.curr=rho*self.curr+math.sqrt(max(1e-9,1-rho**2))*self.rng.normal(0,self.sigma)
        self.last_s=s; return self.curr
if "shadow" not in st.session_state: st.session_state.shadow=ShadowingTrack()

def rician_db(K_dB=8):
    K=10**(K_dB/10); h=math.sqrt(K/(K+1))+(np.random.normal(0,1/np.sqrt(2))+1j*np.random.normal(0,1/np.sqrt(2)))
    p=(abs(h)**2)/(K+1); return 10*np.log10(max(p,1e-6))
def rayleigh_db():
    h=np.random.normal(0,1/np.sqrt(2))+1j*np.random.normal(0,1/np.sqrt(2)); p=abs(h)**2; return 10*np.log10(max(p,1e-6))
def noise_dbm(bw_hz): return -174+10*np.log10(bw_hz)+5

TECH={"5G":dict(freq=3.5,bw=5e6,base_lat=20,snr_ok=3,snr_hold=1),
      "LTE":dict(freq=1.8,bw=3e6,base_lat=35,snr_ok=0,snr_hold=-2),
      "3G":dict(freq=2.1,bw=1.5e6,base_lat=60,snr_ok=-2,snr_hold=-4),
      "GSMR":dict(freq=0.9,bw=200e3,base_lat=120,snr_ok=-4,snr_hold=-6)}
P_TX=43

def serving_bs(lat,lon):
    d=[haversine_m(lat,lon,b[1],b[2]) for b in BASE_STATIONS]; i=int(np.argmin(d))
    return dict(name=BASE_STATIONS[i][0],lat=BASE_STATIONS[i][1],lon=BASE_STATIONS[i][2],
                tech={"5G","LTE","3G","GSMR"}), d[i]

def per_from_snr(snr_db):
    x0,k=2.0,-1.1; per=1/(1+math.exp(k*(snr_db-x0)))
    return max(1e-5,min(0.99,per))

def pick_bearer(snr_table,caps,curr):
    order=["5G","LTE","3G","GSMR"]; avail=[b for b in order if b in caps]
    for b in avail:
        if snr_table.get(b,-99)>=TECH[b]["snr_ok"]: return b, True
    if avail: return max(avail,key=lambda x: snr_table.get(x,-99)), True
    return curr, False

def pick_secondary(primary,snr_table,min_delta_db=2.0):
    alts=[(b,s) for b,s in snr_table.items() if b!=primary]
    if not alts: return None
    b2,s2=max(alts,key=lambda x:x[1])
    return b2 if s2+1e-9>=snr_table[primary]-min_delta_db else None

def index_from_s(route_df,s_m):
    s=float(np.clip(s_m,0.0,float(route_df.s_m.iloc[-1])))
    idx=int(np.searchsorted(route_df.s_m.values,s,side="left"))
    return min(max(idx,0),len(route_df)-1)

# -------------------- LIVE TELEMETRY (no CSV) --------------------
def _init_env_timeseries(SECS,freq_s=1,seed=4242):
    rng=np.random.default_rng(seed); t=np.arange(SECS)*freq_s
    day_frac=(t%86400)/86400.0
    sun=np.clip(np.sin(np.pi*(day_frac-0.25)),0,1)
    clouds=np.cumsum(rng.normal(0,0.02,SECS)); clouds=np.clip(clouds,-1.5,1.5)
    airC=15+7*np.sin(2*np.pi*day_frac)+2*clouds
    wind_sp=np.clip(3+1.5*np.sin(2*np.pi*(day_frac-0.1))+np.cumsum(rng.normal(0,0.02,SECS)),0,None)
    gust=(rng.random(SECS)<0.01); wind_sp=wind_sp+gust*rng.uniform(3,7,size=SECS)
    wind_dir=(160+rng.normal(0,12,SECS))%360; swing=(rng.random(SECS)<0.03); wind_dir[swing]=rng.uniform(0,360,size=swing.sum())
    return dict(t=t,sun=sun,clouds=clouds,airC=airC,wind_sp=wind_sp,wind_dir=wind_dir)

def _init_sensor_params(n_sensors,hotspot_fraction=0.25,seed=777):
    rng=np.random.default_rng(seed); params={}; hot=set(rng.choice(n_sensors,size=max(1,int(n_sensors*hotspot_fraction)),replace=False))
    for i in range(n_sensors):
        sid=f"S{i:02d}"
        params[sid]=dict(bias=rng.normal(0,0.5), pv_gain=float(np.clip(0.9+rng.normal(0,0.05),0.75,1.05)),
                         hotspot=(i in hot), batt_V=float(rng.uniform(11.9,12.8)))
    return params

def sensor_model_step(sid, idx_t, env, p, summer_boost_C):
    sun=float(env["sun"][idx_t]); clouds=float(env["clouds"][idx_t]); airC=float(env["airC"][idx_t])
    light=(sun**1.7)*(1.0-0.25*max(0,clouds))*120_000 + np.random.normal(0,2500)
    irr=(sun**1.3)*p["pv_gain"]*950*(1.0-0.25*max(0,clouds)) + np.random.normal(0,20)
    light=max(0.0,light); irr=max(0.0,irr)

    solar_load=0.22*irr/100.0
    rail_no=airC+solar_load+p["bias"]+np.random.normal(0,0.25)
    last=p.get("_rail_prev",rail_no); rail=0.92*last+0.08*rail_no; p["_rail_prev"]=rail
    if p["hotspot"]: rail += (sun**2.2)*summer_boost_C

    strain=max(0.0, 0.5*(rail-32)+np.random.normal(0,0.35))
    showers=(np.random.random()<0.004); ballast=float(np.clip(0.25+0.25*max(0,-clouds)/1.5+0.15*(1.0 if showers else 0.0)+np.random.normal(0,0.04),0,1))

    V=p["batt_V"]; charge=0.00002*irr; drain=0.0015*(1.0-sun)+0.0008
    V=float(np.clip(V+charge-drain+np.random.normal(0,0.002),10.8,13.0)); p["batt_V"]=V

    risk=1/(1+np.exp(-(0.20*(rail-35)+0.35*(strain-8)+0.9*(ballast-0.6)))); risk=float(np.clip(risk,0,1))
    flags=[]; 
    if rail>=38.0: flags.append("temp")
    if strain>=10.0: flags.append("strain")
    if risk>=0.85: flags.append("risk")

    return dict(light_lux=int(round(light)), irradiance_Wm2=round(irr,1),
                wind_speed_ms=round(float(env["wind_sp"][idx_t])+np.random.normal(0,0.05),2),
                wind_dir_deg=round(float(env["wind_dir"][idx_t]),1),
                rail_temp_C=round(rail,2), strain_kN=round(strain,2),
                ballast_idx=round(ballast,3), battery_V=round(V,3),
                risk_score=round(risk,3), exceeded=",".join(flags), hotspot=bool(p["hotspot"]))

# Initialize env & sensor params once per sim length
if "env_ts" not in st.session_state or len(st.session_state.env_ts.get("t",[]))!=SECS:
    st.session_state.env_ts=_init_env_timeseries(SECS,1,4242)
N_SENS=22
if "sensor_params" not in st.session_state or len(st.session_state.sensor_params)!=N_SENS:
    st.session_state.sensor_params=_init_sensor_params(N_SENS,0.25,777)
if "sensor_hist" not in st.session_state:
    st.session_state.sensor_hist={f"S{i:02d}":[] for i in range(N_SENS)}

# -------------------- Cache static map layers --------------------
@st.cache_data(show_spinner=False)
def static_layers(SECS,route_df,use_tiles):
    step=max(1,SECS//300)
    path_coords=[[route_df.lon.iloc[i],route_df.lat.iloc[i]] for i in range(0,SECS,step)]
    path_df=pd.DataFrame([{"path":path_coords,"name":"Sundsvall→Stockholm"}])
    path_layer=pdk.Layer("PathLayer",data=path_df,get_path="path",get_color=[60,60,160,180],width_scale=4,width_min_pixels=3)
    bs_df=pd.DataFrame(BASE_STATIONS,columns=["name","lat","lon","r_m"])
    rings=[]
    for r in bs_df.itertuples():
        rings += [
            {"lon":r.lon,"lat":r.lat,"radius":r.r_m,          "color":CBL["green"]+[45]},
            {"lon":r.lon,"lat":r.lat,"radius":int(r.r_m*2.2), "color":CBL["orange"]+[35]},
            {"lon":r.lon,"lat":r.lat,"radius":int(r.r_m*3.0), "color":CBL["red"]+[25]},
        ]
    rings_df=pd.DataFrame(rings)
    bs_rings=pdk.Layer("ScatterplotLayer",data=rings_df,get_position="[lon,lat]",get_radius="radius",
                       get_fill_color="color",stroked=True,get_line_color=[0,0,0,60],line_width_min_pixels=1)
    tile_layer=None
    if use_tiles:
        tile_layer=pdk.Layer("TileLayer",data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",min_zoom=0,max_zoom=19,tile_size=256)
    return tile_layer, path_layer, bs_rings, path_coords

# -------------------- Tabs --------------------
tab_map, tab_flow, tab_ops = st.tabs(["Map & KPIs","Comm Flow","Ops"])

with tab_map:
    # Playback & legend row
    c_topL, c_topM, c_topR = st.columns([1.4,1.2,1.4])
    with c_topL:
        st.subheader("Playback")
        c1,c2,c3=st.columns(3)
        if c1.button("▶ Play", use_container_width=True): st.session_state.playing=True
        if c2.button("⏸ Pause", use_container_width=True): st.session_state.playing=False
        if c3.button("⏹ Stop", use_container_width=True):
            st.session_state.playing=False; st.session_state.t_idx=0; st.session_state.train_s_m=0.0; st.session_state.train_v_ms=0.0
        if st.session_state.playing:
            rate_map={"0.5×":1400,"1×":700,"2×":350,"4×":175}
            st_autorefresh(interval=rate_map.get(st.session_state.get("play_rate","1×"),700), key=f"tick_{SECS}")
            st.session_state.t_idx=min(st.session_state.t_idx+1, SECS-1)
            if st.session_state.t_idx>=SECS-1: st.session_state.playing=False
            st.slider("Time (s)",0,SECS-1,value=st.session_state.t_idx,disabled=True)
        else:
            st.session_state.t_idx = st.slider("Time (s)",0,SECS-1,value=st.session_state.t_idx)
    with c_topM:
        st.markdown('<div id="kpi-holder"></div>', unsafe_allow_html=True)
    with c_topR:
        st.subheader("Legend")
        st.markdown(
            f"""
            <div><span class="legend-dot" style="background:rgb{tuple(CBL['green'])};"></span>Coverage GOOD</div>
            <div><span class="legend-dot" style="background:rgb{tuple(CBL['orange'])};"></span>Coverage PATCHY</div>
            <div><span class="legend-dot" style="background:rgb{tuple(CBL['red'])};"></span>Coverage POOR</div>
            <div><span class="legend-dot" style="background:rgb{tuple(CBL['blue'])};"></span>Sensors RAW</div>
            <div><span class="legend-dot" style="background:rgb{tuple(CBL['cyan'])};"></span>Sensors HYBRID</div>
            <div><span class="legend-dot" style="background:rgb{tuple(CBL['purple'])};"></span>Sensors SEMANTIC</div>
            <div><span class="legend-dot" style="background:rgb{tuple(CBL['gold'])};"></span>TSR polygon</div>
            """, unsafe_allow_html=True
        )

    # Frame builder
    def build_frame(t_idx):
        idx_s=index_from_s(route_df, st.session_state.get("train_s_m",0.0))
        trainA=(float(route_df.lat.iloc[idx_s]), float(route_df.lon.iloc[idx_s]))
        t_idx_nb=(len(route_df)-1-idx_s)%len(route_df)
        trainB=(float(route_df.lat.iloc[t_idx_nb]), float(route_df.lon.iloc[t_idx_nb]))
        seg=seg_labels[idx_s]
        sidx=np.linspace(0,len(route_df)-1,N_SENS).astype(int)
        sensors=pd.DataFrame([{"sid":f"S{i:02d}","lat":float(route_df.lat.iloc[j]),"lon":float(route_df.lon.iloc[j])}
                              for i,j in enumerate(sidx)])
        _,_,quality=nearest_bs_quality(*trainA)
        cap_bps, rand_loss=cap_loss(quality,t_idx)
        return dict(t=t_idx,trainA=trainA,trainB=trainB,segment=seg,quality=quality,cap_bps=cap_bps,rand_loss=rand_loss,
                    enforce_stop=False,crash=False,sensors=sensors)

    if "_frame" not in st.session_state or not isinstance(st.session_state._frame.get("sensors",None),pd.DataFrame):
        st.session_state._frame=build_frame(0)
    if "sensor_hist" not in st.session_state: st.session_state.sensor_hist={f"S{i:02d}":[] for i in range(N_SENS)}

    colR, colL = st.columns([1.0,2.6])

    with colR:
        kpi_holder = st.container()
        st.markdown("#### Alerts (Lane-A)")
        alerts_box = st.container()
        st.markdown("#### Affected sensors")
        affected_box = st.container()
        st.markdown("#### Telemetry")
        t1,t2,t3 = st.tabs(["Throughput","Latency/Success","Radio"])

    with colL:
        st.subheader("Live Maps — Real World vs TMS View")
        tile_layer, static_path_layer, static_bs_rings, static_path_coords = static_layers(SECS,route_df,use_tiles)

        # ------ Compute frame physics/comm & sensors ------
        t=st.session_state.t_idx
        idx_s=index_from_s(route_df, st.session_state.train_s_m)
        trainA=(float(route_df.lat.iloc[idx_s]), float(route_df.lon.iloc[idx_s]))
        seg=seg_labels[idx_s]; s_along=float(route_df.s_m.iloc[idx_s])

        frame=st.session_state._frame
        sensors_base = frame["sensors"][["sid","lat","lon"]].copy()

        # PHY at train
        bsA,dA=serving_bs(*trainA); envA=env_class(*trainA); shadow=st.session_state.shadow
        snr_table={}
        for b in ["5G","LTE","3G","GSMR"]:
            if b in bsA["tech"]:
                pl=pathloss_db(TECH[b]["freq"], dA, envA); sh=shadow.sample(s_along); fad=rician_db(8) if envA=="RMa" else rayleigh_db()
                rx=43 - pl + sh + fad; snr_table[b]=rx - noise_dbm(TECH[b]["bw"])
        cand, valid = pick_bearer(snr_table, bsA["tech"], st.session_state.bearer)
        if valid and cand!=st.session_state.bearer:
            st.session_state.bearer_ttt += 700
            if st.session_state.bearer_ttt>=TTT_MS:
                st.session_state.bearer_prev=st.session_state.bearer; st.session_state.bearer=cand
                st.session_state.bearer_ttt=0; st.session_state.handover_gap_until = t + math.ceil(HO_GAP_MS/700.0)
        else: st.session_state.bearer_ttt=0

        bearer=st.session_state.bearer; snr_use=snr_table.get(bearer,-20.0)
        per_single=per_from_snr(snr_use)
        secondary=pick_secondary(bearer,snr_table,dc_min_snr_delta) if enable_dc else None
        per_secondary=per_from_snr(snr_table.get(secondary,-20.0)) if secondary else None
        laneA_success_phy=((1-per_single)**laneA_reps) if not secondary else 1-((1-(1-per_single)**laneA_reps)*(1-(1-per_secondary)**laneA_reps))

        # Macro capacity & gap
        _,_,quality=nearest_bs_quality(*trainA)
        cap_bps_train, rand_loss=cap_loss(quality,t)
        in_gap=(t < st.session_state.handover_gap_until)

        # ---- Per-sensor telemetry (integrated) ----
        def sensor_row(r):
            sid=str(r["sid"]); idx_t=int(st.session_state.t_idx)
            env=st.session_state.env_ts; p=st.session_state.sensor_params[sid]
            telem=sensor_model_step(sid,idx_t,env,p,summer_boost_C=summer_severity)
            score=float(telem["risk_score"])
            label="high" if score>0.75 else ("medium" if score>0.4 else "low")
            _,_,qualS=nearest_bs_quality(r.lat,r.lon); capS,lossS=cap_loss(qualS,idx_t)
            hist=st.session_state.sensor_hist[sid]; hist.append(dict(t=idx_t, **telem))
            if len(hist)>21600: del hist[:len(hist)-21600]
            return dict(score=score,label=label,exceeded=telem["exceeded"],
                        temp=telem["rail_temp_C"],strain=telem["strain_kN"],ballast=telem["ballast_idx"],
                        qualS=qualS,capS=capS,lossS=lossS, light_lux=telem["light_lux"],irradiance_Wm2=telem["irradiance_Wm2"],
                        wind_speed_ms=telem["wind_speed_ms"],wind_dir_deg=telem["wind_dir_deg"],battery_V=telem["battery_V"],
                        hotspot=telem["hotspot"])
        S=sensors_base.apply(sensor_row,axis=1,result_type="expand")
        sensors=pd.concat([sensors_base,S],axis=1)

        # Segment tags
        seg_list=[]; seg_idx=[]
        for r in sensors.itertuples():
            sname, idxp=segment_of_point(r.lat,r.lon,route_df,seg_labels); seg_list.append(sname); seg_idx.append(idxp)
        sensors["segment"]=seg_list; sensors["_seg_idx"]=seg_idx

        # Modality decision
        def choose_modality(r):
            qual=str(r["qualS"]); cap=float(r["capS"]); score=float(r["score"])
            if (qual=="POOR") or (cap<100_000): return "SEMANTIC"
            if (qual=="GOOD") and (score<0.4) and (cap>400_000): return "RAW"
            return "HYBRID"
        sensors["modality"]=sensors.apply(choose_modality,axis=1)

        # Uplink loads
        RAW_HZ={"RAW":2.0,"HYBRID":0.2,"SEMANTIC":0.0}; BYTES_RAW=24; BYTES_ALERT=280; BYTES_SUMMARY=180
        sensors["raw_hz"]=sensors["modality"].map(RAW_HZ)
        sensors["raw_bps"]=(sensors["raw_hz"]*BYTES_RAW)*(1.0-sensors["lossS"])
        raw_bps_delivered=int(sensors["raw_bps"].sum())

        # Lane-A alerts (subject to uplink loss)
        rng=np.random.default_rng(42+t); laneA_alerts=[]
        for r in sensors.itertuples():
            if r.label in ("medium","high") and (("temp" in r.exceeded) or ("strain" in r.exceeded)):
                conf=round(0.6+0.4*r.score,2)
                if rng.uniform() < (1.0-r.lossS):
                    laneA_alerts.append(dict(sid=r.sid,location=dict(lat=r.lat,lon=r.lon),
                                             severity=r.label,confidence=conf,
                                             evidence=dict(rail_temp_C=r.temp,strain_kN=r.strain,ballast_idx=r.ballast)))
        laneB_msgs=[]
        if mode in ("SEMANTIC","HYBRID") and any(m in ("SEMANTIC","HYBRID") for m in sensors["modality"]):
            laneB_msgs.append(dict(type="maintenance_summary", ballast_hotspots=int((sensors.ballast>0.6).sum()),
                                   alerts=len(laneA_alerts), window=f"t={t}s"))
        laneA_bps=len(laneA_alerts)*BYTES_ALERT*(2 if (enable_dc and secondary) else 1)
        laneB_bps=len(laneB_msgs)*BYTES_SUMMARY; raw_bps=raw_bps_delivered; bps_total=laneA_bps+laneB_bps+raw_bps

        # TSR polygons (real)
        def tsr_poly(clat,clon,length_m=1500,half_w=18):
            lat0,lon0=clat,clon; m2deg_lat=1/111111.0; m2deg_lon=1/(111111.0*math.cos(math.radians(lat0)))
            nearest=ROUTE_LS.interpolate(ROUTE_LS.project(Point(lon0,lat0)))
            step_m=length_m/10; pts=[nearest]
            for sgn in (+1,-1):
                acc=0
                while acc<length_m/2:
                    acc+=step_m; s=ROUTE_LS.project(nearest)+sgn*acc; s=max(0,min(s,ROUTE_LS.length)); pts.append(ROUTE_LS.interpolate(s))
            pts=sorted(pts,key=lambda p: ROUTE_LS.project(p)); p0,p1=pts[0],pts[-1]
            dx,dy=p1.x-p0.x,p1.y-p0.y; L=math.hypot(dx,dy)+1e-12; nx,ny=-dy/L,dx/L
            off_lon=(half_w*m2deg_lon)*nx; off_lat=(half_w*m2deg_lat)*ny
            return [[p0.x-off_lon,p0.y-off_lat],[p0.x+off_lon,p0.y+off_lat],[p1.x+off_lon,p1.y+off_lat],[p1.x-off_lon,p1.y-off_lat]]

        new_real=[]
        for a in laneA_alerts:
            if a["confidence"]>=tsr_conf_critical:
                poly=tsr_poly(a["location"]["lat"],a["location"]["lon"])
                very=a["confidence"]>=0.92 and stop_on_critical
                new_real.append(dict(polygon=poly,speed=tsr_speed_kmh,created_idx=t,critical=True,ack_train=False,stop=very))
        for p in new_real: st.session_state.tsr_polys_real.append(p)

        # Forced demo TSRs
        if demo_force_issues and always_show_tsr and len(sensors)>0:
            latv=sensors["lat"].values; lonv=sensors["lon"].values
            for h in HOTSPOTS:
                dist=haversine_vec(latv,lonv,float(h["lat"]),float(h["lon"]))
                in_hot=dist<=float(h["radius_m"])
                if np.any(in_hot):
                    s_hot=sensors.loc[in_hot].sort_values("score",ascending=False).iloc[0]
                    poly=tsr_poly(float(s_hot["lat"]),float(s_hot["lon"]))
                    st.session_state.tsr_polys_real.append(dict(polygon=poly,speed=tsr_speed_kmh,created_idx=t,
                                                                critical=True,ack_train=True,stop=(float(s_hot["score"])>0.92)))

        # Downlink → TMS awareness
        _,_,qual_down=nearest_bs_quality(*trainA); _,rand_down=cap_loss(qual_down,t)
        loss_down=min(0.95, rand_loss + (0.30 if in_gap else 0.0))
        if np.random.random() < (1.0-loss_down):
            for p in st.session_state.tsr_polys_real:
                if p not in st.session_state.tsr_polys_tms: st.session_state.tsr_polys_tms.append(p)

        # STOP & CRASH
        enforce_stop=any(p.get("stop",False) for p in st.session_state.tsr_polys_tms)
        crash=False
        for p in st.session_state.tsr_polys_real:
            if p["critical"] and (p not in st.session_state.tsr_polys_tms) and point_in_poly(trainA[0],trainA[1],p["polygon"]):
                crash=True; break

        # Speed target by TSR
        tsr_kmh_here=None
        for p in st.session_state.tsr_polys_tms:
            if point_in_poly(trainA[0],trainA[1],p["polygon"]):
                tsr_kmh_here=p["speed"] if tsr_kmh_here is None else min(tsr_kmh_here,p["speed"])
        v_target=0.0 if enforce_stop else (tsr_kmh_here/3.6 if tsr_kmh_here is not None else V_MAX_MS)

        # Kinematics
        v_cur=st.session_state.train_v_ms
        v_new=min(v_cur+A_MAX*DT_S, v_target) if v_target>=v_cur else max(v_cur-B_MAX*DT_S, v_target)
        s_new=float(np.clip(st.session_state.train_s_m+v_new*DT_S,0.0,float(route_df.s_m.iloc[-1])))
        if s_new>=float(route_df.s_m.iloc[-1])-1e-6: v_new=0.0; st.session_state.playing=False
        st.session_state.train_v_ms=v_new; st.session_state.train_s_m=s_new

        # Latency & lane-A success
        lat_ms=TECH[bearer]["base_lat"] + (bps_total/1000.0)
        if bps_total>cap_bps_train: lat_ms *= min(4.0, 1.0 + 0.35*(bps_total/cap_bps_train - 1))
        if in_gap: lat_ms += 80
        laneA_success = ((1-per_single)**laneA_reps if not secondary else laneA_success_phy)
        if in_gap and not secondary: laneA_success=max(0.0,laneA_success*0.85)

        # Alerts feed
        for a in laneA_alerts[:6]:
            st.session_state.alerts_feed.append(
                f"t={t}s • {a['sid']} • {a['severity']} • conf={int(a['confidence']*100)}% "
                f"(T:{a['evidence']['rail_temp_C']}°C, S:{a['evidence']['strain_kN']})"
            )
        st.session_state.alerts_feed=st.session_state.alerts_feed[-6:]

        # KPI chips
        badge={"GOOD":("chip-good","🟢"),"PATCHY":("chip-mid","🟠"),"POOR":("chip-bad","🔴")}[quality]
        chip_html=f"""
          <span class="kpi-chip {badge[0]}">{badge[1]} {quality}</span>
          <span class="kpi-chip">Bearer: {bearer}</span>
          <span class="kpi-chip">Cap: {int(cap_bps_train/1000)} kbps</span>
          <span class="kpi-chip">Lane-A succ: {laneA_success*100:.0f}%</span>
          <span class="kpi-chip">Latency: {int(lat_ms)} ms</span>
          <span class="kpi-chip">Speed: {st.session_state.train_v_ms*3.6:,.0f} km/h</span>
          <span class="kpi-chip">Seg: {seg}</span>
        """
        with kpi_holder: st.markdown(chip_html, unsafe_allow_html=True)

        # Build dynamic map layers
        def dynamic_layers(tsr_list,train_pos,sensors_df,quality_macro):
            # heat: nearest sensor risk label → color
            if isinstance(sensors_df,pd.DataFrame) and not sensors_df.empty and "label" in sensors_df.columns:
                latv=sensors_df["lat"].values; lonv=sensors_df["lon"].values
                path_np=np.array(static_path_coords)
                d2=((path_np[:,1][:,None]-latv[None,:])**2 + (path_np[:,0][:,None]-lonv[None,:])**2)
                j=np.argmin(d2,axis=1); labels=sensors_df["label"].values[j]
                col_map={"low":CBL["green"]+[180],"medium":CBL["orange"]+[200],"high":CBL["red"]+[220]}
                heat=[col_map.get(lbl,CBL["green"]+[180]) for lbl in labels]
            else: heat=[CBL["green"]+[180] for _ in static_path_coords]
            heat_df=pd.DataFrame([{"path":static_path_coords,"colors":heat}])
            heat_layer=pdk.Layer("PathLayer",data=heat_df,get_path="path",get_color="colors",width_scale=4,width_min_pixels=3)

            # Sensors
            vis=[]
            if isinstance(sensors_df,pd.DataFrame) and not sensors_df.empty:
                for r in sensors_df.itertuples():
                    modality=getattr(r,"modality",None); qualS=getattr(r,"qualS","GOOD"); label=getattr(r,"label","low")
                    if modality=="RAW": color=CBL["blue"]+[255]
                    elif modality=="HYBRID": color=CBL["cyan"]+[255]
                    elif modality=="SEMANTIC": color=CBL["purple"]+[255]
                    else: color={"GOOD":CBL["green"]+[230],"PATCHY":CBL["orange"]+[230],"POOR":CBL["red"]+[230]}.get(qualS,CBL["gray"]+[220])
                    vis.append({"sid":r.sid,"lon":float(r.lon),"lat":float(r.lat),"color":color,
                                "tooltip":f"{r.sid} • {label} • {qualS} • {modality or '—'}"})
            sens_df=pd.DataFrame(vis)
            sens_layer=pdk.Layer("ScatterplotLayer",data=sens_df,get_position='[lon,lat]',get_fill_color='color',
                                 get_radius=2800,stroked=True,get_line_color=[0,0,0],line_width_min_pixels=1.2,pickable=True)
            text_layer=pdk.Layer("TextLayer",data=sens_df,get_position='[lon,lat]',get_text='sid',get_size=14,get_color=[20,20,20])

            # TSR polygons
            tsr_layer=pdk.Layer("PolygonLayer",
                data=[{"polygon":p["polygon"],"tooltip":f"TSR {p['speed']} km/h"} for p in tsr_list],
                get_polygon="polygon",get_fill_color=CBL["gold"]+[80],get_line_color=CBL["gold"],line_width_min_pixels=1,pickable=True)

            # Train
            halo={"GOOD":CBL["green"]+[210],"PATCHY":CBL["orange"]+[210],"POOR":CBL["red"]+[230]}.get(quality_macro,CBL["green"]+[210])
            cur=pd.DataFrame([{"lat":train_pos[0],"lon":train_pos[1],
                               "icon_data":{"url":"https://img.icons8.com/emoji/48/train-emoji.png","width":128,"height":128,"anchorY":128}}])
            halo_layer=pdk.Layer("ScatterplotLayer",data=cur,get_position='[lon,lat]',get_fill_color=halo,get_radius=5200,
                                 stroked=True,get_line_color=[0,0,0],line_width_min_pixels=1)
            icon_layer=pdk.Layer("IconLayer",data=cur,get_position='[lon,lat]',get_icon='icon_data',get_size=4,size_scale=15)
            return [heat_layer,sens_layer,text_layer,tsr_layer,halo_layer,icon_layer]

        def deck_map(tsr_list,train_pos,sensors_df,quality_macro,use_tiles):
            base_layers=[l for l in [tile_layer,static_path_layer,static_bs_rings] if l is not None]
            dyn=dynamic_layers(tsr_list,train_pos,sensors_df,quality_macro)
            vs=pdk.ViewState(latitude=60.7, longitude=17.5, zoom=6.2)
            return pdk.Deck(layers=base_layers+dyn, initial_view_state=vs,
                            map_provider=None if not use_tiles else "carto",
                            map_style=None if not use_tiles else "light",
                            tooltip={"html":"<b>{tooltip}</b>", "style":{"color":"white","background":"rgba(0,0,0,0.7)","border-radius":"6px","padding":"4px"}})

        colRW,colTMS=st.columns(2)
        with colRW:
            st.markdown('<span class="ribbon">REAL WORLD</span>', unsafe_allow_html=True)
            st.pydeck_chart(deck_map(st.session_state.tsr_polys_real,trainA,sensors,quality,use_tiles),
                            use_container_width=True,height=520)
        with colTMS:
            st.markdown('<span class="ribbon">TMS VIEW</span>', unsafe_allow_html=True)
            st.pydeck_chart(deck_map(st.session_state.tsr_polys_tms,trainA,sensors,quality,use_tiles),
                            use_container_width=True,height=520)

        # Cache minimal frame bits
        st.session_state._frame.update({"t":t,"trainA":trainA,"segment":seg,"quality":quality,"sensors":sensors})

        # ---- Alerts box ----
        with alerts_box:
            if st.session_state.alerts_feed:
                for row in reversed(st.session_state.alerts_feed):
                    st.write("• "+row)
            else:
                st.caption("No Lane-A alerts in this window.")

        # ---- Affected sensors table ----
        with affected_box:
            aff=sensors[(sensors["label"]!="low") | (sensors["modality"]!="RAW")].copy()
            if aff.empty:
                st.caption("No medium/high risks or constrained uplinks this second.")
            else:
                show=(aff.sort_values(["label","score"],ascending=[False,False])
                        [["sid","segment","label","score","qualS","modality","temp","strain","ballast","battery_V"]]
                        .rename(columns={"qualS":"uplink","label":"risk"}))
                show["score"]=(show["score"]*100).round(0).astype(int).astype(str)+"%"
                st.dataframe(show, height=240, use_container_width=True)

        # ---- Telemetry arrays for charts ----
        if "_times" not in st.session_state:
            st.session_state._times=np.full(SECS,np.nan)
            st.session_state.arr={k:np.full(SECS,np.nan) for k in
                ["raw_bits","laneA_bits","laneB_bits","cap_bits","snr_db","laneA_succ","lat_ms"]}
            st.session_state.qual=np.array([""]*SECS,dtype=object); st.session_state.seg=np.array([""]*SECS,dtype=object)
        if math.isnan(st.session_state._times[t]):
            st.session_state._times[t]=t
            st.session_state.arr["raw_bits"][t]=raw_bps
            st.session_state.arr["laneA_bits"][t]=laneA_bps
            st.session_state.arr["laneB_bits"][t]=laneB_bps
            st.session_state.arr["cap_bits"][t]=cap_bps_train
            st.session_state.arr["snr_db"][t]=snr_use
            st.session_state.arr["laneA_succ"][t]=laneA_success
            st.session_state.arr["lat_ms"][t]=lat_ms
            st.session_state.qual[t]=quality; st.session_state.seg[t]=seg

        arr=st.session_state.arr; x=np.arange(SECS); t_cur=t
        s=lambda k: [None if (isinstance(v,float) and math.isnan(v)) else v for v in arr[k]]

        with t1:
            fig=go.Figure()
            fig.add_scatter(x=x,y=s("raw_bits"),name="RAW (bps)",mode="lines")
            fig.add_scatter(x=x,y=s("laneA_bits"),name="Lane-A (bps)",mode="lines")
            fig.add_scatter(x=x,y=s("laneB_bits"),name="Lane-B (bps)",mode="lines")
            fig.add_scatter(x=x,y=s("cap_bits"),name="Capacity (bps)",mode="lines")
            fig.add_vline(x=t_cur,line_width=2,line_dash="dash",line_color="gray")
            fig.update_layout(height=210,margin=dict(l=10,r=10,t=10,b=10),legend=dict(orientation="h",y=1.02))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        with t2:
            fig=go.Figure()
            fig.add_scatter(x=x,y=s("lat_ms"),name="Latency (ms)",mode="lines")
            fig.add_vline(x=t_cur,line_width=2,line_dash="dash",line_color="gray")
            fig.update_layout(height=180,margin=dict(l=10,r=10,t=10,b=10),yaxis_title="ms")
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        with t3:
            snr=s("snr_db"); succ=[None if v is None else (float(v)*100.0) for v in s("laneA_succ")]
            fig=go.Figure()
            fig.add_scatter(x=x,y=snr,name="SNR (dB)",mode="lines",yaxis="y1")
            fig.add_scatter(x=x,y=succ,name="Lane-A success (%)",mode="lines",yaxis="y2")
            fig.add_vline(x=t_cur,line_width=2,line_dash="dash",line_color="gray")
            fig.update_layout(height=190,margin=dict(l=10,r=10,t=10,b=10),
                              yaxis=dict(title="SNR (dB)"),
                              yaxis2=dict(title="Success (%)",overlaying="y",side="right",range=[0,100]),
                              legend=dict(orientation="h",y=1.02))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    # ========================= Sensor Inspector (live) =========================
    st.markdown("### 🔎 Sensor Inspector (live)")

    # Controls
    colF1,colF2,colF3=st.columns(3)
    with colF1: auto_follow_risk=st.checkbox("Follow highest-risk",value=True,key="auto_follow_risk")
    with colF2: auto_follow_near=st.checkbox("Follow sensor near train",value=False,key="auto_follow_near")
    with colF3: pin_choice=st.checkbox("Pin selection",value=False,key="pin_sensor")

    sensors = st.session_state._frame["sensors"]
    if not sensors.empty:
        # choose default
        sid_risk=sensors.sort_values("score",ascending=False)["sid"].iloc[0]
        lat_t,lon_t=st.session_state._frame["trainA"]
        d2=((sensors["lat"]-lat_t)**2+(sensors["lon"]-lon_t)**2); sid_near=sensors.loc[d2.idxmin(),"sid"]
        if auto_follow_risk: sid_seed=sid_risk
        elif auto_follow_near: sid_seed=sid_near
        else: sid_seed=sid_risk
    else:
        sid_seed="S00"

    if "sid_manual" not in st.session_state: st.session_state.sid_manual=sid_seed
    if pin_choice: sid_seed=st.session_state.sid_manual

    sid_pick = st.selectbox("Choose sensor", options=list(st.session_state.sensor_hist.keys()),
                            index=list(st.session_state.sensor_hist.keys()).index(sid_seed), key="sensor_selectbox")
    if pin_choice: st.session_state.sid_manual=sid_pick

    hist=st.session_state.sensor_hist.get(sid_pick,[])
    if not hist:
        st.info("No history yet — press ▶ Play.")
    else:
        hdf=pd.DataFrame(hist)
        r_now=sensors.loc[sensors["sid"]==sid_pick].iloc[0]
        modality=r_now["modality"]; uplink=r_now["qualS"]; score=float(r_now["score"]); exceeded=r_now["exceeded"]
        badge={"RAW":"🟦 RAW","HYBRID":"🟩 HYBRID","SEMANTIC":"🟪 SEMANTIC"}[modality]
        upcol={"GOOD":"🟢 GOOD","PATCHY":"🟠 PATCHY","POOR":"🔴 POOR"}[uplink]
        cA,cB,cC=st.columns([1.1,1.1,1])
        with cA: st.metric("Modality (uplink policy)",badge); st.metric("Uplink quality",upcol)
        with cB: st.metric("Risk (now)",f"{score*100:.0f}%"); st.metric("Exceeded",exceeded or "—")
        with cC: st.metric("Battery (V)",f"{float(r_now['battery_V']):.2f}"); st.metric("Hotspot","Yes" if bool(r_now["hotspot"]) else "No")

        x=hdf["t"]; fig_ts=go.Figure()
        fig_ts.add_scatter(x=x,y=hdf["rail_temp_C"],name="Rail temp (°C)",mode="lines")
        fig_ts.add_scatter(x=x,y=hdf["strain_kN"],name="Strain (kN)",mode="lines",yaxis="y2")
        fig_ts.update_layout(height=230,margin=dict(l=8,r=8,t=24,b=8),
                             yaxis=dict(title="°C"),yaxis2=dict(title="kN",overlaying="y",side="right"),
                             legend=dict(orientation="h",y=1.02))
        st.plotly_chart(fig_ts,use_container_width=True,config={"displayModeBar":False})

        fig_sun=go.Figure()
        fig_sun.add_scatter(x=x,y=hdf["light_lux"],name="Light (lux)",mode="lines")
        fig_sun.add_scatter(x=x,y=hdf["irradiance_Wm2"],name="Irradiance (W/m²)",mode="lines",yaxis="y2")
        fig_sun.update_layout(height=200,margin=dict(l=8,r=8,t=24,b=8),
                              yaxis=dict(title="lux"),yaxis2=dict(title="W/m²",overlaying="y",side="right"),
                              legend=dict(orientation="h",y=1.02))
        st.plotly_chart(fig_sun,use_container_width=True,config={"displayModeBar":False})

        fig_wb=go.Figure()
        fig_wb.add_scatter(x=x,y=hdf["wind_speed_ms"],name="Wind (m/s)",mode="lines")
        fig_wb.add_scatter(x=x,y=hdf["battery_V"],name="Battery (V)",mode="lines",yaxis="y2")
        fig_wb.update_layout(height=200,margin=dict(l=8,r=8,t=24,b=8),
                             yaxis=dict(title="m/s"),yaxis2=dict(title="V",overlaying="y",side="right"),
                             legend=dict(orientation="h",y=1.02))
        st.plotly_chart(fig_wb,use_container_width=True,config={"displayModeBar":False})

        fig_risk=go.Figure()
        fig_risk.add_scatter(x=x,y=hdf["risk_score"]*100,name="Risk (%)",mode="lines")
        fig_risk.add_hline(y=85,line_color="red",line_dash="dot",annotation_text="TSR/STOP candidate (85%)")
        fig_risk.update_layout(height=160,margin=dict(l=8,r=8,t=24,b=8),yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_risk,use_container_width=True,config={"displayModeBar":False})

        last_N=12
        events=hdf[hdf["exceeded"]!=""][["t","rail_temp_C","strain_kN","ballast_idx","risk_score","exceeded"]].tail(last_N)
        st.caption(f"Semantic events (last {last_N} for {sid_pick})")
        if events.empty: st.write("• None in window.")
        else: st.dataframe(events.rename(columns={"risk_score":"risk"}),height=200,use_container_width=True)

# -------------------- Comm Flow --------------------
with tab_flow:
    st.subheader("Communication Flow")
    t=st.session_state.t_idx; arr=st.session_state.get("arr",None); bearer=st.session_state.bearer
    if not arr: st.info("Press ▶ Play to populate telemetry.")
    else:
        getv=lambda k:(int(arr[k][t]) if (not math.isnan(arr[k][t])) else 0)
        raw_bps=getv("raw_bits"); laneA_bps=getv("laneA_bits"); laneB_bps=getv("laneB_bits"); cap_bps=getv("cap_bits")
        sensors_to_bs=max(1, raw_bps+laneA_bps+laneB_bps); bs_to_core=sensors_to_bs; core_to_tms=sensors_to_bs
        tms_to_train=max(1, laneA_bps+laneB_bps); tms_to_maint=max(1, 100 if laneB_bps>0 else 0)
        nodes=["Sensors","BS/Edge",f"Network ({bearer})","TMS","Train DAS","Maintenance"]; idx={n:i for i,n in enumerate(nodes)}
        fig=go.Figure(data=[go.Sankey(
            node=dict(label=nodes),
            link=dict(
                source=[idx["Sensors"],idx["BS/Edge"],idx[f"Network ({bearer})"],idx["TMS"],idx["TMS"]],
                target=[idx["BS/Edge"],idx[f"Network ({bearer})"],idx["TMS"],idx["Train DAS"],idx["Maintenance"]],
                value=[sensors_to_bs,bs_to_core,core_to_tms,tms_to_train,tms_to_maint],
                label=["uplink data+alerts","backhaul","to TMS","advisories/TSR/STOP","work orders"]
            )
        )])
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig,use_container_width=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("RAW (bps)",f"{raw_bps:,}"); c2.metric("Lane-A (bps)",f"{laneA_bps:,}")
        c3.metric("Lane-B (bps)",f"{laneB_bps:,}"); c4.metric("Capacity (kbps)",f"{cap_bps//1000:,}")

# -------------------- Ops --------------------
with tab_ops:
    st.subheader("Maintenance & Incidents")
    t=st.session_state.t_idx
    for w in st.session_state.work_orders:
        if w["status"]=="Dispatched" and t>=w["eta_done_idx"]: w["status"]="Resolved"
    resolved={tuple(map(tuple,w["polygon"])) for w in st.session_state.work_orders if w["status"]=="Resolved"}
    st.session_state.tsr_polys_real=[p for p in st.session_state.tsr_polys_real if tuple(map(tuple,p["polygon"])) not in resolved]
    st.session_state.tsr_polys_tms =[p for p in st.session_state.tsr_polys_tms  if tuple(map(tuple,p["polygon"])) not in resolved]

    if st.session_state.work_orders:
        rows=[dict(id=f"WO-{i+1:03d}",status=w["status"],created_s=w["created_idx"],eta_done_s=w["eta_done_idx"])
              for i,w in enumerate(st.session_state.work_orders)]
        st.dataframe(pd.DataFrame(rows),use_container_width=True)
    else:
        st.info("No active work orders. High-confidence buckling alerts will create them automatically.")

    if any(p.get("stop",False) for p in st.session_state.tsr_polys_tms): st.warning("STOP order in effect (TMS view).")
    if any((p in st.session_state.tsr_polys_real) and (p not in st.session_state.tsr_polys_tms) for p in st.session_state.tsr_polys_real):
        st.error("⚠️ Discrepancy: Real TSR exists that TMS may not know yet (risk of missed alert).")
