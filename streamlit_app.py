import json, zlib, math, time, os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import cbor2
import pydeck as pdk

# ========= Page / session =========
st.set_page_config(page_title="ENSURE-6G • Semantic Rail Demo (OSM Map)", layout="wide")
st.title("ENSURE-6G: Semantic Communication — Live Rail Demo (OpenStreetMap)")
st.caption("OpenStreetMap basemap • Approximated railway Sundsvall→Stockholm • Base stations + coverage • Moving train • Safety vs Ops semantics")

if "t_idx" not in st.session_state: st.session_state.t_idx = 0
if "playing" not in st.session_state: st.session_state.playing = False

# ========= Sidebar =========
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

# ========= Geography: approximated railway polyline =========
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
CMD_CENTER = ("Trafikledning Stockholm", 59.3326, 18.0649)

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

def nearest_bs_quality(lat, lon):
    best=None
    for name, blat, blon, r in BASE_STATIONS:
        d=haversine
