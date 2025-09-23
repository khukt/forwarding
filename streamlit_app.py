import time, math, io, json
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
import cbor2

st.set_page_config(page_title="ENSURE-6G — Live Semantic Communication Simulator", layout="wide")
st.title("ENSURE-6G: Live Semantic Communication Simulator (Rail • Winter • 6G)")
st.caption("Realistic time-series → features → events + codebook tokens → Lane A (safety) / Lane B (ops) packets • Coverage-aware transmission")

# -------------------------- Controls --------------------------
with st.sidebar:
    st.header("Simulation Controls")
    duration_min = st.slider("Simulated duration (minutes)", 5, 30, 12, step=1)
    seed = st.number_input("Random seed", value=42, step=1)
    st.markdown("---")
    st.subheader("Winter conditions")
    ambient_c = st.slider("Ambient temp (°C)", -30, 10, -12)
    snowfall = st.select_slider("Snowfall", options=["none","light","moderate","heavy"], value="moderate")
    icing = st.select_slider("Icing risk", options=["low","medium","high"], value="high")
    st.markdown("---")
    st.subheader("Coverage profile")
    coverage_profile = st.selectbox("Rural 5G/6G coverage", ["Good", "Patchy (realistic)", "Poor"])
    strategy = st.radio("Transmit strategy", ["Raw", "Semantic", "Adaptive (prefer Semantic)"], index=2)
    ensure_events = st.checkbox("Guarantee visible events", value=True)
    st.markdown("---")
    st.subheader("Codebook")
    k_codebook = st.select_slider("k-means codebook size (z tokens)", options=[32, 64, 128, 256], value=128)
    show_cbor = st.checkbox("Use CBOR for Lane B packets (smaller than JSON)", value=True)

st.info("Tip: Use 'Patchy (realistic)' coverage + moderate snow + high icing to see semantic resilience vs raw.")

# -------------------------- Constants --------------------------
FS_ACCEL = 200   # Hz per axis
FS_FORCE = 500   # Hz pantograph
BITS_PER_SAMPLE = 16  # assume int16 raw
SECS = duration_min * 60
rng = np.random.default_rng(int(seed))

# Coverage mask generator
def build_coverage_mask(n_steps: int, profile: str, rng) -> np.ndarray:
    if profile == "Good":
        return (rng.random(n_steps) > 0.1).astype(np.uint8)  # ~90% available
    if profile == "Poor":
        return (rng.random(n_steps) > 0.6).astype(np.uint8)  # ~40% available
    # Patchy: deterministic long gaps plus microfades
    mask = np.ones(n_steps, dtype=np.uint8)
    # two rural long gaps
    for start_frac in [0.22, 0.56]:
        start = int(n_steps * start_frac)
        gap = int(n_steps * 0.08)
        mask[start:start+gap] = 0
    # random microfades
    idxs = rng.integers(0, n_steps, size=max(1, n_steps//20))
    mask[idxs] = 0
    return mask

coverage = build_coverage_mask(SECS, coverage_profile, rng)

# -------------------------- Signal synthesis (per-second windows) --------------------------
# We do not store raw waveforms to keep it light. We synthesize features directly from parameterized signals.

@dataclass
class Feat:
    t: int
    seg: str
    rms: float
    crest: float
    band_20_50: float
    band_50_120: float
    jerk: float
    temp_peak: float
    temp_ewma: float
    temp_slope_cpm: float
    slip_ratio: float
    wsp_count: int
    panto_varN: float

# Route segments (km approximate weights for label only)
ROUTE = [("Sundsvall","Hudiksvall",120), ("Hudiksvall","Söderhamn",50), ("Söderhamn","Gävle",75), ("Gävle","Uppsala",100), ("Uppsala","Stockholm",70)]
total_km = sum(x[2] for x in ROUTE)

def section_by_time(t: int) -> str:
    pos = (t/SECS) * total_km
    acc = 0
    for a,b,l in ROUTE:
        if pos <= acc + l:
            return f"{a}→{b}"
        acc += l
    a,b,_ = ROUTE[-1]; return f"{a}→{b}"

# Winter factors
snow_factor = {"none":0.3, "light":0.7, "moderate":1.0, "heavy":1.5}[snowfall]
ice_factor  = {"low":0.6, "medium":1.0, "high":1.4}[icing]

# Feature generation model (statistically plausible)
def synth_features(t: int) -> Feat:
    seg = section_by_time(t)
    # Baseline vibration features
    base_rms = 0.18 + 0.05*np.sin(2*np.pi*t/180.0)
    # Snow increases low/mid band power and rms
    rms = abs(base_rms * (1 + 0.6*(snow_factor-0.3)) + rng.normal(0, 0.02))
    band_20_50 = max(0.0, 0.12*(snow_factor) + rng.normal(0, 0.015))
    band_50_120 = max(0.0, 0.08*(snow_factor) + rng.normal(0, 0.015))
    crest = 2.2 + 0.2*(snow_factor) + rng.normal(0, 0.05)
    jerk = 0.6 + 0.3*(snow_factor) + rng.normal(0, 0.05)

    # Temperature (bearing) dynamics
    temp_base = 45 + 0.02*t + (0 if ambient_c < -15 else 2)  # colder outside → slower rise
    temp_noise = rng.normal(0, 0.3)
    temp_inst = temp_base + temp_noise + (3.0 if snow_factor>1.2 and (t%600>300) else 0.0)
    # Simple EWMA & slope estimates (simulate)
    ewma = 0.8*temp_base + 0.2*temp_inst
    slope_cpm = (0.8 + 0.4*(ice_factor-1.0))  # °C per minute approx
    temp_peak = temp_inst + max(0, rng.normal(0.5, 0.2))

    # Slip ratio and WSP
    slip_ratio = max(0.0, rng.normal(0.05, 0.01) + 0.08*(snow_factor-0.3))
    wsp_count = int(max(0, rng.normal(5, 2) + 12*(snow_factor-0.3)))

    # Pantograph variance (ice -> higher)
    panto_varN = max(0.0, rng.normal(40, 8) + 25*(ice_factor-1.0))

    return Feat(t, seg, float(rms), float(crest), float(band_20_50), float(band_50_120),
                float(jerk), float(temp_peak), float(ewma), float(slope_cpm),
                float(slip_ratio), int(wsp_count), float(panto_varN))

# Generate per-second features
features: List[Feat] = [synth_features(t) for t in range(SECS)]

# -------------------------- Event logic (dwell & hysteresis) --------------------------
@dataclass
class Event:
    t: int
    intent: str
    slots: Dict[str, object]

events: List[Event] = []
# thresholds (tunable)
RMS_HIGH = 0.35
SLIP_HIGH = 0.18
TEMP_HIGH = 85.0
PANTO_VAR_HIGH = 80.0
DWELL_S = 120

# state
rms_above_since = None
slip_above_since = None
temp_above_since = None
panto_above_since = None

for f in features:
    # ride degradation
    if f.rms >= RMS_HIGH:
        rms_above_since = f.t if rms_above_since is None else rms_above_since
    else:
        rms_above_since = None
    if rms_above_since is not None and f.t - rms_above_since >= DWELL_S:
        events.append(Event(f.t, "ride_degradation", {"segment": f.seg, "rms": round(f.rms,3), "dwell_s": f.t - rms_above_since}))
        rms_above_since = None  # emit sparsely

    # low adhesion
    if f.slip_ratio >= SLIP_HIGH:
        slip_above_since = f.t if slip_above_since is None else slip_above_since
    else:
        slip_above_since = None
    if slip_above_since is not None and f.t - slip_above_since >= 60:
        km = round((f.t/SECS)*sum(x[2] for x in ROUTE), 1)
        events.append(Event(f.t, "low_adhesion_event", {"km": km, "slip_ratio": round(f.slip_ratio,3), "duration_s": f.t - slip_above_since}))
        slip_above_since = None

    # bearing overtemp
    if f.temp_peak >= TEMP_HIGH:
        temp_above_since = f.t if temp_above_since is None else temp_above_since
    else:
        temp_above_since = None
    if temp_above_since is not None and f.t - temp_above_since >= 180:
        events.append(Event(f.t, "bearing_overtemp", {"axle": "2L", "peak_c": round(f.temp_peak,1), "dwell_s": f.t - temp_above_since}))
        temp_above_since = None

    # pantograph ice
    if f.panto_varN >= PANTO_VAR_HIGH and ambient_c <= -8:
        panto_above_since = f.t if panto_above_since is None else panto_above_since
    else:
        panto_above_since = None
    if panto_above_since is not None and f.t - panto_above_since >= 90:
        events.append(Event(f.t, "pantograph_ice", {"varN": int(f.panto_varN), "temp_c": ambient_c}))
        panto_above_since = None

# Guarantee events for demo if selected
if ensure_events and len(events) == 0:
    # inject one of each
    t_inj = min(SECS-10, 300)
    events.extend([
        Event(t_inj, "ride_degradation", {"segment": section_by_time(t_inj), "rms": 0.42, "dwell_s": 180}),
        Event(t_inj+20, "low_adhesion_event", {"km": 200.3, "slip_ratio": 0.22, "duration_s": 90}),
        Event(t_inj+40, "pantograph_ice", {"varN": 95, "temp_c": ambient_c}),
    ])

# -------------------------- Codebook tokens (k-means on feature vectors) --------------------------
# Build feature matrix
X = np.array([[f.rms, f.crest, f.band_20_50, f.band_50_120, f.jerk] for f in features], dtype=np.float32)
# We fit k-means on this run (small k, fast); in production you'd preload centroids.
kmeans = KMeans(n_clusters=int(k_codebook), n_init=5, random_state=int(seed)).fit(X)
tokens = kmeans.predict(X)  # z per second
# Optional: simple human-readable labels for a few clusters (demo)
# We'll assign labels by centroid RMS/band power ranking
centroids = kmeans.cluster_centers_
cent_rms = centroids[:,0]
order = np.argsort(cent_rms)
labels = ["smooth"]*len(centroids)
if len(centroids) >= 4:
    labels[order[-1]] = "rough-snow"
    labels[order[-2]] = "curve-rough"
    labels[order[0]]  = "very-smooth"

# -------------------------- Packetization + Network --------------------------
# Lane A: safety telegram examples (fixed fields, tiny)
def laneA_adhesion_state(f: Feat) -> Dict[str, int]:
    mu_est = max(0.0, min(0.6, 0.6 - 0.5*f.slip_ratio))  # crude lower bound
    return {
        "mu_q7_9": int((mu_est) * (1<<9)),   # fixed-point
        "conf_pct": int(max(0, min(100, 100 - 120*abs(0.2-f.slip_ratio)))),
        "slip": int(f.slip_ratio >= SLIP_HIGH),
        "wsp": int(min(255, f.wsp_count)),
        "valid_ms": 500
    }

def encode_laneA(pkt: Dict[str,int], seq: int) -> bytes:
    # Serialize deterministically: tiny CBOR-like but using cbor2 for convenience
    payload = dict(pkt); payload.update({"seq": seq})
    b = cbor2.dumps(payload)  # in practice you might use fixed-width binary
    return b

# Lane B: operational events (JSON/CBOR)
def encode_laneB(event: Event, use_cbor: bool) -> bytes:
    d = {"i": event.intent, "ts": int(event.t), "s": event.slots}
    if use_cbor:
        return cbor2.dumps(d)
    return json.dumps(d, separators=(",",":")).encode()

# Transmission simulation
per_sec_bits = np.zeros(SECS, dtype=np.int64)
delivered = 0; dropped = 0
laneA_msgs = 0
laneB_msgs = 0

# Raw budget if attempted (per second)
raw_bits_per_sec = (
    FS_ACCEL*3*BITS_PER_SAMPLE +
    FS_FORCE*BITS_PER_SAMPLE +
    4*1*BITS_PER_SAMPLE +   # temps 1 Hz x4
    400                     # GNSS NMEA-ish bits/s
)

# For each second, decide what to send based on strategy & coverage
seqA = 0
laneB_trace = []
for t in range(SECS):
    cov = coverage[t] == 1

    # Lane A: adhesion telegram every 1s
    pa = laneA_adhesion_state(features[t])
    ba = encode_laneA(pa, seqA); seqA += 1
    laneA_msgs += 1
    if cov:
        per_sec_bits[t] += len(ba)*8
        delivered += 1
    else:
        dropped += 1

    # Strategy behaviors for raw/semantic
    if strategy in ["Raw","Adaptive (prefer Semantic)"]:
        if cov:
            # In Adaptive, we lower raw near coverage holes (we model as 70% reduction)
            add = raw_bits_per_sec
            if strategy == "Adaptive (prefer Semantic)":
                if (t>0 and coverage[t-1]==0) or (t<SECS-1 and coverage[t+1]==0):
                    add = int(add * 0.3)
            per_sec_bits[t] += add

    # Lane B: send an event if any occurring at t
    for ev in [e for e in events if e.t == t]:
        bb = encode_laneB(ev, show_cbor)
        laneB_msgs += 1
        if cov:
            per_sec_bits[t] += len(bb)*8
            delivered += 1
            laneB_trace.append({"t": t, "intent": ev.intent, "slots": ev.slots, "bytes": len(bb), "encoding": "CBOR" if show_cbor else "JSON", "segment": section_by_time(t), "token_z": int(tokens[t]), "token_label": labels[tokens[t]]})
        else:
            dropped += 1

# -------------------------- Overview Metrics --------------------------
theoretical_raw_bits = raw_bits_per_sec * SECS
actual_sent_bits = int(per_sec_bits.sum())
saved = 1.0 - (actual_sent_bits / max(theoretical_raw_bits,1))

c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Duration (min)", duration_min)
with c2: st.metric("Raw if streamed (MB)", f"{theoretical_raw_bits/(8*1024*1024):.2f}")
with c3: st.metric("Actual sent (MB)", f"{actual_sent_bits/(8*1024*1024):.2f}")
with c4: st.metric("Bandwidth saved vs raw", f"{100*saved:.1f}%")

c5,c6,c7 = st.columns(3)
with c5: st.metric("Lane A telegrams", laneA_msgs)
with c6: st.metric("Lane B events", laneB_msgs)
with c7: st.metric("Messages delivered", delivered)

# -------------------------- Charts --------------------------
st.subheader("Transmitted bits per second & coverage")
df = pd.DataFrame({"t": np.arange(SECS), "bits": per_sec_bits, "coverage": coverage})
st.line_chart(df.set_index("t")[["bits"]])
st.area_chart(df.set_index("t")[["coverage"]])

# -------------------------- Event Table (Lane B) --------------------------
st.subheader("Operational semantic events (Lane B)")
if laneB_trace:
    st.dataframe(pd.DataFrame(laneB_trace))
else:
    st.info("No Lane B events fired under current conditions. Toggle 'Guarantee visible events' or increase snow/icing.")

# -------------------------- Packet Inspector --------------------------
st.subheader("Packet inspector")
example_t = min(SECS-1, max(0, SECS//3))
example_ev = next((e for e in events if e.t >= example_t), None)
colL, colR = st.columns(2)

with colL:
    st.markdown("**Lane A (safety) adhesion_state**")
    pktA = laneA_adhesion_state(features[example_t])
    bA = encode_laneA(pktA, 123)
    st.code(json.dumps(pktA, indent=2))
    st.write(f"Encoded size: {len(bA)} bytes (CBOR)")

with colR:
    st.markdown("**Lane B (ops) example event**")
    if example_ev is None:
        example_ev = Event(example_t, "ride_degradation", {"segment": section_by_time(example_t), "rms": 0.41, "dwell_s": 160})
    bB_json = encode_laneB(example_ev, use_cbor=False)
    bB_cbor = encode_laneB(example_ev, use_cbor=True)
    st.code(json.dumps({"i": example_ev.intent, "ts": example_ev.t, "s": example_ev.slots}, indent=2))
    st.write(f"JSON size: {len(bB_json)} bytes • CBOR size: {len(bB_cbor)} bytes")

# -------------------------- Download trace --------------------------
st.subheader("Download delivered Lane B events (for analysis)")
delivered_events = [e for e in laneB_trace]
st.download_button("Download JSON", data=json.dumps(delivered_events, indent=2), file_name="laneB_events.json", mime="application/json")

st.markdown("---")
st.caption("Live simulation • Realistic per-second features and events • Codebook tokens (k-means) • Coverage-aware transmission • Lane A safety telegrams are tiny and bounded; Lane B carries rich ops semantics.")
