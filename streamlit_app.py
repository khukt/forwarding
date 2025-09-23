import time, json, zlib
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict
from sklearn.cluster import KMeans
import cbor2

st.set_page_config(page_title="ENSURE-6G — Semantic Comms with Channel Model", layout="wide")
st.title("ENSURE-6G: Semantic Communication • Live Simulation with Channel Modeling")
st.caption("Sensors → features → events/tokens → Lane A (safety) / Lane B (ops) → PRIORITY SCHEDULER over a bursty channel with capacity, latency, jitter, loss, BER, CRC")

# ───────────────────────────────── Controls ─────────────────────────────────
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
    coverage_profile = st.selectbox("Rural 5G/FRMCS coverage", ["Good", "Patchy (realistic)", "Poor"])
    strategy = st.radio("Transmit strategy", ["Raw", "Semantic", "Adaptive (prefer Semantic)"], index=2)
    ensure_events = st.checkbox("Guarantee visible events", value=True)
    st.markdown("---")
    st.subheader("Codebook (k-means)")
    k_codebook = st.select_slider("Codebook size (z tokens)", options=[32, 64, 128, 256], value=128)
    show_cbor = st.checkbox("Use CBOR for Lane B packets", value=True)
    st.markdown("---")
    st.subheader("Channel model")
    base_capacity_kbps = st.slider("Base capacity (kbps)", 64, 2048, 512, 64)
    burst_factor = st.slider("Burst capacity factor", 1.0, 4.0, 2.0, 0.1,
                             help="During good seconds, capacity = base × factor")
    ge_good_p = st.slider("Gilbert-Elliott: P(stay GOOD)", 0.70, 0.99, 0.90, 0.01)
    ge_bad_p  = st.slider("Gilbert-Elliott: P(stay BAD)", 0.70, 0.99, 0.85, 0.01)
    good_loss_pct = st.slider("Loss in GOOD state (%)", 0.0, 5.0, 0.5, 0.1)
    bad_loss_pct  = st.slider("Loss in BAD state (%)", 5.0, 60.0, 20.0, 1.0)
    ber_good = st.slider("Bit error rate in GOOD", 0.0, 5e-6, 1e-6, 1e-6, format="%.6f")
    ber_bad  = st.slider("Bit error rate in BAD", 0.0, 5e-5, 2e-5, 1e-6, format="%.6f")
    base_latency_ms = st.slider("Base latency (ms)", 10, 300, 80, 10)
    jitter_ms = st.slider("Random jitter (±ms)", 0, 150, 40, 5)
    reorder_prob = st.slider("Reordering probability", 0.0, 0.2, 0.05, 0.01)
    dup_prob = st.slider("Duplication probability", 0.0, 0.2, 0.02, 0.01)

st.info("Tip: Patchy + higher BAD loss/BER shows why semantic + priority scheduling protects Lane A.")

# ─────────────────────────── Constants & helpers ───────────────────────────
FS_ACCEL = 200     # Hz per axis
FS_FORCE = 500     # Hz pantograph
BITS_PER_SAMPLE = 16
SECS = duration_min * 60
rng = np.random.default_rng(int(seed))

ROUTE = [("Sundsvall","Hudiksvall",120), ("Hudiksvall","Söderhamn",50),
         ("Söderhamn","Gävle",75), ("Gävle","Uppsala",100), ("Uppsala","Stockholm",70)]
total_km = sum(x[2] for x in ROUTE)

def section_by_time(t: int) -> str:
    pos = (t/SECS) * total_km
    acc = 0
    for a,b,l in ROUTE:
        if pos <= acc + l:
            return f"{a}→{b}"
        acc += l
    a,b,_ = ROUTE[-1]; return f"{a}→{b}"

# ───────────────────── Coverage mask (macro availability) ──────────────────
def build_coverage_mask(n_steps: int, profile: str, rng) -> np.ndarray:
    if profile == "Good":
        return (rng.random(n_steps) > 0.1).astype(np.uint8)
    if profile == "Poor":
        return (rng.random(n_steps) > 0.6).astype(np.uint8)
    mask = np.ones(n_steps, dtype=np.uint8)
    for start_frac in [0.22, 0.56]:
        start = int(n_steps * start_frac)
        gap = int(n_steps * 0.08)
        mask[start:start+gap] = 0
    idxs = rng.integers(0, n_steps, size=max(1, n_steps//20))
    mask[idxs] = 0
    return mask

coverage = build_coverage_mask(SECS, coverage_profile, rng)

# ───────────────────────── Feature synthesis per second ────────────────────
@dataclass
class Feat:
    t: int; seg: str; rms: float; crest: float; band_20_50: float; band_50_120: float
    jerk: float; temp_peak: float; temp_ewma: float; temp_slope_cpm: float
    slip_ratio: float; wsp_count: int; panto_varN: float

snow_factor = {"none":0.3, "light":0.7, "moderate":1.0, "heavy":1.5}[snowfall]
ice_factor  = {"low":0.6, "medium":1.0, "high":1.4}[icing]

def synth_features(t: int) -> Feat:
    seg = section_by_time(t)
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

features: List[Feat] = [synth_features(t) for t in range(SECS)]

# ───────────────────────────── Event detection ─────────────────────────────
@dataclass
class Event:
    t: int; intent: str; slots: Dict[str, object]

events: List[Event] = []
RMS_HIGH = 0.35; SLIP_HIGH = 0.18; TEMP_HIGH = 85.0; PANTO_VAR_HIGH = 80.0; DWELL_S = 120
rms_since=slip_since=temp_since=panto_since=None

for f in features:
    # ride degradation
    rms_since = f.t if (f.rms>=RMS_HIGH and rms_since is None) else (rms_since if f.rms>=RMS_HIGH else None)
    if rms_since is not None and f.t - rms_since >= DWELL_S:
        events.append(Event(f.t,"ride_degradation",{"segment":f.seg,"rms":round(f.rms,3),"dwell_s":f.t-rms_since}))
        rms_since=None
    # low adhesion
    slip_since = f.t if (f.slip_ratio>=SLIP_HIGH and slip_since is None) else (slip_since if f.slip_ratio>=SLIP_HIGH else None)
    if slip_since is not None and f.t - slip_since >= 60:
        km = round((f.t/SECS)*total_km,1)
        events.append(Event(f.t,"low_adhesion_event",{"km":km,"slip_ratio":round(f.slip_ratio,3),"duration_s":f.t-slip_since}))
        slip_since=None
    # overtemp
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
        Event(t0,"ride_degradation",{"segment":section_by_time(t0),"rms":0.42,"dwell_s":180}),
        Event(t0+20,"low_adhesion_event",{"km":200.3,"slip_ratio":0.22,"duration_s":90}),
        Event(t0+40,"pantograph_ice",{"varN":95,"temp_c":ambient_c}),
    ]

# ───────────────────── Codebook tokens (k-means on features) ───────────────
X = np.array([[f.rms, f.crest, f.band_20_50, f.band_50_120, f.jerk] for f in features], dtype=np.float32)
kmeans = KMeans(n_clusters=int(k_codebook), n_init=5, random_state=int(seed)).fit(X)
tokens = kmeans.predict(X)
centroids = kmeans.cluster_centers_; order = np.argsort(centroids[:,0])
labels = ["smooth"]*len(centroids)
if len(centroids)>=4:
    labels[order[-1]]="rough-snow"; labels[order[-2]]="curve-rough"; labels[order[0]]="very-smooth"

# ───────────────────────── Packet definitions & encode ─────────────────────
def laneA_adhesion_state(f: Feat) -> Dict[str,int]:
    mu_est = max(0.0, min(0.6, 0.6 - 0.5*f.slip_ratio))
    return {"mu_q7_9":int(mu_est*(1<<9)),
            "conf_pct":int(max(0,min(100,100-120*abs(0.2-f.slip_ratio)))),
            "slip":int(f.slip_ratio>=SLIP_HIGH),
            "wsp":int(min(255,f.wsp_count)),
            "valid_ms":500}

def encode_laneA(pkt: Dict[str,int], seq: int) -> bytes:
    payload = dict(pkt); payload.update({"seq":seq})
    b = cbor2.dumps(payload)
    crc = zlib.crc32(b).to_bytes(4, "big")
    return b + crc

def encode_laneB(event: Event, tkn:int, tlabel:str, use_cbor: bool) -> bytes:
    d = {"i":event.intent,"ts":int(event.t),"s":event.slots,"z":int(tkn),"zl":tlabel}
    if use_cbor:
        b = cbor2.dumps(d)
    else:
        b = json.dumps(d, separators=(",",":")).encode()
    crc = zlib.crc32(b).to_bytes(4,"big")
    return b + crc

# ───────────────────────────── Channel model ────────────────────────────────
@dataclass
class TxPkt:
    t_enq: int; lane: str; size_bits: int; seq: int; payload: bytes

# Gilbert-Elliott state per second
def ge_sequence(n:int, pGG:float, pBB:float, rng) -> np.ndarray:
    state = np.zeros(n, dtype=np.uint8)  # 1=GOOD, 0=BAD
    state[0] = 1
    for i in range(1,n):
        if state[i-1]==1:
            state[i] = 1 if rng.random()<pGG else 0
        else:
            state[i] = 0 if rng.random()<pBB else 1
    return state

ge = ge_sequence(SECS, ge_good_p, ge_bad_p, rng)

# Capacity time series (bits/s), modulated by coverage and GE state
cap_bits = np.zeros(SECS, dtype=np.int64)
for t in range(SECS):
    if coverage[t]==0:
        cap_bits[t]=0
    else:
        cap = base_capacity_kbps*1000
        if ge[t]==1:
            cap = int(cap * burst_factor)
        cap_bits[t]=cap

# Loss/BER by state
loss_pct_ts = np.where(ge==1, good_loss_pct, bad_loss_pct)
ber_ts = np.where(ge==1, ber_good, ber_bad)

# Scheduler: priority Lane A over Lane B, queues, latency/jitter, reorder/dup
laneA_q: List[TxPkt] = []
laneB_q: List[TxPkt] = []

raw_bits_per_sec = (
    FS_ACCEL*3*BITS_PER_SAMPLE +
    FS_FORCE*BITS_PER_SAMPLE +
    4*1*BITS_PER_SAMPLE +  # temps
    400                    # GNSS approx bits/s
)

seqA = 0; seqB = 0
delivered, dropped = 0, 0
per_sec_bits_sent = np.zeros(SECS, dtype=np.int64)
queue_len_A = np.zeros(SECS, dtype=np.int32)
queue_len_B = np.zeros(SECS, dtype=np.int32)

# For latency stats
latencies_A_ms = []
latencies_B_ms = []
delivered_events = []

def maybe_corrupt_and_check(payload: bytes, ber: float, rng) -> bool:
    """Flip bits with BER; return True if CRC ok after corruption, else False."""
    if len(payload)<5:  # payload + 4 CRC
        return False
    data, crc = payload[:-4], payload[-4:]
    # BER corruption: approximate by flipping k bits where k~Binomial(n_bits, ber)
    n_bits = len(data)*8
    flips = rng.binomial(n_bits, min(max(ber,0.0), 0.5))
    if flips>0:
        arr = bytearray(data)
        for _ in range(min(flips, 16)):  # cap flips for speed
            pos = rng.integers(0, len(arr))
            bit = 1 << rng.integers(0,8)
            arr[pos] ^= bit
        data = bytes(arr)
    crc_ok = zlib.crc32(data).to_bytes(4,"big") == crc
    return crc_ok

for t in range(SECS):
    # Enqueue one Lane A adhesion telegram each second
    pa = laneA_adhesion_state(features[t])
    ba = encode_laneA(pa, seqA); seqA += 1
    laneA_q.append(TxPkt(t_enq=t, lane="A", size_bits=len(ba)*8, seq=seqA, payload=ba))

    # Raw strategy
    want_raw = (strategy in ["Raw","Adaptive (prefer Semantic)"])
    send_raw_bits = 0
    if want_raw and coverage[t]==1:
        add = raw_bits_per_sec
        if strategy=="Adaptive (prefer Semantic)":
            if (t>0 and coverage[t-1]==0) or (t<SECS-1 and coverage[t+1]==0):
                add = int(add*0.3)
        # raw consumes capacity directly
        send_raw_bits = add

    # Lane B events at this second
    for ev in [e for e in events if e.t==t]:
        bb = encode_laneB(ev, tokens[t], labels[tokens[t]], show_cbor)
        laneB_q.append(TxPkt(t_enq=t, lane="B", size_bits=len(bb)*8, seq=seqB, payload=bb))
        seqB += 1

    # Service queues with available capacity cap_bits[t]
    budget = cap_bits[t] - send_raw_bits
    if budget < 0: budget = 0

    # Helper to transmit a packet with channel effects
    def try_tx(pkt: TxPkt, t_now:int) -> bool:
        nonlocal delivered, dropped
        # Loss by state (Gilbert-Elliott) before spending budget
        if rng.random() < loss_pct_ts[t_now]/100.0:
            dropped += 1
            return False
        # Consume capacity
        # (We assume if it's scheduled here, size_bits has already been subtracted)
        # Latency + jitter
        lat = base_latency_ms + rng.integers(-jitter_ms, jitter_ms+1)
        if rng.random() < reorder_prob:
            lat += rng.integers(20, 120)  # add extra delay for reordering
        # Bit errors
        ok = maybe_corrupt_and_check(pkt.payload, float(ber_ts[t_now]), rng)
        if not ok:
            dropped += 1
            return False
        # Duplication
        deliver_twice = (rng.random() < dup_prob)
        # Record delivery (with latency)
        if pkt.lane=="A":
            latencies_A_ms.append(max(0,lat))
        else:
            latencies_B_ms.append(max(0,lat))
        delivered += 1
        if deliver_twice:
            if pkt.lane=="A": latencies_A_ms.append(max(0,lat+5))
            else: latencies_B_ms.append(max(0,lat+5))
            delivered += 1
        return True

    # Serve Lane A first, then Lane B
    # We allow multiple packets per second subject to budget
    # Use FIFO within each lane
    # Lane A
    i = 0
    while i < len(laneA_q) and budget >= laneA_q[i].size_bits:
        pkt = laneA_q.pop(i)
        budget -= pkt.size_bits
        per_sec_bits_sent[t] += pkt.size_bits
        try_tx(pkt, t)
        # do not increment i because we popped
    # Lane B
    j = 0
    while j < len(laneB_q) and budget >= laneB_q[j].size_bits:
        pkt = laneB_q.pop(j)
        budget -= pkt.size_bits
        per_sec_bits_sent[t] += pkt.size_bits
        ok = try_tx(pkt, t)
        if ok:
            # decode for table (safe: payload may be JSON or CBOR)
            body = pkt.payload[:-4]
            try:
                if show_cbor:
                    d = cbor2.loads(body)
                else:
                    d = json.loads(body.decode())
                delivered_events.append({
                    "t_enq": pkt.t_enq,
                    "lane": pkt.lane,
                    "intent": d.get("i",""),
                    "segment": d.get("s",{}).get("segment", section_by_time(pkt.t_enq)),
                    "bytes": len(pkt.payload),
                    "encoding": "CBOR" if show_cbor else "JSON",
                    "z": d.get("z",""),
                    "zl": d.get("zl","")
                })
            except Exception:
                pass
        # do not increment j because we popped

    queue_len_A[t] = len(laneA_q)
    queue_len_B[t] = len(laneB_q)

# ───────────────────────── Overview metrics ────────────────────────────────
# Theoretical raw if streamed continuously (ignores channel caps)
theoretical_raw_bits = raw_bits_per_sec * SECS
actual_sent_bits = int(per_sec_bits_sent.sum())
saved = 1.0 - (actual_sent_bits / max(theoretical_raw_bits,1))

c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Duration (min)", duration_min)
with c2: st.metric("Raw if streamed (MB)", f"{theoretical_raw_bits/(8*1024*1024):.2f}")
with c3: st.metric("Actual sent (MB)", f"{actual_sent_bits/(8*1024*1024):.2f}")
with c4: st.metric("Bandwidth saved vs raw", f"{100*saved:.1f}%")

c5,c6,c7,c8 = st.columns(4)
with c5: st.metric("Lane A queued (max)", int(queue_len_A.max()))
with c6: st.metric("Lane B queued (max)", int(queue_len_B.max()))
with c7: st.metric("Delivered msgs", delivered)
with c8: st.metric("Dropped (loss/BER)", dropped)

# ─────────────────────────── Charts & tables ───────────────────────────────
st.subheader("Channel capacity, transmitted bits, and coverage")
df = pd.DataFrame({"t":np.arange(SECS),
                   "capacity_bits": cap_bits,
                   "bits_sent": per_sec_bits_sent,
                   "coverage": coverage,
                   "GE_state": ge})
st.line_chart(df.set_index("t")[["capacity_bits","bits_sent"]])
st.area_chart(df.set_index("t")[["coverage"]])

st.subheader("Queue sizes (priority scheduler)")
dfq = pd.DataFrame({"t":np.arange(SECS),
                    "LaneA_queue": queue_len_A,
                    "LaneB_queue": queue_len_B})
st.line_chart(dfq.set_index("t"))

st.subheader("Latency distributions")
colA, colB = st.columns(2)
with colA:
    if len(latencies_A_ms)>0:
        st.write(f"Lane A: {len(latencies_A_ms)} deliveries")
        st.bar_chart(pd.DataFrame({"lat_ms":latencies_A_ms}).clip(0,500))
    else:
        st.info("No Lane A deliveries recorded (check channel settings)")
with colB:
    if len(latencies_B_ms)>0:
        st.write(f"Lane B: {len(latencies_B_ms)} deliveries")
        st.bar_chart(pd.DataFrame({"lat_ms":latencies_B_ms}).clip(0,500))
    else:
        st.info("No Lane B deliveries recorded")

st.subheader("Operational semantic events delivered (Lane B)")
if delivered_events:
    st.dataframe(pd.DataFrame(delivered_events))
else:
    st.info("No Lane B events delivered. Try increasing capacity or lowering loss/BER.")

# ───────────────────────── Packet inspector ────────────────────────────────
st.subheader("Packet inspector (sizes include CRC32 footer)")
example_t = min(SECS-1, max(0, SECS//3))
pktA_demo = laneA_adhesion_state(features[example_t])
bA_demo = encode_laneA(pktA_demo, 123)
st.markdown("**Lane A (safety) adhesion_state**")
st.code(json.dumps(pktA_demo, indent=2))
st.write(f"Encoded size: {len(bA_demo)} bytes (CBOR+CRC)")

example_ev = next((e for e in events if e.t >= example_t), None)
if example_ev is None:
    example_ev = Event(example_t, "ride_degradation", {"segment": section_by_time(example_t), "rms": 0.41, "dwell_s": 160})
bB_json = encode_laneB(example_ev, tokens[example_t], labels[tokens[example_t]], use_cbor=False)
bB_cbor = encode_laneB(example_ev, tokens[example_t], labels[tokens[example_t]], use_cbor=True)
st.markdown("**Lane B (ops) example event**")
st.code(json.dumps({"i":example_ev.intent,"ts":example_ev.t,"s":example_ev.slots,"z":int(tokens[example_t]),"zl":labels[tokens[example_t]]}, indent=2))
st.write(f"JSON size: {len(bB_json)} bytes • CBOR size: {len(bB_cbor)} bytes (both include CRC)")

# ───────────────────────────── Download trace ──────────────────────────────
st.subheader("Download delivered Lane B events (JSON)")
st.download_button("Download JSON", data=json.dumps(delivered_events, indent=2),
                   file_name="laneB_events_delivered.json", mime="application/json")

st.markdown("---")
st.caption("This simulation includes: priority scheduling, time-varying capacity, Gilbert–Elliott burst losses, BER with CRC32 integrity check, latency/jitter, reordering & duplication, and coverage gaps. Lane A is prioritized and tiny; Lane B is best-effort.")
