
import json
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import streamlit as st

# ---------------------- App Config ----------------------
st.set_page_config(page_title="Semantic Communication Demo (ENSURE‑6G)", layout="wide")
st.title("Semantic Communication for Next‑Generation Networks — ENSURE‑6G")
st.caption("Realistic winter rail scenario: Sundsvall ↔ Stockholm • Transmit meaning, not raw sensor floods")

# ---------------------- Domain Defaults ----------------------
FS_ACCEL = 200           # Hz (bogie tri‑ax accelerometer)
FS_FORCE = 500           # Hz (pantograph contact force)
FS_TEMP  = 1             # Hz (bearing / ambient)
FS_GNSS  = 1             # Hz

BITS_PER_SAMPLE = 16     # assume 16‑bit raw for numeric sensors

ROUTE_SECTIONS = [
    ("Sundsvall", "Hudiksvall", 120),
    ("Hudiksvall", "Söderhamn", 50),
    ("Söderhamn", "Gävle", 75),
    ("Gävle", "Uppsala", 100),
    ("Uppsala", "Stockholm", 70),
]

# ---------------------- Controls ----------------------
with st.sidebar:
    st.header("Simulation Controls")
    duration_min = st.slider("Trip duration to simulate (minutes)", 10, 60, 20, step=5)
    accel_hz = st.slider("Bogie accelerometer rate (Hz)", 100, 400, FS_ACCEL, step=50)
    force_hz = st.slider("Pantograph force rate (Hz)", 200, 800, FS_FORCE, step=100)
    temp_hz  = st.slider("Bearing temp rate (Hz)", 1, 5, FS_TEMP, step=1)
    gnss_hz  = st.slider("GNSS rate (Hz)", 1, 5, FS_GNSS, step=1)

    st.markdown("---")
    st.subheader("Winter Conditions")
    outside_temp = st.slider("Ambient temperature (°C)", -30, 5, -12)
    snow_intensity = st.select_slider("Snowfall", options=["none","light","moderate","heavy"], value="moderate")
    icing_risk = st.select_slider("Icing risk", options=["low","medium","high"], value="high")

    st.markdown("---")
    st.subheader("Connectivity")
    coverage_profile = st.selectbox("Rural 5G/6G coverage profile", ["Good", "Patchy (realistic)", "Poor"])
    send_mode = st.radio("Transmit strategy", ["Raw streams", "Semantic events", "Adaptive (prefer Semantic)"], index=2)

# ---------------------- Helpers ----------------------
def bits_to_mb(bits: float) -> float:
    return bits / (8 * 1024 * 1024)

def section_by_time(t: float, total_s: int) -> str:
    # map time to route segment name for nice labels
    cum = 0
    seg_len = [s[2] for s in ROUTE_SECTIONS]
    total_len = sum(seg_len)
    pos = (t / total_s) * total_len
    c = 0
    for (a,b,l) in ROUTE_SECTIONS:
        if pos <= c + l:
            return f"{a}→{b}"
        c += l
    return f"{ROUTE_SECTIONS[-1][0]}→{ROUTE_SECTIONS[-1][1]}"

# coverage mask over time (0..1 fraction of time steps available)
def build_coverage_mask(n_steps: int, profile: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    if profile == "Good":
        # 90% availability
        return (rng.random(n_steps) > 0.1).astype(np.uint8)
    if profile == "Poor":
        # 40% availability
        return (rng.random(n_steps) > 0.6).astype(np.uint8)
    # Patchy: long rural gaps + some micro‑fades
    mask = np.ones(n_steps, dtype=np.uint8)
    # long gaps
    for start in [int(n_steps*0.2), int(n_steps*0.55)]:
        gap = int(n_steps*0.08)
        mask[start:start+gap] = 0
    # microfades
    for _ in range(int(n_steps*0.05)):
        idx = rng.integers(0, n_steps)
        mask[idx] = 0
    return mask

@dataclass
class Event:
    t: int
    intent: str
    slots: Dict[str, object]

def simulate(duration_min: int,
             accel_hz: int, force_hz: int, temp_hz: int, gnss_hz: int,
             outside_temp: float, snow_intensity: str, icing_risk: str,
             coverage_profile: str, send_mode: str):
    total_s = duration_min * 60
    # --- RAW sizes (bits) if streamed continuously ---
    accel_samples = total_s * accel_hz * 3  # tri‑ax
    accel_bits = accel_samples * BITS_PER_SAMPLE

    force_samples = total_s * force_hz
    force_bits = force_samples * BITS_PER_SAMPLE

    temp_samples = total_s * temp_hz * 4  # suppose 4 bearings
    temp_bits = temp_samples * BITS_PER_SAMPLE

    gnss_samples = total_s * gnss_hz
    # assume ~50 bytes per NMEA → 400 bits/sample
    gnss_bits = gnss_samples * 400

    raw_bits_total = accel_bits + force_bits + temp_bits + gnss_bits

    # --- Event generation rates (semantic) ---
    # Use winter settings to modulate expected anomalies
    snow_factor = {"none":0.2,"light":0.5,"moderate":1.0,"heavy":1.5}[snow_intensity]
    icing_factor = {"low":0.5,"medium":1.0,"high":1.5}[icing_risk]

    # Expected counts over whole trip
    ride_events = max(1, int(2 * snow_factor))           # ride_degradation events
    adhesion_events = max(1, int(2 * snow_factor))       # low_adhesion_event
    panto_events = int(1 * icing_factor)                 # pantograph_ice
    bearing_events = 1 if outside_temp < -10 else 0      # bearing_overtemp (rare, still possible)
    delay_events = 1                                     # delay_alert at a major station

    rng = np.random.default_rng(0)
    ev_times = np.clip(rng.integers(60, total_s-60, size=(ride_events+adhesion_events+panto_events+bearing_events+delay_events)), 0, total_s-1)
    ev_times.sort()

    events: List[Event] = []
    for i, t in enumerate(ev_times):
        seg = section_by_time(t, total_s)
        if i < ride_events:
            events.append(Event(t, "ride_degradation",
                                {"segment": seg, "severity": rng.choice(["low","medium","high"], p=[0.4,0.5,0.1]),
                                 "rms": round(float(rng.uniform(0.3, 0.8)), 2), "dwell_s": int(rng.integers(120, 420))}))
        elif i < ride_events + adhesion_events:
            km = round(float(rng.uniform(100, 300)), 1)
            events.append(Event(t, "low_adhesion_event",
                                {"km": km, "slip_ratio": round(float(rng.uniform(0.15, 0.35)), 2),
                                 "duration_s": int(rng.integers(60, 180))}))
        elif i < ride_events + adhesion_events + panto_events:
            events.append(Event(t, "pantograph_ice",
                                {"varN": int(rng.integers(60, 140)), "temp_c": outside_temp}))
        elif i < ride_events + adhesion_events + panto_events + bearing_events:
            axle = rng.choice(["1L","1R","2L","2R"])
            events.append(Event(t, "bearing_overtemp",
                                {"axle": axle, "peak_c": round(float(rng.uniform(80, 90)), 1), "dwell_s": int(rng.integers(180, 420))}))
        else:
            # delay at Gävle
            events.append(Event(t, "delay_alert",
                                {"station": "Gävle", "delay_min": int(rng.integers(3, 10))}))

    # --- Calculate transmitted bytes under different strategies ---
    coverage = build_coverage_mask(total_s, coverage_profile)
    sent_bits_time = np.zeros(total_s, dtype=np.int64)
    dropped_events = 0
    delivered_events = 0

    # Raw: assume we try to stream continuously, but when coverage=0 nothing goes through.
    if send_mode in ["Raw streams", "Adaptive (prefer Semantic)"]:
        per_sec_bits_raw = int(raw_bits_total // total_s)
        for t in range(total_s):
            if coverage[t] == 1:
                sent_bits_time[t] += per_sec_bits_raw
        raw_bits_delivered = int(sent_bits_time.sum())
    else:
        raw_bits_delivered = 0

    # Semantic: send compact JSON (~120–180 bytes/event). We model 150 bytes avg.
    avg_pkt_bytes = 150
    for ev in events:
        pkt_bits = avg_pkt_bytes * 8
        if coverage[ev.t] == 1:
            # Sent successfully
            delivered_events += 1
            sent_bits_time[ev.t] += pkt_bits if send_mode != "Raw streams" else 0
        else:
            dropped_events += 1

    # Adaptive: if coverage is poor, prioritize semantics and skip raw
    if send_mode == "Adaptive (prefer Semantic)":
        # Already accounted semantic; now reduce raw during low coverage windows
        # Reduce raw by 70% during any second where coverage==1 but previous/next show gaps (edge of holes)
        for t in range(total_s):
            if coverage[t] == 1 and ((t>0 and coverage[t-1]==0) or (t<total_s-1 and coverage[t+1]==0)):
                reduction = int(0.7 * (raw_bits_total // total_s))
                sent_bits_time[t] = max(0, sent_bits_time[t] - reduction)
        raw_bits_delivered = int(sent_bits_time.sum() - delivered_events * avg_pkt_bytes * 8)

    # Final tallies
    total_bits_sent = int(sent_bits_time.sum())
    theoretical_raw_bits = int(raw_bits_total)  # what raw would be with perfect coverage

    return {
        "events": events,
        "coverage": coverage,
        "per_sec_bits": sent_bits_time,
        "raw_bits_delivered": raw_bits_delivered,
        "total_bits_sent": total_bits_sent,
        "theoretical_raw_bits": theoretical_raw_bits,
        "delivered_events": delivered_events,
        "dropped_events": dropped_events,
        "total_seconds": total_s
    }

# ---------------------- Run Simulation ----------------------
res = simulate(duration_min, accel_hz, force_hz, temp_hz, gnss_hz,
               outside_temp, snow_intensity, icing_risk,
               coverage_profile, send_mode)

# ---------------------- UI: Metrics ----------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Trip duration (min)", duration_min)
with c2:
    st.metric("Theoretical RAW size (MB)", f"{(res['theoretical_raw_bits']/(8*1024*1024)):.2f}")
with c3:
    st.metric("Actual sent (MB)", f"{(res['total_bits_sent']/(8*1024*1024)):.2f}")
with c4:
    savings = 1.0 - (res["total_bits_sent"] / max(res["theoretical_raw_bits"], 1))
    st.metric("Bandwidth saved vs RAW", f"{100*savings:.1f}%")

c5, c6 = st.columns(2)
with c5:
    st.metric("Semantic events delivered", res["delivered_events"])
with c6:
    st.metric("Semantic events dropped (no coverage)", res["dropped_events"])

# ---------------------- Plots ----------------------
st.subheader("Per‑second transmitted data & coverage")
import pandas as pd
df = pd.DataFrame({
    "second": np.arange(res["total_seconds"]),
    "bits_sent": res["per_sec_bits"],
    "coverage": res["coverage"]
})
st.line_chart(df[["bits_sent"]])
st.area_chart(df[["coverage"]])

# ---------------------- Event Log (Semantic) ----------------------
st.subheader("Semantic Events (what the command center sees)")
if res["events"]:
    rows = []
    for ev in res["events"]:
        status = "delivered" if res["coverage"][ev.t]==1 and send_mode!="Raw streams" else ("dropped (no coverage)" if send_mode!="Raw streams" else "not sent (raw mode)")
        rows.append({
            "t (s)": ev.t,
            "segment": section_by_time(ev.t, res["total_seconds"]),
            "intent": ev.intent,
            "slots": json.dumps(ev.slots),
            "status": status
        })
    st.dataframe(pd.DataFrame(rows))
else:
    st.info("No semantic events occurred in this run. Adjust winter conditions to increase events.")

# ---------------------- Bytes Accounting (by sensor class) ----------------------
st.subheader("Raw vs Semantic: back‑of‑envelope accounting")
sensor_rows = [
    {"Sensor": "Bogie accel (3‑axis)", "Rate": f"{accel_hz} Hz", "Raw MB (trip)": f"{bits_to_mb(res['total_seconds']*accel_hz*3*BITS_PER_SAMPLE):.2f}",
     "Semantic": "ride_degradation (~0.15 KB/event)", "Notes": "Only when abnormal; dwell+severity slots"},
    {"Sensor": "Pantograph force", "Rate": f"{force_hz} Hz", "Raw MB (trip)": f"{bits_to_mb(res['total_seconds']*force_hz*BITS_PER_SAMPLE):.2f}",
     "Semantic": "pantograph_ice (~0.15 KB/event)", "Notes": "Force variance + ambient temp"},
    {"Sensor": "Bearing temperature x4", "Rate": f"{temp_hz} Hz", "Raw MB (trip)": f"{bits_to_mb(res['total_seconds']*temp_hz*4*BITS_PER_SAMPLE):.2f}",
     "Semantic": "bearing_overtemp (~0.12 KB/event)", "Notes": "PeakC + dwell"},
    {"Sensor": "GNSS", "Rate": f"{gnss_hz} Hz", "Raw MB (trip)": f"{(res['total_seconds']*gnss_hz*400/(8*1024*1024)):.2f}",
     "Semantic": "delay_alert / position_conf_low (~0.12–0.18 KB/event)", "Notes": "Station‑level or confidence events"},
]
st.table(pd.DataFrame(sensor_rows))

# ---------------------- Download: sample semantic trace ----------------------
sample = [{"t": ev.t, "intent": ev.intent, "slots": ev.slots} for ev in res["events"] if res["coverage"][ev.t]==1 and send_mode!="Raw streams"]
st.download_button("Download delivered semantic events (JSON)", data=json.dumps(sample, indent=2), file_name="semantic_events.json", mime="application/json")

st.markdown("---")
st.caption("This demo models realistic sensor rates & winter hazards, showing why semantic communication is essential for 6G rail in rural Nordic conditions.")
