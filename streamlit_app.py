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
