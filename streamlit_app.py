# -------------------------------
# Tab 3 — Banking Example (Real App Mapping)
# -------------------------------
with _tabs[2]:
    st.subheader("Banking — Real App Walkthrough")
    st.caption("P1=ATM, P2=MobileApp, P3=API Gateway (skeleton host), P4=Account Service (object)")

    # Use separate state variables for banking autoplay
    if 'bank_step_idx' not in st.session_state:
        st.session_state.bank_step_idx = 1
    if 'bank_is_playing' not in st.session_state:
        st.session_state.bank_is_playing = False

    leftB, rightB = st.columns([3, 2])
    with rightB:
        st.markdown("**Steps**: " + " → ".join([f"{i}. {name}" for i, name in BANKING_STEPS]))
        # Slider for step selection, decoupled from autoplay index
        _b_val = st.slider("Banking Step", 1, len(BANKING_STEPS), value=st.session_state.bank_step_idx, key="bank_step_ui")
        if not st.session_state.bank_is_playing:
            st.session_state.bank_step_idx = _b_val

        speedB = st.select_slider("Animation speed", options=["slow", "normal", "fast"], value="normal", key="anim_speed_bank")
        delayB = {"slow": 1.2, "normal": 0.8, "fast": 0.4}[speedB]

        colbp = st.columns(3)
        with colbp[0]:
            if st.button("◀ Prev ", key="bank_prev"):
                st.session_state.bank_step_idx = max(1, st.session_state.bank_step_idx - 1)
        with colbp[1]:
            if st.button("▶ Play " if not st.session_state.bank_is_playing else "⏸ Pause", key="bank_play_toggle"):
                st.session_state.bank_is_playing = not st.session_state.bank_is_playing
        with colbp[2]:
            if st.button("Next ▶ ", key="bank_next"):
                st.session_state.bank_step_idx = min(len(BANKING_STEPS), st.session_state.bank_step_idx + 1)

    # Build world & highlights for this banking step
    wbank, capB, edgesB = banking_step(st.session_state.bank_step_idx)

    with leftB:
        draw_world(wbank, highlight_edges=edgesB, title=f"Banking Step {st.session_state.bank_step_idx}: {BANKING_STEPS[st.session_state.bank_step_idx-1][1]}")

    with rightB:
        st.subheader("Explanation")
        st.markdown(capB)
        st.caption("Edges highlight the current flow (proxy IPC, skeleton forwarding, or service call).")

    if st.session_state.bank_is_playing:
        time.sleep(delayB)
        st.session_state.bank_step_idx = 1 + (st.session_state.bank_step_idx % len(BANKING_STEPS))
        st.rerun()
