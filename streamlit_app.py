clicked = plotly_events(
    fig, click_event=True, select_event=False,
    override_width=780, override_height=780,
    key="ring_step2_click", config=PLOTLY_CONFIG
)

if clicked:
    meta = clicked[0].get("customdata")  # {'id': int, 'kind': 'disabled'|'active'}
    if isinstance(meta, dict) and "id" in meta and "kind" in meta:
        nid = int(meta["id"])
        kind = meta["kind"]
        nodes = set(st.session_state.active_nodes)

        if kind == "disabled":
            # Join + select
            nodes.add(nid)
            st.session_state.active_nodes = sorted(nodes)
            st.session_state.selected = nid
            st.rerun()

        elif kind == "active":
            if st.session_state.allow_remove_click:
                # Remove (unless it's the last one)
                if len(nodes) > 1:
                    nodes.remove(nid)
                    st.session_state.active_nodes = sorted(nodes)
                    # keep selection valid
                    st.session_state.selected = (
                        st.session_state.active_nodes[0]
                        if st.session_state.selected not in nodes else st.session_state.selected
                    )
                    st.rerun()
                else:
                    st.warning("Cannot remove the last active node.")
            else:
                # Just select (turns red)
                st.session_state.selected = nid
                st.rerun()
