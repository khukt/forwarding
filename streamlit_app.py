# -------------------------------
# Directional Graph Helper (bigger arrows + legend)
# -------------------------------

def draw_world_big(world: World, highlight_nodes: Optional[Set[str]] = None, highlight_edges: Optional[List[Tuple[str, str]]] = None, title: Optional[str] = None):
    highlight_nodes = highlight_nodes or set()
    highlight_edges = highlight_edges or []

    G = nx.DiGraph()

    # Add process nodes
    for pname, proc in world.processes.items():
        label = pname
        if proc.holds_object:
            label += "\n[OBJECT]"
        if proc.proxies:
            label += "\nProxies: " + ", ".join(proc.proxies)
        if proc.skeletons:
            label += f"\nSkeletons: {len(proc.skeletons)}"
        G.add_node(pname, label=label)

    # Edges from proxies to their known skeleton process
    for pid, pr in world.proxies.items():
        G.add_edge(pr.owner_process, pr.known_skeleton_process, kind="proxy", label=f"proxy {pid}")

    # Forwarding pointers between skeletons
    for s in world.skeletons.values():
        if s.forwards_to:
            G.add_edge(s.host_process, s.forwards_to, kind="forward", label='forward')

    # Edges from skeleton host to object process (conceptual service edge)
    for s in world.skeletons.values():
        G.add_edge(s.host_process, world.object_process, kind="serve", label='serve')

    pos = nx.spring_layout(G, seed=7, k=1.2)

    fig = plt.figure(figsize=(9.5, 6.8))

    # Nodes
    node_colors = ["#ffd166" if n in highlight_nodes else "#a8dadc" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, edgecolors="#1d3557", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes})

    def draw_edges(kind: str, style: str, width: float, alpha: float):
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == kind]
        widths = [3.0 if (u, v) in highlight_edges else width for (u, v) in edges]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            arrows=True,
            arrowsize=28,
            connectionstyle="arc3,rad=0.08",
            style=style,
            width=widths,
            alpha=alpha
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G.edges[u, v]['label'] for (u, v) in edges})

    draw_edges("proxy", style="solid", width=2.6, alpha=0.95)
    draw_edges("forward", style="dashed", width=2.6, alpha=0.95)
    draw_edges("serve", style="dotted", width=2.0, alpha=0.8)

    plt.title(title or "")
    plt.axis('off')
    st.pyplot(fig, width='stretch')

    # Legend / Direction Guide
    st.markdown(
        "> **Arrow Key**  \
        > — **solid →** proxy call (client → skeleton)  \
        > — **dashed →** forwarding pointer (old skeleton → new location)  \
        > — **dotted →** skeleton dispatch to object (skeleton → object)"
    )

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
        st.subheader("Explanation")
        st.markdown(capB)
        st.caption("Edges highlight the current flow (proxy IPC, skeleton forwarding, or service call).")

        st.markdown("---")
        st.subheader("Process Details")
        st.markdown(
            """
            **Proxy process (client side)**
            - Looks like the object locally.
            - **Marshals** parameters and performs a **remote call** to the skeleton.
            - Updates its cached location after the first successful call (**shortcutting**).

            **Skeleton process (server side)**
            - Listens for network requests from proxies.
            - **Unmarshals** parameters and invokes the real object in the object process.
            - If the object moved, uses a **forwarding pointer** to send the request to the new host.

            **Local invocation process**
            - When proxy and skeleton live in the **same process**, the call is local (no network hop).
            - Try a local example in the Tutorial tab: *Local Invocation* step — note the lack of IPC edge.
            """
        )

    if st.session_state.bank_is_playing:
        time.sleep(delayB)
        st.session_state.bank_step_idx = 1 + (st.session_state.bank_step_idx % len(BANKING_STEPS))
        st.rerun()
