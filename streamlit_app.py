# streamlit_app.py
# Educational simulator to understand Forwarding Pointers, Proxy, Skeleton, and Identical Proxy
# Run with:  streamlit run streamlit_app.py

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

st.set_page_config(page_title="Forwarding Pointers • Proxy • Skeleton — Visual Simulator", layout="wide")

# -------------------------------
# Data Model
# -------------------------------
@dataclass
class Process:
    name: str
    proxies: List[str] = field(default_factory=list)   # proxy ids living in this process
    skeletons: List[str] = field(default_factory=list) # skeleton ids living in this process
    holds_object: bool = False                        # whether the real object is here

@dataclass
class ProxyRef:
    id: str
    owner_process: str
    # The proxy believes the object's skeleton is at this process
    known_skeleton_process: str

@dataclass
class SkeletonRef:
    id: str
    host_process: str
    forwards_to: Optional[str] = None  # process name of next skeleton (forwarding pointer)

@dataclass
class World:
    processes: Dict[str, Process]
    proxies: Dict[str, ProxyRef]
    skeletons: Dict[str, SkeletonRef]
    object_process: str
    log: List[str] = field(default_factory=list)

    def log_event(self, msg: str):
        self.log.append(msg)

# -------------------------------
# Helpers
# -------------------------------

def init_default_world(identical_proxy_count: int = 2) -> World:
    # Create four processes P1..P4 by default
    processes = {f"P{i}": Process(name=f"P{i}") for i in range(1, 5)}

    # Place real object at P4
    processes["P4"].holds_object = True

    # Place a skeleton at P3 that serves the object in P4
    skel_main = SkeletonRef(id="S@P3", host_process="P3", forwards_to=None)
    processes["P3"].skeletons.append(skel_main.id)

    proxies: Dict[str, ProxyRef] = {}
    # Create N identical proxies for the same object (one per client process)
    for i in range(identical_proxy_count):
        proc_name = f"P{i+1}"
        pid = f"p{i+1}"
        proxies[pid] = ProxyRef(id=pid, owner_process=proc_name, known_skeleton_process="P3")
        processes[proc_name].proxies.append(pid)

    world = World(
        processes=processes,
        proxies=proxies,
        skeletons={skel_main.id: skel_main},
        object_process="P4",
    )
    world.log_event("Initialized world: object at P4, skeleton at P3, identical proxies at P1 and P2.")
    return world


def move_object(world: World, dst_process: str, create_new_skeleton: bool, keep_forwarding: bool):
    if dst_process == world.object_process:
        world.log_event(f"Object already at {dst_process} — no move performed.")
        return

    src = world.object_process
    world.processes[src].holds_object = False
    world.object_process = dst_process
    world.processes[dst_process].holds_object = True

    # Option: create a new skeleton at the destination (common in practice)
    if create_new_skeleton:
        new_id = f"S@{dst_process}"
        if new_id not in world.skeletons:
            world.skeletons[new_id] = SkeletonRef(id=new_id, host_process=dst_process)
            world.processes[dst_process].skeletons.append(new_id)
        world.log_event(f"Object moved {src} → {dst_process}. New skeleton created at {dst_process}.")

        # Keep forwarding from old skeleton(s) to the new one (forwarding pointer chain)
        if keep_forwarding:
            for s in world.skeletons.values():
                if s.host_process != dst_process and (s.forwards_to is None):
                    s.forwards_to = dst_process
            world.log_event(f"Established forwarding pointers from old skeletons → {dst_process}.")
        else:
            # If not keeping forwarding, proxies will break until they learn the new location
            for s in world.skeletons.values():
                if s.host_process != dst_process:
                    s.forwards_to = None
            world.log_event("No forwarding pointers kept. Old skeletons do not forward.")
    else:
        # No new skeleton created: assume the old skeleton still serves the object remotely
        world.log_event(f"Object moved {src} → {dst_process}. No new skeleton; old skeletons remain.")


def invoke(world: World, proxy_id: str, shortcut_after_first: bool = True) -> Tuple[bool, str]:
    """Simulate an invocation from a proxy.
    Returns (success, route_str) and updates the proxy's known location if shortcutting.
    """
    if proxy_id not in world.proxies:
        return False, f"Proxy {proxy_id} not found"

    pr = world.proxies[proxy_id]
    route = [pr.owner_process, pr.known_skeleton_process]

    # Find an existing skeleton at the known process
    target_skel = None
    for s in world.skeletons.values():
        if s.host_process == pr.known_skeleton_process:
            target_skel = s
            break

    if target_skel is None:
        # No skeleton at the known process — invocation fails unless the proxy already knows the new one
        msg = f"Proxy {proxy_id} tried {pr.known_skeleton_process} — no skeleton there (stale reference)."
        world.log_event(msg)
        return False, " → ".join(route + ["✖ no skeleton"]) 

    # Follow forwarding pointers (at most a few hops for demo)
    visited = set()
    current = target_skel
    hops = 0
    while current.forwards_to and current.forwards_to not in visited and hops < 8:
        visited.add(current.host_process)
        route.append(f"(forward→{current.forwards_to})")
        # Move to skeleton at forwards_to, if any
        next_skel = None
        for s in world.skeletons.values():
            if s.host_process == current.forwards_to:
                next_skel = s
                break
        if next_skel is None:
            # Implicit skeleton at destination (for demo assume local dispatch)
            route.append(current.forwards_to)
            break
        current = next_skel
        hops += 1

    # Now we are at a skeleton presumed capable of reaching the object
    route.append(world.object_process)

    # Success; optionally apply shortcutting (proxy learns the new skeleton location)
    if shortcut_after_first:
        pr.known_skeleton_process = current.host_process
        world.log_event(f"{proxy_id} learned new skeleton location: {current.host_process} (shortcut applied).")

    return True, " → ".join(route)


def garbage_collect_old_skeletons(world: World):
    # Remove skeletons that nobody points to (no process hosts the object there, and no forwarding)
    to_remove = []
    for sid, s in world.skeletons.items():
        still_needed = s.host_process == world.object_process or s.forwards_to is not None
        if not still_needed:
            # Also keep a skeleton if any proxy still targets it (to simulate lingering references)
            if any(p.known_skeleton_process == s.host_process for p in world.proxies.values()):
                continue
            to_remove.append(sid)
    for sid in to_remove:
        host = world.skeletons[sid].host_process
        if sid in world.processes[host].skeletons:
            world.processes[host].skeletons.remove(sid)
        del world.skeletons[sid]
        world.log_event(f"Garbage collected skeleton at {host}.")

# -------------------------------
# Visualization
# -------------------------------

def draw_world(world: World):
    G = nx.DiGraph()

    # Add process nodes
    for pname, proc in world.processes.items():
        label = pname
        if proc.holds_object:
            label += "\n[OBJECT]"
        if proc.proxies:
            label += "\nProxies: " + ", ".join(proc.proxies)
        if proc.skeletons:
            label += "\nSkeletons: " + ", ".join([sid.split('@')[0] for sid in proc.skeletons])
        G.add_node(pname, label=label, type='process')

    # Edges from proxies to their known skeleton process
    for pid, pr in world.proxies.items():
        G.add_edge(pr.owner_process, pr.known_skeleton_process, label=pid)

    # Forwarding pointers between skeletons
    for s in world.skeletons.values():
        if s.forwards_to:
            G.add_edge(s.host_process, s.forwards_to, label='forward')

    # Edges from skeleton host to object process (conceptual service edge)
    for s in world.skeletons.values():
        # Show an edge representing that the skeleton can reach the object
        G.add_edge(s.host_process, world.object_process, label='serve')

    pos = nx.spring_layout(G, seed=7, k=1.2)

    fig = plt.figure(figsize=(8.5, 6))
    nx.draw_networkx_nodes(G, pos, node_size=2000)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes})
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):d.get('label','') for u,v,d in G.edges(data=True)})
    plt.axis('off')
    st.pyplot(fig, use_container_width=True)

# -------------------------------
# UI
# -------------------------------
if 'world' not in st.session_state:
    st.session_state.world = init_default_world(identical_proxy_count=2)

world: World = st.session_state.world

with st.sidebar:
    st.title("Simulator Controls")

    st.markdown("**Concepts**\n- **Proxy**: client-side representative of a remote object.\n- **Identical proxies**: several proxies for the same object (e.g., p1 at P1 and p2 at P2).\n- **Skeleton**: server-side dispatcher for method calls.\n- **Forwarding pointer**: old skeleton forwards to the object's new location after migration.")

    st.divider()
    st.subheader("Initialize")
    n = st.slider("Number of identical proxies (clients)", 1, 4, 2)
    if st.button("Reset world"):
        st.session_state.world = init_default_world(identical_proxy_count=n)
        world = st.session_state.world

    st.divider()
    st.subheader("Move Object (Migration)")
    dst = st.selectbox("Move object to:", options=list(world.processes.keys()), index=3)
    create_new_skel = st.checkbox("Create NEW skeleton at destination", value=True)
    keep_fwd = st.checkbox("Keep forwarding pointers from old skeletons", value=True)
    if st.button("Move object"):
        move_object(world, dst, create_new_skeleton=create_new_skel, keep_forwarding=keep_fwd)

    st.divider()
    st.subheader("Invoke")
    colA, colB = st.columns(2)
    with colA:
        choose_proxy = st.selectbox("Choose proxy (client)", options=list(world.proxies.keys()))
    with colB:
        shortcut = st.checkbox("Shortcut after first invocation (proxy learns new location)", value=True)
    if st.button("Invoke via selected proxy"):
        ok, route = invoke(world, choose_proxy, shortcut_after_first=shortcut)
        world.log_event(("SUCCESS" if ok else "FAIL") + f" — route: {route}")

    if st.button("Garbage-collect unused skeletons"):
        garbage_collect_old_skeletons(world)

st.title("Forwarding Pointers • Proxy • Skeleton — Visual Simulator")

left, right = st.columns([3,2])
with left:
    draw_world(world)
with right:
    st.subheader("Event Log")
    if world.log:
        for line in world.log[-30:][::-1]:
            st.markdown(f"• {line}")
    else:
        st.info("No events yet. Use the controls to move the object or invoke via a proxy.")

st.divider()
with st.expander("How to Use / What to Observe", expanded=False):
    st.markdown(
        """
        **Try these:**
        1) Click **Invoke via selected proxy**. Notice the route: `P1 → P3 → (forward→P4) → P4`.
        2) Toggle **Shortcut after first invocation** to see the proxy learn the new skeleton location.
        3) **Move object** to another process (e.g., P2), **keep forwarding pointers**, then invoke again: the route follows the forward pointer once, then future calls go direct (shortcut).
        4) Click **Garbage-collect** once all proxies have updated, to remove obsolete skeletons.
        
        **Concept mapping:**
        - **Proxy**: the edge labeled with proxy id from client process to the skeleton host it believes in.
        - **Identical proxies**: set the number of proxies > 1; they all behave the same but live in different client processes.
        - **Skeleton**: box that serves the object; when the object moves and a new skeleton is created at destination, old skeletons can **forward**.
        - **Forwarding pointer**: edge labeled `forward` between skeleton hosts.
        - **Shortcutting**: after a first call via forwarding, the proxy updates its known skeleton host so future calls go direct.
        """
    )
