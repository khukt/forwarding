
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ------------------------
# Educational Streamlit App
# ------------------------

st.set_page_config(page_title="Distributed Systems: Proxies, Skeletons & Forwarding Pointers", layout="wide")

# ---------- Models ----------

@dataclass
class ProcessNode:
    name: str
    kind: str  # 'client' or 'server'
    notes: str = ""

@dataclass
class ObjectImpl:
    name: str = "FileService"
    process: str = "P3"  # where the object currently lives

@dataclass
class ProxyState:
    process: str = "P1"     # where the proxy lives (client process)
    target_skeleton: str = "P3" # which server's skeleton it thinks to call
    shortcut_installed: bool = True  # direct to current skeleton if True

@dataclass
class ForwardingState:
    # map from old_skeleton_process -> new_skeleton_process (forward pointers)
    pointers: Dict[str, str] = field(default_factory=dict)

@dataclass
class WorldState:
    processes: Dict[str, ProcessNode] = field(default_factory=dict)
    object_impl: ObjectImpl = field(default_factory=ObjectImpl)
    proxy: ProxyState = field(default_factory=ProxyState)
    forwards: ForwardingState = field(default_factory=ForwardingState)
    log: List[str] = field(default_factory=list)

def init_world() -> WorldState:
    processes = {
        "P1": ProcessNode("P1", "client", "Student laptop / client app"),
        "P2": ProcessNode("P2", "server", "Server A (skeleton)"),
        "P3": ProcessNode("P3", "server", "Server B (skeleton)"),
        "P4": ProcessNode("P4", "server", "Server C (skeleton + object)"),
    }
    obj = ObjectImpl(name="CloudFile", process="P4")
    proxy = ProxyState(process="P1", target_skeleton="P4", shortcut_installed=True)
    forwards = ForwardingState(pointers={})
    return WorldState(processes, obj, proxy, forwards, log=[])

# ---------- Helpers ----------

def add_log(state: WorldState, msg: str):
    state.log.append(msg)

def migrate_object(state: WorldState, to_process: str):
    frm = state.object_impl.process
    if frm == to_process:
        add_log(state, f"Object already at {to_process}. No migration.")
        return
    # leave a forwarding pointer from the old skeleton to the new one
    state.forwards.pointers[frm] = to_process
    state.object_impl.process = to_process
    # shortcut becomes invalid (temporarily)
    state.proxy.shortcut_installed = False
    add_log(state, f"Object migrated from {frm} -> {to_process}. Forwarding pointer left behind.")

def resolve_path(state: WorldState) -> Tuple[List[str], bool]:
    """
    Returns the path taken for an invocation from proxy.process to the object's skeleton.
    Also returns whether a forwarding hop occurred.
    """
    path = []
    forwarding_hop = False
    # proxy calls what it believes is the skeleton
    current = state.proxy.target_skeleton
    path.append(state.proxy.process)
    path.append(current)

    # follow forwarding pointers until we reach the actual object location
    while current in state.forwards.pointers:
        forwarding_hop = True
        nxt = state.forwards.pointers[current]
        path.append(nxt)
        current = nxt

    # final hop to the object (same process as current skeleton)
    obj_proc = state.object_impl.process
    if current != obj_proc:
        # If inconsistent, assume skeleton lives where object is
        current = obj_proc
        path.append(current)

    # path now ends at object's process
    return path, forwarding_hop

def invoke(state: WorldState):
    path, fwd = resolve_path(state)
    add_log(state, f"Invocation request: {' → '.join(path)} (forwarded={fwd})")
    # After a successful invocation, install a shortcut directly to current skeleton
    current_obj_proc = state.object_impl.process
    state.proxy.target_skeleton = current_obj_proc
    state.proxy.shortcut_installed = True
    # cleanup: remove obsolete chains that are no longer referenced (simple heuristic)
    # Here, because the proxy now points to the latest process, we can drop all pointers that lead to it.
    obsolete = [k for k, v in state.forwards.pointers.items() if v == current_obj_proc]
    for k in obsolete:
        state.forwards.pointers.pop(k, None)
    if fwd and obsolete:
        add_log(state, f"Shortcut installed: proxy now calls {current_obj_proc} directly. Removed obsolete pointer(s): {obsolete}")
    elif fwd:
        add_log(state, f"Shortcut installed: proxy now calls {current_obj_proc} directly.")

# ---------- Visualization ----------

def draw_world(state: WorldState):
    G = nx.DiGraph()
    # add nodes
    for pid, p in state.processes.items():
        label = f"{pid}\n({p.kind})"
        G.add_node(pid, label=label)

    # base layout: fixed positions for clarity
    pos = {
        "P1": (-1.2, 0.0),
        "P2": (0.0, 0.8),
        "P3": (0.0, 0.0),
        "P4": (0.0, -0.8),
    }

    # add edges for possible IPC
    for pid, p in state.processes.items():
        if p.kind == "server":
            G.add_edge("P1", pid)

    # Now highlight the actual invocation path
    path, fwd = resolve_path(state)

    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw_networkx_nodes(G, pos, node_size=1800, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()}, ax=ax)

    # draw all light edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=True, ax=ax)

    # draw forwarding pointers as dashed arrows
    for k, v in state.forwards.pointers.items():
        nx.draw_networkx_edges(
            G, pos, edgelist=[(k, v)], style="dashed", width=2.0, arrows=True, ax=ax
        )

    # draw current invocation path as thicker arrows
    if len(path) >= 2:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3.0, arrows=True, ax=ax)

    # annotate object & proxy
    # object
    obj_p = state.object_impl.process
    ax.text(pos[obj_p][0]+0.25, pos[obj_p][1]-0.15, f"Object: {state.object_impl.name}", fontsize=10)
    # proxy
    ax.text(pos["P1"][0]-0.1, pos["P1"][1]+0.15, "Proxy (client)", fontsize=10)

    ax.set_axis_off()
    st.pyplot(fig)

# ---------- UI ----------

if "world" not in st.session_state:
    st.session_state.world = init_world()

state: WorldState = st.session_state.world

st.title("Distributed Systems Tutor: Proxies, Skeletons & Forwarding Pointers")

with st.sidebar:
    st.header("Controls")
    st.markdown("**Object Location**: " + state.object_impl.process)
    st.markdown("**Proxy Target Skeleton**: " + state.proxy.target_skeleton)
    st.markdown("**Shortcut Installed**: " + ("Yes" if state.proxy.shortcut_installed else "No"))
    st.divider()
    colA, colB = st.columns(2)
    with colA:
        to_proc = st.selectbox("Migrate object to:", ["P2", "P3", "P4"], index=["P2","P3","P4"].index(state.object_impl.process))
    with colB:
        if st.button("🔁 Migrate Object"):
            migrate_object(state, to_proc)
    if st.button("⚡ Invoke (RPC call)"):
        invoke(state)
    if st.button("♻️ Reset world"):
        st.session_state.world = init_world()
        state = st.session_state.world

    st.divider()
    st.subheader("Forward Pointers (current)")
    if not state.forwards.pointers:
        st.caption("No forwarding pointers at the moment.")
    else:
        for k, v in state.forwards.pointers.items():
            st.code(f"{k} ⟶ {v}")

    st.divider()
    st.subheader("Event Log")
    if state.log:
        for line in reversed(state.log[-14:]):
            st.write("• " + line)
    else:
        st.caption("Interact with the buttons to see events.")

# Main content
tab1, tab2, tab3 = st.tabs(["Simulator", "Concepts", "Exercises"])

with tab1:
    st.subheader("Invocation Path & Topology")
    draw_world(state)
    st.markdown("""
**How to use**
1. Click **Invoke** to perform an RPC call from the client proxy to the object's skeleton.
2. Click **Migrate Object** to move the object between servers (P2/P3/P4). A forwarding pointer is left at the old location.
3. Invoke again: the call will **follow** the forwarding pointer chain the first time, **then install a shortcut**.
    """)

with tab2:
    st.subheader("Key Concepts (Distributed Objects / RPC)")
    with st.expander("Process"):
        st.write("An executing program with its own address space. In this app: P1 is a client; P2–P4 are servers.")
    with st.expander("Proxy (client-side stub)"):
        st.write("A local object that exposes the same interface as a remote object, turning method calls into RPC messages.")
    with st.expander("Skeleton (server-side stub)"):
        st.write("Receives RPC requests, unmarshals data, and calls the actual object implementation.")
    with st.expander("Object (service implementation)"):
        st.write("The actual service code and state. Here it migrates across P2/P3/P4.")
    with st.expander("Interprocess Communication (IPC)"):
        st.write("Network communication between processes. In RPC this is often TCP+protobuf (gRPC) or HTTP+JSON.")
    with st.expander("Invocation request"):
        st.write("The client's method call (e.g., read(), write()) that travels proxy → skeleton → object.")
    with st.expander("Forwarding pointer"):
        st.write("A reference left at the previous location when an object migrates, used to forward requests to the new location. Prevents broken references.")
    with st.expander("Identical proxy"):
        st.write("Different clients each get their own proxy instances that behave the same (identical interface/semantics).")

with tab3:
    st.subheader("Quick Exercises")
    st.markdown("**Q1.** After migrating the object from P4 to P2, what happens to the first invocation from the proxy?")
    a1 = st.radio("Choose one", [
        "It fails because the proxy still points to P4.",
        "It succeeds by being forwarded from P4 to P2, then installs a shortcut.",
        "It always goes directly to P2 even before learning the new location."], index=1)
    if a1:
        st.success("Expected: It succeeds via forwarding, then the proxy installs a shortcut.")

    st.markdown("---")
    st.markdown("**Q2.** Why do systems remove forwarding chains (pointer compression)?")
    a2 = st.checkbox("To reduce latency and avoid long chains of indirection.")
    if a2:
        st.info("Correct. Shortcutting keeps calls fast and resource usage low.")

    st.markdown("---")
    st.markdown("**Q3.** Map each term to the app:")
    cols = st.columns(3)
    with cols[0]: st.write("- Proxy"); st.write("- Skeleton"); st.write("- Object")
    with cols[1]: st.write("= Client-side stub"); st.write("= Server-side stub"); st.write("= Service implementation")
    with cols[2]: st.write("✅")

st.caption("Tip: Run locally with `streamlit run app.py`. This app uses only matplotlib and networkx for visuals.")
