# banking_demo_app.py
# Standalone Streamlit app focused on a real-world BANKING example
# Demonstrates: Process, Proxy, Skeleton, Identical Proxy/Skeleton, IPC vs Local,
# Forwarding Pointers, Shortcutting (clients learn new location), and simple account ops
# Run:  streamlit run banking_demo_app.py

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Set

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏦 Banking Demo — Proxy • Skeleton • Forwarding", layout="wide")

# -------------------------------
# Data model
# -------------------------------
@dataclass
class Process:
    name: str
    label: str
    proxies: List[str] = field(default_factory=list)
    skeletons: List[str] = field(default_factory=list)
    holds_object: bool = False

@dataclass
class ProxyRef:
    id: str
    owner_process: str
    known_skeleton_process: str  # where the client thinks the skeleton is

@dataclass
class SkeletonRef:
    id: str
    host_process: str
    forwards_to: Optional[str] = None  # forwarding pointer target process

@dataclass
class World:
    processes: Dict[str, Process]
    proxies: Dict[str, ProxyRef]
    skeletons: Dict[str, SkeletonRef]
    object_process: str
    # Banking state: a minimal account ledger held by the Account Service
    accounts: Dict[str, int] = field(default_factory=lambda: {"A-100": 500, "B-200": 800})
    log: List[str] = field(default_factory=list)

    def log_event(self, msg: str):
        self.log.append(msg)

# -------------------------------
# World setup & helpers
# -------------------------------

def init_banking_world() -> World:
    # P1=ATM, P2=MobileApp, P3=API Gateway (skeleton host), P4=Account Service (object)
    processes = {
        "P1": Process(name="P1", label="ATM (Client)"),
        "P2": Process(name="P2", label="MobileApp (Client)"),
        "P3": Process(name="P3", label="API Gateway (Skeleton Host)"),
        "P4": Process(name="P4", label="Account Service (Object)", holds_object=True),
    }

    # Gateway skeleton pointing to Account Service initially
    skel = SkeletonRef(id="S@P3", host_process="P3", forwards_to="P4")
    processes["P3"].skeletons.append(skel.id)

    # Identical proxies at ATM and Mobile App
    proxies = {
        "px_atm": ProxyRef(id="px_atm", owner_process="P1", known_skeleton_process="P3"),
        "px_mob": ProxyRef(id="px_mob", owner_process="P2", known_skeleton_process="P3"),
    }
    processes["P1"].proxies.append("px_atm")
    processes["P2"].proxies.append("px_mob")

    w = World(processes=processes, proxies=proxies, skeletons={skel.id: skel}, object_process="P4")
    w.log_event("Initialized: ATM & MobileApp have identical proxies → API Gateway skeleton → Account Service object.")
    return w


def move_object(w: World, dst_process: str, create_new_skeleton: bool, keep_forwarding: bool):
    if dst_process == w.object_process:
        w.log_event(f"Object already at {dst_process} — no move.")
        return
    src = w.object_process
    w.processes[src].holds_object = False
    w.processes[dst_process].holds_object = True
    w.object_process = dst_process

    # Optional: create new skeleton at destination
    if create_new_skeleton:
        sid = f"S@{dst_process}"
        if sid not in w.skeletons:
            w.skeletons[sid] = SkeletonRef(id=sid, host_process=dst_process)
            w.processes[dst_process].skeletons.append(sid)
        w.log_event(f"Object moved {src} → {dst_process}; created new skeleton at {dst_process}.")

    # Manage forwarding pointers from older skeletons
    if keep_forwarding:
        for s in w.skeletons.values():
            if s.host_process != dst_process:
                s.forwards_to = dst_process
        w.log_event(f"Forwarding pointers: old skeletons → {dst_process}.")
    else:
        for s in w.skeletons.values():
            if s.host_process != dst_process:
                s.forwards_to = None
        w.log_event("No forwarding kept; clients must learn new location via discovery.")


def invoke(w: World, proxy_id: str, do_shortcut: bool = True) -> Tuple[bool, str, List[Tuple[str, str]]]:
    if proxy_id not in w.proxies:
        return False, f"Proxy {proxy_id} not found", []

    pr = w.proxies[proxy_id]
    route_text: List[str] = []
    edges: List[Tuple[str, str]] = []

    # Client → known skeleton (IPC unless same process)
    edges.append((pr.owner_process, pr.known_skeleton_process))
    hop_type = "LOCAL" if pr.owner_process == pr.known_skeleton_process else "IPC"
    route_text.append(f"{pr.owner_process} → {pr.known_skeleton_process} ({hop_type})")

    # Find skeleton
    target = None
    for s in w.skeletons.values():
        if s.host_process == pr.known_skeleton_process:
            target = s
            break
    if not target:
        w.log_event(f"{proxy_id}: no skeleton at {pr.known_skeleton_process} (stale reference)")
        return False, " → ".join(route_text + ["✖ no skeleton"]), edges

    # Follow forwarding pointers (short chain)
    visited: Set[str] = set()
    current = target
    k = 0
    while current.forwards_to and current.forwards_to not in visited and k < 8:
        visited.add(current.host_process)
        edges.append((current.host_process, current.forwards_to))
        route_text.append(f"{current.host_process} → {current.forwards_to} (forward)")
        # step to skeleton at forwards_to if exists
        nxt = None
        for s in w.skeletons.values():
            if s.host_process == current.forwards_to:
                nxt = s
                break
        if nxt is None:
            break
        current = nxt
        k += 1

    # Final service hop to where object lives (conceptual)
    if current.host_process != w.object_process:
        edges.append((current.host_process, w.object_process))
        route_text.append(f"{current.host_process} → {w.object_process} (serve)")

    # Shortcut after first success: update proxy to known skeleton at object (if any), else current
    if do_shortcut:
        new_known = w.object_process if f"S@{w.object_process}" in w.skeletons else current.host_process
        pr.known_skeleton_process = new_known
        w.log_event(f"{proxy_id} learned skeleton location: {new_known} (shortcut).")

    return True, " → ".join(route_text + [f"→ {w.object_process} (object)"]), edges


def gc_skeletons(w: World):
    # Remove skeletons not at object host, not forwarding, and not targeted by any proxy
    to_rm = []
    for sid, s in w.skeletons.items():
        if s.host_process == w.object_process:
            continue
        if s.forwards_to is not None:
            continue
        if any(p.known_skeleton_process == s.host_process for p in w.proxies.values()):
            continue
        to_rm.append(sid)
    for sid in to_rm:
        host = w.skeletons[sid].host_process
        if sid in w.processes[host].skeletons:
            w.processes[host].skeletons.remove(sid)
        del w.skeletons[sid]
        w.log_event(f"GC: removed skeleton at {host}.")

# -------------------------------
# Banking operations (handled by Account Service)
# -------------------------------

def op_balance(w: World, account: str) -> int:
    return w.accounts.get(account, 0)

def op_deposit(w: World, account: str, amount: int):
    w.accounts[account] = w.accounts.get(account, 0) + amount
    w.log_event(f"DEPOSIT: {account} += {amount} → {w.accounts[account]}")

def op_withdraw(w: World, account: str, amount: int) -> bool:
    bal = w.accounts.get(account, 0)
    if amount <= bal:
        w.accounts[account] = bal - amount
        w.log_event(f"WITHDRAW: {account} -= {amount} → {w.accounts[account]}")
        return True
    w.log_event(f"WITHDRAW FAIL: {account} has {bal}, need {amount}")
    return False

# -------------------------------
# Visualization
# -------------------------------

def draw_world(w: World, highlight_edges: Optional[List[Tuple[str, str]]] = None, highlight_nodes: Optional[Set[str]] = None, title: str = ""):
    highlight_edges = highlight_edges or []
    highlight_nodes = highlight_nodes or set()

    G = nx.DiGraph()
    for pname, proc in w.processes.items():
        label = f"{pname}\n{proc.label}"
        if proc.holds_object:
            label += "\n[OBJECT]"
        if proc.proxies:
            label += "\nProxies: " + ", ".join(proc.proxies)
        if proc.skeletons:
            label += f"\nSkeletons: {len(proc.skeletons)}"
        G.add_node(pname, label=label)

    # Proxy edges (client → skeleton host)
    for pid, pr in w.proxies.items():
        G.add_edge(pr.owner_process, pr.known_skeleton_process, kind="proxy", label=pid)

    # Forwarding pointer edges (skeleton host → next)
    for s in w.skeletons.values():
        if s.forwards_to:
            G.add_edge(s.host_process, s.forwards_to, kind="forward", label="forward")

    # Service edges (skeleton host → object)
    for s in w.skeletons.values():
        G.add_edge(s.host_process, w.object_process, kind="serve", label="serve")

    pos = nx.spring_layout(G, seed=42, k=1.1)

    fig = plt.figure(figsize=(9.5, 6.8))

    # Nodes
    node_colors = ["#ffd166" if n in highlight_nodes else "#a8dadc" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=2100, node_color=node_colors, edgecolors="#1d3557", linewidths=1.4)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes})

    # Helper to draw each edge type
    def draw_kind(kind: str, style: str, width: float, alpha: float):
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == kind]
        widths = [3.0 if (u, v) in (highlight_edges or []) else width for (u, v) in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True, style=style, width=widths, alpha=alpha)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G.edges[u, v]['label'] for (u, v) in edges})

    draw_kind("proxy", style="solid", width=2.0, alpha=0.95)
    draw_kind("forward", style="dashed", width=2.0, alpha=0.95)
    draw_kind("serve", style="dotted", width=1.6, alpha=0.75)

    plt.title(title)
    plt.axis('off')
    st.pyplot(fig, width='stretch')

# -------------------------------
# UI — Two tabs: Guided Steps • Interactive Playground
# -------------------------------
GUIDE_STEPS = [
    (1, "Actors"),
    (2, "Login (Identical Proxies)"),
    (3, "Balance (Skeleton Dispatch)"),
    (4, "Withdraw (IPC Path)"),
    (5, "Scale-out (Identical Skeletons)"),
    (6, "Migration (Forwarding Pointer)"),
    (7, "Shortcutting & Cleanup"),
]

if 'world_bank' not in st.session_state:
    st.session_state
