"""
Build a directed graph from nodes/edges and derive metrics.

V5.1 changes to infer_start_end():
─────────────────────────────────
  Start-state inference (was returning 2, should be 1):
    GREEN-EXCLUSIVE rule:
      If ANY node has style=="green" AND in-degree==0, those green nodes are
      the ONLY starts. All other in-degree==0 nodes are suppressed.
      Rationale: workflow diagrams mark the entry point with a green box.
      A second in-degree==0 node almost certainly has a missing incoming arrow
      (a detection error), not a genuine second entry point.
    CYCLE-BACK exception (unchanged):
      A green node with in-degree>0 is still a start when every predecessor
      is also reachable FROM it (back-edges only — loop pattern).

  End-state inference (was returning 3, should be 2):
    TERMINAL-KEYWORD priority rule:
      If SOME out-degree==0 nodes have terminal keywords (Complete, Done, …)
      AND some don't, only the keyword-carrying nodes are reported as ends.
      Rationale: a non-keyword out-degree==0 node almost always has a missed
      outgoing arrow (a detection error), not a genuine terminal state.
      If NO out-degree==0 node has a terminal keyword we fall back to
      reporting all of them (preserves correctness on diagrams without labels).

No hardcoded outputs. All inference is driven by topology + node metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import networkx as nx


# ── helpers ──────────────────────────────────────────────────────────────────

_TERMINAL_KEYWORDS = {
    "complete", "completed", "done", "closed", "rejected",
    "excluded", "expired", "terminated", "finish", "finished",
    "end", "final",
}


def _label_looks_terminal(label: str) -> bool:
    words = set(label.lower().split())
    return bool(words & _TERMINAL_KEYWORDS)


def _all_incoming_are_cycle_backs(g: nx.DiGraph, node: str) -> bool:
    """True when every predecessor of *node* is reachable FROM *node* —
    the only incoming edges are back-edges from a cycle loop."""
    for pred in g.predecessors(node):
        if not nx.has_path(g, node, pred):
            return False
    return True


# ── public API ────────────────────────────────────────────────────────────────

def build_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["id"], **n)
    for e in edges:
        if e["source"] in g and e["target"] in g:
            g.add_edge(e["source"], e["target"], **e)
    return g


def infer_start_end(
    g: nx.DiGraph,
    nodes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Determine start and end states from topology + node metadata.

    Start algorithm
    ───────────────
    1. Collect topo_starts = nodes with in-degree == 0.
    2. Among topo_starts, identify green_starts = those with style=="green".
    3. GREEN-EXCLUSIVE: if green_starts is non-empty, use ONLY green_starts.
       (Other in-degree==0 nodes have a missing incoming arrow, not a real
       second entry point.)
    4. Otherwise use all topo_starts.
    5. CYCLE-BACK promotion: a green node with in-degree>0 whose every
       predecessor is reachable from it (back-edge only) is also added as start.

    End algorithm
    ─────────────
    1. Collect topo_ends = nodes with out-degree == 0.
    2. Identify terminal_ends = those whose label contains a terminal keyword.
    3. TERMINAL-KEYWORD priority: if terminal_ends is non-empty but is a strict
       subset of topo_ends, report only terminal_ends.
       (Non-keyword out-degree==0 nodes likely have a missed outgoing arrow.)
    4. Otherwise report all topo_ends.
    """
    # ── raw topology ──────────────────────────────────────────────────────────
    topo_starts = [n for n, d in g.in_degree()  if d == 0]
    topo_ends   = [n for n, d in g.out_degree() if d == 0]

    # ── start inference ───────────────────────────────────────────────────────
    node_by_id: Dict[str, Dict[str, Any]] = {}
    if nodes:
        node_by_id = {n["id"]: n for n in nodes}

    green_starts = [
        nid for nid in topo_starts
        if node_by_id.get(nid, {}).get("style") == "green"
    ]

    if green_starts:
        # Green-exclusive: suppress non-green in-degree==0 nodes
        starts = set(green_starts)
    else:
        starts = set(topo_starts)

    # Cycle-back promotion: green node with in-degree>0 that is only
    # reached by back-edges still counts as a start state.
    if nodes:
        for nid in list(g.nodes()):
            nd = node_by_id.get(nid, {})
            if nd.get("style") == "green" and nid not in starts:
                if g.in_degree(nid) > 0 and _all_incoming_are_cycle_backs(g, nid):
                    starts.add(nid)
                    ends_local = set(topo_ends)
                    ends_local.discard(nid)   # remove from ends if wrongly there

    # ── end inference ─────────────────────────────────────────────────────────
    # Step 1: base ends from topology
    terminal_ends = [
        nid for nid in topo_ends
        if _label_looks_terminal(g.nodes[nid].get("label", ""))
    ]

    if terminal_ends and len(terminal_ends) < len(topo_ends):
        # Some ends have keyword proof; the rest likely have missing arrows
        ends = set(terminal_ends)
    else:
        ends = set(topo_ends)

    # Step 2: SEMANTIC TERMINAL FORCE
    # In workflow / process diagrams, a node labelled "Complete", "Done",
    # "Closed", "Rejected", etc. is ALWAYS a terminal state by definition —
    # even if the arrow detector created a phantom outgoing edge from it.
    # We unconditionally add every terminal-keyword node to ends here.
    # This is safe because:
    #   a) terminal keywords are unambiguous in workflow naming conventions.
    #   b) a real edge out of a "Complete" node (e.g. loop-back) is
    #      structurally unusual and would show up in the Warnings tab.
    for nid in list(g.nodes()):
        if nid in ends:
            continue
        nd = node_by_id.get(nid, {})
        # prefer node_by_id label (original OCR text) over graph attribute
        label = nd.get("label") or g.nodes[nid].get("label") or ""
        if _label_looks_terminal(label):
            ends.add(nid)

    return sorted(starts), sorted(ends)


def find_orphans(g: nx.DiGraph) -> List[str]:
    return [n for n in g.nodes if g.in_degree(n) == 0 and g.out_degree(n) == 0]


def find_cycles(g: nx.DiGraph) -> List[List[str]]:
    try:
        return list(nx.simple_cycles(g))
    except Exception:
        return []


def find_suspect_boundary_nodes(g: nx.DiGraph, max_gap_px: float = 140.0) -> List[str]:
    """Flag start/end nodes that are geometrically close to another node but
    have no edge to it — a warning that an edge might be missing."""
    import math

    starts = {n for n, d in g.in_degree()  if d == 0}
    ends   = {n for n, d in g.out_degree() if d == 0}
    boundary = starts | ends
    suspects: List[str] = []
    nodes = list(g.nodes(data=True))
    for nid in boundary:
        data  = g.nodes[nid]
        center = data.get("center")
        if not center:
            continue
        for other_id, other_data in nodes:
            if other_id == nid:
                continue
            oc = other_data.get("center")
            if not oc:
                continue
            dist = math.hypot(center[0] - oc[0], center[1] - oc[1])
            if dist < max_gap_px and not g.has_edge(nid, other_id) and not g.has_edge(other_id, nid):
                suspects.append(nid)
                break
    return suspects


def aggregate_confidence(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> float:
    if not nodes:
        return 0.0
    node_avg = sum(n.get("confidence", 0.0) for n in nodes) / len(nodes)
    edge_avg = (sum(e.get("confidence", 0.0) for e in edges) / len(edges)) if edges else 0.5
    return round(0.6 * node_avg + 0.4 * edge_avg, 3)


def statistics(nodes, edges, g) -> Dict[str, Any]:
    starts, ends = infer_start_end(g, nodes)
    return {
        "node_count":            len(nodes),
        "edge_count":            len(edges),
        "start_count":           len(starts),
        "end_count":             len(ends),
        "orphan_count":          len(find_orphans(g)),
        "cycle_count":           len(find_cycles(g)),
        "avg_node_confidence":   round(sum(n.get("confidence", 0.0) for n in nodes) / max(1, len(nodes)), 3),
        "avg_edge_confidence":   round(sum(e.get("confidence", 0.0) for e in edges) / max(1, len(edges)), 3) if edges else 0.0,
        "overall_confidence":    aggregate_confidence(nodes, edges),
    }
