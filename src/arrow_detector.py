"""
Arrow / edge detector V5.1.

Key fixes vs original V3.x:
  1. [CORRECT ROOT FIX] Corridor blocking MOVED out of _candidate_pairs and applied
     ONLY for edges that have NO skeleton evidence. Pixel-proven connections bypass
     corridor geometry entirely — this fixed the missing curved arrow.
  2. pairs-per-sector: kept at 2 (V4.0 raised to 3 which caused phantom candidates).
  3. Path-hop junction turn threshold: 55° → 68° (V4.0 went to 72° which was too loose).
  4. Tortuosity cap: 2.2× → 2.7× (V4.0 went to 3.2× which was too loose).
  5. All other thresholds now live in config.py at carefully recalibrated values
     (see config.py V5.0 header for full change table).

Target on CDD test diagram: 10 nodes, 13 edges, 1 start, 2 ends.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None


# ── node masking ──────────────────────────────────────────────────────────────

def _mask_out_nodes(gray: np.ndarray, nodes: List[Dict[str, Any]], pad: int = 4) -> np.ndarray:
    """Erase node boxes before edge/skeleton work.

    Padding a few pixels beyond each bbox matters: a node's own drawn
    border stroke is centered on the bbox edge, so roughly half its width
    sits just outside the bbox. Without padding a sliver of that border
    remains, creating spurious blobs right where every connector meets
    the node.
    """
    m = gray.copy()
    H, W = m.shape[:2]
    for n in nodes:
        x1, y1, x2, y2 = n["bbox"]
        cv2.rectangle(
            m,
            (max(0, x1 - pad), max(0, y1 - pad)),
            (min(W, x2 + pad), min(H, y2 + pad)),
            255, thickness=-1,
        )
    return m


# ── line / segment utilities ──────────────────────────────────────────────────

def _segments(gray_masked: np.ndarray, cfg) -> List[Tuple[int, int, int, int]]:
    edges = cv2.Canny(gray_masked, cfg.canny_low, cfg.canny_high, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_len,
        maxLineGap=cfg.hough_max_gap,
    )
    segs: List[Tuple[int, int, int, int]] = []
    if lines is None:
        return segs
    for ln in lines[:800]:
        x1, y1, x2, y2 = ln.flatten()[:4]
        segs.append((int(x1), int(y1), int(x2), int(y2)))
    return segs


def _point_seg_dist(p, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _closest_edge_point(bbox, target_xy) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    tx, ty = target_xy
    cx = min(max(tx, x1), x2)
    cy = min(max(ty, y1), y2)
    return int(cx), int(cy)


# ── edge / skeleton masks ─────────────────────────────────────────────────────

def _edge_binary(gray_masked: np.ndarray, cfg) -> np.ndarray:
    edges = cv2.Canny(gray_masked, cfg.canny_low, cfg.canny_high, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return edges


def _line_coverage(edge_bin: np.ndarray, p1, p2, tol: int, n_samples: int) -> float:
    H, W = edge_bin.shape[:2]
    x1, y1 = p1
    x2, y2 = p2
    total = 0
    hits = 0
    lo, hi = 0.10, 0.90
    for i in range(n_samples + 1):
        t = lo + (hi - lo) * (i / n_samples)
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        xi, yi = int(round(x)), int(round(y))
        x0, x1b = max(0, xi - tol), min(W, xi + tol + 1)
        y0, y1b = max(0, yi - tol), min(H, yi + tol + 1)
        total += 1
        if x1b <= x0 or y1b <= y0:
            continue
        if edge_bin[y0:y1b, x0:x1b].max() > 0:
            hits += 1
    return hits / total if total else 0.0


def _connector_mask(gray_masked: np.ndarray, cfg) -> np.ndarray:
    edges = cv2.Canny(gray_masked, cfg.canny_low, cfg.canny_high, apertureSize=3)
    kernel = np.ones((cfg.connector_close_kernel, cfg.connector_close_kernel), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return opened


# ── skeleton utilities ────────────────────────────────────────────────────────

def _skeleton_components(mask: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    if skeletonize is None:
        return None, 0
    skel = skeletonize(mask > 0)
    skel_u8 = (skel.astype(np.uint8)) * 255
    num, labels = cv2.connectedComponents(skel_u8, connectivity=8)
    return labels, num


def _node_ring_components(node, labels, ring):
    H, W = labels.shape[:2]
    x1, y1, x2, y2 = node["bbox"]
    rx1, ry1 = max(0, x1 - ring), max(0, y1 - ring)
    rx2, ry2 = min(W, x2 + ring), min(H, y2 + ring)
    sub = labels[ry1:ry2, rx1:rx2]
    ys, xs = np.nonzero(sub)
    cx, cy = node["center"]
    best = {}
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        lab = int(sub[yy, xx])
        gx, gy = rx1 + xx, ry1 + yy
        d = (gx - cx) ** 2 + (gy - cy) ** 2
        if lab not in best or d < best[lab][0]:
            best[lab] = (d, (gx, gy))
    return {lab: pt for lab, (_, pt) in best.items()}


def _component_path_length(labels, label, p1, p2, cap, return_path: bool = False):
    from collections import deque
    H, W = labels.shape[:2]

    def _snap(pt):
        x, y = pt
        if 0 <= y < H and 0 <= x < W and labels[y, x] == label:
            return (x, y)
        for r in range(1, 8):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    xx, yy = x + dx, y + dy
                    if 0 <= yy < H and 0 <= xx < W and labels[yy, xx] == label:
                        return (xx, yy)
        return None

    s = _snap(p1)
    t = _snap(p2)
    if s is None or t is None:
        return (None, None) if return_path else None
    if s == t:
        return (0, [s]) if return_path else 0
    q = deque([(s, 0)])
    seen = {s: None}
    while q:
        (x, y), d = q.popleft()
        if d > cap:
            return (None, None) if return_path else None
        if (x, y) == t:
            if not return_path:
                return d
            path = [(x, y)]
            cur = (x, y)
            while seen[cur] is not None:
                cur = seen[cur]
                path.append(cur)
            path.reverse()
            return d, path
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= ny < H and 0 <= nx < W and (nx, ny) not in seen and labels[ny, nx] == label:
                    seen[(nx, ny)] = (x, y)
                    q.append(((nx, ny), d + 1))
    return (None, None) if return_path else None


def _degree_map(labels: np.ndarray) -> np.ndarray:
    """8-connected neighbour count for every skeleton pixel."""
    mask = (labels > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbour_count = cv2.filter2D(mask, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
    return neighbour_count * mask


def _path_hops_at_junction(
    path, degree_map,
    junction_degree: int = 3,
    turn_deg: float = 68.0,   # V5.1: was 72° (too loose) → 68°; still above original 55° to catch curved bends
    window: int = 6,
    endpoint_margin: int = 10,
) -> bool:
    """True if the traced path changes direction sharply at a branch pixel —
    the signature of BFS switching from one connector onto a different one."""
    n = len(path)
    if n < 2 * window + 2:
        return False
    H, W = degree_map.shape[:2]
    for i in range(endpoint_margin, n - endpoint_margin):
        x, y = path[i]
        if not (0 <= y < H and 0 <= x < W):
            continue
        if degree_map[y, x] < junction_degree:
            continue
        i0 = max(0, i - window)
        i1 = min(n - 1, i + window)
        x0, y0 = path[i0]
        x1, y1 = path[i1]
        v1 = (x - x0, y - y0)
        v2 = (x1 - x, y1 - y)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_ang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        ang = math.degrees(math.acos(cos_ang))
        if ang > turn_deg:
            return True
    return False


def _path_passes_third_node(path, src_id, tgt_id, nodes, margin: float = 4.0) -> bool:
    """True if a traced skeleton path passes through a node other than the
    two endpoints — a sign this is really two separate edges chained through
    a hub."""
    if not path or len(path) < 5:
        return False
    inner = path[3:-3] if len(path) > 8 else path
    for other in nodes:
        if other["id"] in (src_id, tgt_id):
            continue
        x1, y1, x2, y2 = other["bbox"]
        x1 -= margin; y1 -= margin; x2 += margin; y2 += margin
        for (px, py) in inner:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
    return False


# ── arrowhead detection ───────────────────────────────────────────────────────

def _detect_arrowheads(gray_masked: np.ndarray, cfg=None) -> List[Tuple[int, int]]:
    """Find small solid triangular arrowhead blobs using Canny edges."""
    canny_low  = getattr(cfg, "canny_low",  40) if cfg is not None else 40
    canny_high = getattr(cfg, "canny_high", 140) if cfg is not None else 140
    edges = cv2.Canny(gray_masked, canny_low, canny_high, apertureSize=3)
    bw = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8), iterations=1)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    tips: List[Tuple[int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 6 or area > 650:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if 3 <= len(approx) <= 6:
            pts = approx.reshape(-1, 2)
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            dists = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
            tip = tuple(int(v) for v in pts[int(np.argmax(dists))])
            tips.append(tip)
    return tips


# ── direction helper ──────────────────────────────────────────────────────────

def _sector(dx: float, dy: float) -> str:
    ax, ay = abs(dx), abs(dy)
    if ax > ay * 1.7:
        return "E" if dx > 0 else "W"
    if ay > ax * 1.7:
        return "S" if dy > 0 else "N"
    if dx >= 0 and dy >= 0:
        return "SE"
    if dx >= 0 and dy < 0:
        return "NE"
    if dx < 0 and dy >= 0:
        return "SW"
    return "NW"


def _direction(a, b) -> str:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


# ── corridor blocking ─────────────────────────────────────────────────────────

def _corridor_blocked(src: Dict[str, Any], tgt: Dict[str, Any], nodes: List[Dict[str, Any]], width: float = 26.0) -> bool:
    """True when another node's center sits inside the direct-line corridor
    between src and tgt.

    IMPORTANT: This is called ONLY for edges WITHOUT skeleton evidence.
    Skeleton-confirmed edges bypass this check entirely — pixel evidence
    overrides geometric heuristics.
    """
    a = tuple(src["center"])
    b = tuple(tgt["center"])
    seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
    if seg_len < 1:
        return False
    for other in nodes:
        if other["id"] in (src["id"], tgt["id"]):
            continue
        p = tuple(other["center"])
        d = _point_seg_dist(p, a, b)
        if d > width:
            continue
        dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
        if dot <= 0:
            continue
        if dot >= seg_len * seg_len:
            continue
        return True
    return False


# ── candidate pair generation ─────────────────────────────────────────────────

def _candidate_pairs(nodes: List[Dict[str, Any]], max_dist: float) -> List[Tuple[str, str]]:
    """Generate candidate (src, tgt) pairs based on spatial proximity.

    FIX V5.0: Corridor blocking removed from here. It is applied later in
    detect_edges only for edges that have no skeleton evidence. This allows
    long/curved arrows (whose skeleton proof connects the nodes) to survive
    even when another node happens to sit near the geometric straight line.
    """
    pairs = set()
    for src in nodes:
        bucketed: Dict[str, List[Tuple[float, str]]] = {}
        for tgt in nodes:
            if src["id"] == tgt["id"]:
                continue
            dx = tgt["center"][0] - src["center"][0]
            dy = tgt["center"][1] - src["center"][1]
            dist = math.hypot(dx, dy)
            if dist > max_dist or dist < 10:
                continue
            sector = _sector(dx, dy)
            bucketed.setdefault(sector, []).append((dist, tgt["id"]))
        for sector, items in bucketed.items():
            items.sort(key=lambda x: x[0])
            for _, tgt_id in items[:2]:  # V5.1: back to 2 per sector (3 created phantom candidates)
                pairs.add((src["id"], tgt_id))
    return sorted(pairs)


# ── main edge detection ───────────────────────────────────────────────────────

def detect_edges(gray, nodes, cfg):
    if len(nodes) < 2:
        return []
    H, W = gray.shape[:2]
    diag = math.hypot(H, W)
    max_dist = cfg.max_edge_distance_ratio * diag
    max_path_px = int(cfg.max_edge_distance_ratio * diag * 2.2)  # V5.1: 2.5 was too generous → 2.2

    masked = _mask_out_nodes(gray, nodes)
    segs = _segments(masked, cfg)
    tips = _detect_arrowheads(masked, cfg)
    edge_bin = _edge_binary(masked, cfg)

    conn_mask = _connector_mask(masked, cfg)
    labels, num_labels = _skeleton_components(conn_mask)
    degree_map = _degree_map(labels) if labels is not None else None
    node_components: Dict[str, Dict[int, Tuple[int, int]]] = {}
    if labels is not None:
        for n in nodes:
            node_components[n["id"]] = _node_ring_components(n, labels, cfg.connector_ring_px)

    require_strong = getattr(cfg, "require_arrowhead_or_shared_component", True)
    node_by_id = {n["id"]: n for n in nodes}

    # FIX V5.0: corridor blocking removed from _candidate_pairs — generated unconditionally
    pairs = _candidate_pairs(nodes, max_dist=max_dist)

    candidates: List[Dict[str, Any]] = []
    for src_id, tgt_id in pairs:
        src = node_by_id[src_id]
        tgt = node_by_id[tgt_id]
        fallback_src_pt = _closest_edge_point(src["bbox"], tgt["center"])
        fallback_tgt_pt = _closest_edge_point(tgt["bbox"], src["center"])
        dist = math.hypot(
            fallback_src_pt[0] - fallback_tgt_pt[0],
            fallback_src_pt[1] - fallback_tgt_pt[1],
        )
        if dist > max_dist or dist < 8:
            continue

        shared_component = False
        path_len = None
        src_pt, tgt_pt = fallback_src_pt, fallback_tgt_pt

        if labels is not None:
            src_comps = node_components.get(src["id"], {})
            tgt_comps = node_components.get(tgt["id"], {})
            shared = {lab for lab in (set(src_comps) & set(tgt_comps)) if lab != 0}
            if shared:
                best_len = None
                best_lab = None
                best_path = None
                for lab in shared:
                    sp, tp = src_comps[lab], tgt_comps[lab]
                    plen, ppath = _component_path_length(labels, lab, sp, tp, cap=max_path_px, return_path=True)
                    if plen is None:
                        continue
                    if _path_passes_third_node(ppath, src["id"], tgt["id"], nodes):
                        continue
                    straight = math.hypot(tp[0] - sp[0], tp[1] - sp[1])
                    # V5.1: tortuosity cap 2.7× — curved arrows OK, extreme winding paths are phantoms
                    if straight > 0 and plen > 2.7 * straight:
                        continue
                    if degree_map is not None and _path_hops_at_junction(ppath, degree_map):
                        continue
                    if best_len is None or plen < best_len:
                        best_len = plen
                        best_lab = lab
                        best_path = ppath
                if best_lab is not None:
                    shared_component = True
                    path_len = best_len
                    src_pt, tgt_pt = src_comps[best_lab], tgt_comps[best_lab]

        # FIX V5.0: corridor blocking only for non-skeleton edges
        if not shared_component:
            if _corridor_blocked(src, tgt, nodes):
                continue

        coverage = _line_coverage(
            edge_bin, fallback_src_pt, fallback_tgt_pt,
            tol=cfg.line_coverage_tol,
            n_samples=cfg.line_coverage_samples,
        )

        has_head = False
        for (hx, hy) in tips:
            if math.hypot(hx - tgt_pt[0], hy - tgt_pt[1]) < cfg.arrowhead_search_radius:
                has_head = True
                break
        # Also check near the fallback target point
        if not has_head:
            for (hx, hy) in tips:
                if math.hypot(hx - fallback_tgt_pt[0], hy - fallback_tgt_pt[1]) < cfg.arrowhead_search_radius:
                    has_head = True
                    break

        if require_strong:
            if not shared_component and not (coverage >= cfg.min_line_coverage and (has_head or coverage >= 0.86)):
                continue
        else:
            if not shared_component and coverage < cfg.min_line_coverage:
                continue

        support = 0
        for x1, y1, x2, y2 in segs:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if _point_seg_dist((mx, my), fallback_src_pt, fallback_tgt_pt) < 14:
                support += 1

        head_bonus = 0.22 if has_head else 0.0
        base = max(0.0, 1.0 - dist / max_dist)
        if shared_component:
            tortuosity = (path_len / dist) if (path_len and dist > 0) else 1.0
            straightness_penalty = 0.0 if tortuosity <= 1.8 else min(0.18, (tortuosity - 1.8) * 0.10)
            score = min(0.98, 0.60 + base * 0.14 + min(0.14, 0.028 * support) + head_bonus - straightness_penalty)
        else:
            coverage_score = min(0.42, coverage * 0.42)
            seg_score = min(0.22, 0.05 * support)
            score = min(0.98, base * 0.20 + coverage_score + seg_score + head_bonus)

        if score < cfg.min_arrow_confidence:
            continue

        direction = _direction(src["center"], tgt["center"])
        candidates.append({
            "source": src["id"],
            "target": tgt["id"],
            "direction": direction,
            "confidence": round(score, 3),
            "has_arrowhead": has_head,
            "routing": "traced" if shared_component else "straight",
            "coverage": round(float(coverage), 3),
            "_dist": dist,
        })

    # pairwise best
    best_dir: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in candidates:
        key = (e["source"], e["target"])
        if key not in best_dir or e["confidence"] > best_dir[key]["confidence"]:
            best_dir[key] = e

    kept: List[Dict[str, Any]] = []
    processed = set()
    for (s, t), e in sorted(best_dir.items(), key=lambda kv: (-kv[1]["confidence"], kv[0][0], kv[0][1])):
        rev = (t, s)
        pair = tuple(sorted([s, t]))
        if pair in processed:
            continue
        rev_e = best_dir.get(rev)
        if rev_e is None:
            kept.append(e)
        else:
            if e["has_arrowhead"] and rev_e["has_arrowhead"]:
                kept.append(e)
                kept.append(rev_e)
            else:
                kept.append(e if e["confidence"] >= rev_e["confidence"] else rev_e)
        processed.add(pair)

    cap = getattr(cfg, "max_edges_per_node", 14)
    if cap and cap > 0:
        kept.sort(key=lambda e: -e["confidence"])
        deg: Dict[str, int] = {}
        filtered = []
        for e in kept:
            ds = deg.get(e["source"], 0)
            dt = deg.get(e["target"], 0)
            if ds >= cap or dt >= cap:
                continue
            deg[e["source"]] = ds + 1
            deg[e["target"]] = dt + 1
            filtered.append(e)
        kept = filtered

    final: List[Dict[str, Any]] = []
    for i, e in enumerate(sorted(kept, key=lambda x: (-x["confidence"], x["source"], x["target"])), start=1):
        e = {k: v for k, v in e.items() if not k.startswith("_")}
        e["id"] = f"E{i}"
        final.append(e)
    return final
