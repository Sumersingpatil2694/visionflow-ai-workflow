"""
Node detector.

Signals used:
  1) OCR text clusters (when OCR is available)
  2) Colored node-border candidates (excellent for workflow screenshots)
  3) Generic contour candidates on the binary image

We merge, deduplicate, then OCR each crop for the final label.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import re


def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _contain(a: List[int], b: List[int]) -> float:
    """How much of b is inside a."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / area_b


def _clean_label(text: str) -> str:
    text = (text or "").replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:[a-zA-Z]{0,2}[^A-Za-z0-9\s-]{1,3}\s+)+", "", text).strip()
    return text


def _style_from_crop(color_bgr: np.ndarray, bbox: List[int]) -> str:
    x1, y1, x2, y2 = bbox
    h, w = color_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    crop = color_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    colored = hsv[(sat > 30) & (val > 70)]
    if colored.size == 0:
        return "neutral"
    hue = float(np.median(colored[:, 0]))
    if 35 <= hue <= 90:
        return "green"
    if 90 < hue <= 135:
        return "blue"
    if 0 <= hue < 20 or hue > 160:
        return "red_or_orange"
    if 20 <= hue < 35:
        return "yellow"
    return "colored"


def color_candidates(color_bgr: np.ndarray, cfg) -> List[Dict[str, Any]]:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > 25) & (val > 80)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < max(1200, cfg.min_node_area * 0.6):
            continue
        if w < cfg.min_node_width or h < cfg.min_node_height:
            continue
        if w > cfg.max_node_width or h > max(cfg.max_node_height + 30, 80):
            continue
        ar = w / max(h, 1)
        if ar < 1.2 or ar > cfg.max_node_aspect:
            continue
        cands.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "shape": "rounded_rectangle",
            "source": "color_border",
        })
    return cands


def contour_candidates(binary: np.ndarray, cfg) -> List[Dict[str, Any]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    H, W = binary.shape[:2]
    img_area = H * W
    cands: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < cfg.min_node_area:
            continue
        if area > 0.55 * img_area:
            continue
        if w < cfg.min_node_width or h < cfg.min_node_height:
            continue
        if w > cfg.max_node_width or h > cfg.max_node_height:
            continue
        ar = w / max(h, 1)
        if ar < cfg.min_node_aspect or ar > cfg.max_node_aspect:
            continue
        contour_area = cv2.contourArea(cnt)
        density = contour_area / max(1, area)
        if density < 0.12:   # raised from 0.05 to reject merged line blobs
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        shape = "rounded_rectangle" if 4 <= len(approx) <= 12 else "polygon"
        cands.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "shape": shape,
            "source": "contour",
        })
    return cands


def ocr_cluster_candidates(ocr_items: List[Dict[str, Any]], img_shape) -> List[Dict[str, Any]]:
    if not ocr_items:
        return []
    H, W = img_shape[:2]
    dy_thresh = max(12, min(40, H // 80))
    dx_thresh = max(40, min(110, W // 16))

    sorted_items = sorted(ocr_items, key=lambda it: (it["bbox"][1], it["bbox"][0]))
    clusters: List[List[Dict[str, Any]]] = []
    for it in sorted_items:
        placed = False
        for cl in clusters:
            for member in cl:
                mx1, my1, mx2, my2 = member["bbox"]
                ix1, iy1, ix2, iy2 = it["bbox"]
                horizontally_close = (ix1 - mx2 < dx_thresh and mx1 - ix2 < dx_thresh)
                vertically_close = (iy1 - my2 < dy_thresh and my1 - iy2 < dy_thresh)
                if horizontally_close and vertically_close:
                    cl.append(it)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([it])

    cands: List[Dict[str, Any]] = []
    for cl in clusters:
        xs1 = [m["bbox"][0] for m in cl]
        ys1 = [m["bbox"][1] for m in cl]
        xs2 = [m["bbox"][2] for m in cl]
        ys2 = [m["bbox"][3] for m in cl]
        pad_x, pad_y = 10, 6
        bbox = [
            max(0, min(xs1) - pad_x),
            max(0, min(ys1) - pad_y),
            min(W, max(xs2) + pad_x),
            min(H, max(ys2) + pad_y),
        ]
        label = _clean_label(" ".join(m["text"] for m in cl).strip())
        conf = float(np.mean([m["confidence"] for m in cl])) if cl else 0.0
        if label:
            cands.append({
                "bbox": bbox,
                "shape": "rounded_rectangle",
                "source": "ocr_cluster",
                "label": label,
                "ocr_confidence": round(conf, 3),
            })
    return cands


def merge_and_dedupe(candidates: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    order_key = {"ocr_cluster": 0, "color_border": 1, "contour": 2}
    candidates = sorted(
        candidates,
        key=lambda c: (
            order_key.get(c["source"], 9),
            -((c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]))
        )
    )
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        drop = False
        for k in kept:
            if _iou(c["bbox"], k["bbox"]) > iou_thr:
                drop = True; break
            if _contain(k["bbox"], c["bbox"]) > 0.90:
                drop = True; break
        if not drop:
            kept.append(c)
    return kept


def detect_nodes(
    color_bgr: np.ndarray,
    gray: np.ndarray,
    binary: np.ndarray,
    ocr_items: List[Dict[str, Any]],
    ocr_engine,
    cfg,
) -> List[Dict[str, Any]]:
    ocr_cands = ocr_cluster_candidates(ocr_items, color_bgr.shape)
    color_cands = color_candidates(color_bgr, cfg)
    contour_cands = contour_candidates(binary, cfg)
    merged = merge_and_dedupe(ocr_cands + color_cands + contour_cands, cfg.overlap_iou_threshold)

    final: List[Dict[str, Any]] = []
    for idx, cand in enumerate(merged, start=1):
        bbox = cand["bbox"]
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        label = _clean_label(cand.get("label", ""))
        base_conf = cand.get("ocr_confidence", 0.0)
        if not label:
            crop_label = ocr_engine.read_crop(color_bgr, bbox, upscale=cfg.ocr_upscale, pad=cfg.node_padding)
            label = _clean_label(crop_label or "")
            base_conf = 0.62 if label else 0.18
        if not label:
            continue
        aspect = w / max(h, 1)
        geom_score = 1.0
        if aspect < 1.1 or aspect > 10.0:
            geom_score *= 0.8
        area = w * h
        if area < cfg.min_node_area * 1.2:
            geom_score *= 0.9
        confidence = round(min(0.99, 0.55 * base_conf + 0.45 * geom_score), 3)
        final.append({
            "id": f"N{idx}",
            "label": label,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "center": [int(x1 + w / 2), int(y1 + h / 2)],
            "shape": cand.get("shape", "rounded_rectangle"),
            "source": cand.get("source", "unknown"),
            "confidence": confidence,
            "style": _style_from_crop(color_bgr, bbox),
        })

    best_by_label: Dict[str, Dict[str, Any]] = {}
    for n in final:
        key = " ".join(n["label"].lower().split())
        if not key:
            continue
        if key not in best_by_label or n["confidence"] > best_by_label[key]["confidence"]:
            best_by_label[key] = n
    final = list(best_by_label.values())

    final.sort(key=lambda n: (n["center"][1], n["center"][0]))
    for i, n in enumerate(final, start=1):
        n["id"] = f"N{i}"

    return final
