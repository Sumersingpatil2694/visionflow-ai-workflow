"""
VisionFlow AI — vlm_extractor.py
Vision-LLM extractor — primary node/edge detector.

Talks to a local Ollama server (default http://localhost:11434) using a
vision-capable model (MiniCPM-V, Qwen2.5-VL, Llama-Vision, ... — whichever
is set in config.PipelineConfig.vision_model). Falls back automatically:
pipeline.py only uses this module's output when vlm_available() is True and
extract_diagram() actually returns non-empty nodes; otherwise it drops back
to the classical CV path (node_detector.py + arrow_detector.py) untouched.

Public API expected by pipeline.py (unchanged across revisions):
    vlm_available(host, model) -> bool
    extract_diagram(host, model, image, timeout) -> {"nodes":[...], "edges":[...]} | None
    snap_boxes_to_cv(nodes, color_bgr, binary, cfg) -> List[node dict]

Output contract:
  - node dicts match node_detector.detect_nodes() exactly:
        id, label, bbox [x1,y1,x2,y2], center [cx,cy], shape, source,
        confidence, style
  - edge dicts match arrow_detector.detect_edges() exactly:
        id, source, target, direction, confidence, has_arrowhead,
        routing, coverage

No geometry is ever hardcoded here — everything is derived from the model's
response for THIS image, same rule as the rest of the pipeline. Every
network call and every parser is wrapped so a bad/missing model, a flaky
Ollama server, or malformed output can never crash the pipeline — the worst
case is an empty/None result, which pipeline.py already treats as "fall
back to classical CV".
"""
from __future__ import annotations

import base64
import difflib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests

from .prompts import build_vlm_extraction_prompt
from .node_detector import _clean_label, _style_from_crop, color_candidates
from .arrow_detector import _direction

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz  # type: ignore
except Exception:
    _rapidfuzz_fuzz = None

logger = logging.getLogger(__name__)

_OLLAMA_CHAT_RETRIES = 1          # one automatic retry on invalid/empty JSON
_JSON_RETRY_REMINDER = (
    "\n\nREMINDER: your previous response could not be parsed as JSON. "
    "Respond with ONLY the raw JSON object. No markdown, no prose, no code fences."
)


# ── availability ──────────────────────────────────────────────────────────────

def _list_installed_models(host: str, timeout: int = 4) -> List[str]:
    """Returns installed Ollama model names, or [] on any failure. Never
    raises — an unreachable Ollama server is a normal, expected state."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        if r.status_code != 200:
            logger.debug("Ollama /api/tags returned status %s", r.status_code)
            return []
        data = r.json() or {}
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except requests.exceptions.RequestException as exc:
        logger.debug("Ollama unreachable while listing models: %s", exc)
        return []
    except Exception as exc:
        logger.debug("Unexpected error listing Ollama models: %s", exc)
        return []


def _model_name_matches(installed: str, wanted: str) -> bool:
    """'minicpm-v:latest' should match a configured 'minicpm-v' (and vice
    versa), and small punctuation/version differences shouldn't break the
    match either. Works the same regardless of which vision model family
    (MiniCPM-V / Qwen2.5-VL / Llama-Vision) is configured."""
    if not installed or not wanted:
        return False
    a = installed.lower().strip()
    b = wanted.lower().strip()
    if a == b:
        return True
    a_base = a.split(":")[0]
    b_base = b.split(":")[0]
    if a_base == b_base:
        return True
    if _rapidfuzz_fuzz is not None:
        return _rapidfuzz_fuzz.ratio(a_base, b_base) >= 92
    return difflib.SequenceMatcher(None, a_base, b_base).ratio() >= 0.92


def vlm_available(host: str, model: str, timeout: int = 4) -> bool:
    """True only if Ollama is reachable AND the configured vision model is
    actually pulled locally. Never raises — pipeline.py treats any falsy
    result as "use classical CV instead"."""
    if not host or not model:
        return False
    installed = _list_installed_models(host, timeout=timeout)
    if not installed:
        return False
    return any(_model_name_matches(name, model) for name in installed)


# ── image encoding ───────────────────────────────────────────────────────────

def _encode_image_b64(bgr: np.ndarray) -> Optional[str]:
    try:
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as exc:
        logger.debug("Image encoding failed: %s", exc)
        return None


# ── Ollama chat call ─────────────────────────────────────────────────────────

def _call_ollama_vision(host: str, model: str, img_b64: str, prompt: str, timeout: int) -> Optional[str]:
    """One request/response round trip against the Ollama /api/chat endpoint.
    Returns the raw assistant text, or None on any failure. Compatible with
    the current Ollama chat API for any vision-capable model (MiniCPM-V,
    Qwen2.5-VL, Llama-Vision, ...) — nothing model-specific here."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            # Ollama's default num_predict is too small for busy diagrams —
            # a 10-node/13-edge JSON payload needs ~400-800 tokens. Without
            # this, the model's (grammar-constrained) JSON gets closed off
            # early and we silently lose the tail of the nodes/edges arrays.
            "num_predict": 4096,
        },
    }
    try:
        r = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout)
        if r.status_code != 200:
            # Some Ollama builds / model templates reject "format":"json" for
            # vision models — retry once without it before giving up.
            logger.debug("Ollama chat returned %s with format=json, retrying without it", r.status_code)
            payload.pop("format", None)
            r = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout)
            if r.status_code != 200:
                logger.debug("Ollama chat failed with status %s", r.status_code)
                return None
        data = r.json() or {}
        return (data.get("message") or {}).get("content", "")
    except requests.exceptions.Timeout:
        logger.debug("Ollama chat call timed out after %ss", timeout)
        return None
    except requests.exceptions.RequestException as exc:
        logger.debug("Ollama chat call failed: %s", exc)
        return None
    except Exception as exc:
        logger.debug("Unexpected error calling Ollama chat: %s", exc)
        return None


# ── robust JSON recovery ─────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _iter_balanced_json_objects(text: str) -> List[str]:
    """Scans the whole string for every top-level balanced {...} block,
    respecting string literals (so braces inside label text don't confuse
    the scanner). Vision models occasionally emit more than one JSON block
    (e.g. a stray example before the real answer) — returning all of them
    lets the caller pick the one that actually matches the schema instead
    of blindly trusting whichever came first."""
    blocks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escape = False
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        blocks.append(text[start:j + 1])
                        break
            j += 1
        i = j + 1
    return blocks


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_stray_escapes(text: str) -> str:
    """Vision models sometimes emit invalid backslash escapes (e.g. a raw
    '\\_' or a lone trailing '\\') that break strict JSON parsing. Escape
    any backslash that isn't followed by a valid JSON escape character."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _try_parse(candidate: str) -> Optional[Dict[str, Any]]:
    for transform in (
        lambda s: s,
        _remove_trailing_commas,
        _fix_stray_escapes,
        lambda s: _remove_trailing_commas(_fix_stray_escapes(s)),
    ):
        try:
            parsed = json.loads(transform(candidate))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _safe_json_parse(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Recovers from: markdown fences, leading/trailing prose, multiple
    JSON blocks in one response, trailing commas, nested braces, invalid
    escaping, and single-quoted JSON-ish output. Returns None (never
    raises) if nothing usable can be salvaged, so callers can fail safe."""
    if not raw or not raw.strip():
        return None

    cleaned = _strip_code_fences(raw)

    # direct attempt first (fast path — this is what a well-behaved model
    # with format="json" will produce almost every time)
    direct = _try_parse(cleaned)
    if direct is not None and ("nodes" in direct or "edges" in direct):
        return direct

    # scan for every balanced {...} block and prefer the one that actually
    # has a "nodes" key, largest first (most complete extraction)
    blocks = _iter_balanced_json_objects(cleaned)
    blocks.sort(key=len, reverse=True)
    schema_matches: List[Dict[str, Any]] = []
    other_matches: List[Dict[str, Any]] = []
    for block in blocks:
        parsed = _try_parse(block)
        if parsed is None:
            continue
        if "nodes" in parsed:
            schema_matches.append(parsed)
        else:
            other_matches.append(parsed)
    if schema_matches:
        return schema_matches[0]
    if other_matches:
        return other_matches[0]
    if direct is not None:
        return direct

    # last resort: single-quoted JSON-ish text
    if "'" in cleaned and '"' not in cleaned:
        parsed = _try_parse(cleaned.replace("'", '"'))
        if parsed is not None:
            return parsed

    return None


# ── node conversion ──────────────────────────────────────────────────────────

_KNOWN_SHAPES = {
    "rectangle", "rounded_rectangle", "diamond", "ellipse", "circle",
    "parallelogram", "polygon",
}


def _normalize_shape(shape: Any) -> str:
    s = str(shape or "").strip().lower().replace(" ", "_").replace("-", "_")
    if s in _KNOWN_SHAPES:
        return s
    match = difflib.get_close_matches(s, _KNOWN_SHAPES, n=1, cutoff=0.5)
    return match[0] if match else "rounded_rectangle"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
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


def _percent_bbox_to_px(bbox_pct: Any, width: int, height: int) -> Optional[List[int]]:
    if not isinstance(bbox_pct, (list, tuple)) or len(bbox_pct) != 4:
        return None
    try:
        x_pct, y_pct, w_pct, h_pct = [float(v) for v in bbox_pct]
    except (TypeError, ValueError):
        return None

    x_pct = _clamp(x_pct, 0, 100)
    y_pct = _clamp(y_pct, 0, 100)
    w_pct = _clamp(w_pct, 0.1, 100)
    h_pct = _clamp(h_pct, 0.1, 100)

    x1 = int(round((x_pct / 100.0) * width))
    y1 = int(round((y_pct / 100.0) * height))
    x2 = int(round(((x_pct + w_pct) / 100.0) * width))
    y2 = int(round(((y_pct + h_pct) / 100.0) * height))

    x1 = int(_clamp(x1, 0, width - 1))
    y1 = int(_clamp(y1, 0, height - 1))
    x2 = int(_clamp(x2, x1 + 1, width))
    y2 = int(_clamp(y2, y1 + 1, height))
    return [x1, y1, x2, y2]


def _nodes_from_vlm(raw_nodes: List[Any], color_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Converts the VLM's raw node list into the exact dict shape
    node_detector.detect_nodes() produces, so downstream modules
    (graph_builder, excel_export, the annotator in pipeline.py) can't tell
    the difference between a VLM node and a classical-CV node."""
    h, w = color_bgr.shape[:2]
    out: List[Dict[str, Any]] = []
    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        label = _clean_label(str(item.get("label", "")).strip())
        if not label:
            continue  # unlabeled boxes carry no value downstream — skip
        bbox = _percent_bbox_to_px(item.get("bbox"), w, h)
        if bbox is None:
            continue

        try:
            vlm_confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            vlm_confidence = 0.6
        if vlm_confidence <= 0.0:
            # A literal 0.0 almost always means the model echoed the prompt's
            # placeholder rather than reasoning about its actual certainty —
            # treat it as "not provided" instead of "genuinely zero".
            vlm_confidence = 0.6
        vlm_confidence = _clamp(vlm_confidence, 0.05, 0.99)

        x1, y1, x2, y2 = bbox
        out.append({
            "label": label,
            "bbox": bbox,
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "shape": _normalize_shape(item.get("shape")),
            "source": "vlm",
            "confidence": round(vlm_confidence, 3),
            "style": _style_from_crop(color_bgr, bbox),
            "_vlm_confidence": vlm_confidence,  # kept for downstream blending, stripped before return
        })

    # de-dup near-identical boxes (VLM occasionally repeats a node)
    deduped: List[Dict[str, Any]] = []
    for n in out:
        dup_idx = None
        for idx, k in enumerate(deduped):
            if _iou(n["bbox"], k["bbox"]) > 0.75 and _label_similarity(n["label"], k["label"]) > 0.92:
                dup_idx = idx
                break
        if dup_idx is None:
            deduped.append(n)
        elif n["confidence"] > deduped[dup_idx]["confidence"]:
            deduped[dup_idx] = n

    # stable reading order (top-to-bottom, left-to-right), then assign ids —
    # the same convention node_detector.py uses, so ids are deterministic.
    deduped.sort(key=lambda n: (n["center"][1], n["center"][0]))
    for i, n in enumerate(deduped, start=1):
        n["id"] = f"N{i}"

    return deduped


# ── fuzzy label matching for edges ───────────────────────────────────────────

def _normalize_for_match(text: str) -> str:
    """Normalizes case, whitespace and hyphen/underscore variants so
    'Sign-Off', 'sign off', and 'SignOff' all compare as equal-ish."""
    text = (text or "").strip().lower()
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _label_similarity(a: str, b: str) -> float:
    a_norm, b_norm = _normalize_for_match(a), _normalize_for_match(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if _rapidfuzz_fuzz is not None:
        return _rapidfuzz_fuzz.token_sort_ratio(a_norm, b_norm) / 100.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def _match_label_to_node(
    query: str,
    nodes: List[Dict[str, Any]],
    id_lookup: Dict[str, Dict[str, Any]],
    normalized_labels: Dict[str, str],
) -> Optional[str]:
    """Resolves a free-text 'from'/'to' string from the VLM into a node id.
    Handles OCR noise, spacing/hyphen differences, capitalization, and
    minor spelling mistakes via fuzzy matching (RapidFuzz if installed,
    difflib otherwise)."""
    query = (query or "").strip()
    if not query:
        return None

    # 1) exact id match (VLM occasionally echoes an id like "N1")
    if query.upper() in id_lookup:
        return query.upper()

    # 2) exact normalized-label match
    q_norm = _normalize_for_match(query)
    for node_id, label_norm in normalized_labels.items():
        if label_norm == q_norm:
            return node_id

    # 3) fuzzy match — best score wins, above a sane floor so unrelated
    # labels never get silently paired up
    best_id, best_score = None, 0.0
    for n in nodes:
        score = _label_similarity(query, n["label"])
        if score > best_score:
            best_score, best_id = score, n["id"]
    if best_id is not None and best_score >= 0.55:
        return best_id
    return None


def _edges_from_vlm(raw_edges: List[Any], nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts VLM edge (label -> label) pairs into internal (id -> id)
    edges. Robust to curved, diagonal, loop-back and crossing connectors
    since matching is purely label-based — the VLM's visual read of *which*
    two nodes are connected is trusted, not any particular line geometry."""
    if not nodes:
        return []

    id_lookup = {n["id"]: n for n in nodes}
    normalized_labels = {n["id"]: _normalize_for_match(n["label"]) for n in nodes}

    candidates: List[Dict[str, Any]] = []
    for item in raw_edges:
        if not isinstance(item, dict):
            continue
        src_label = str(item.get("from", "")).strip()
        tgt_label = str(item.get("to", "")).strip()
        if not src_label or not tgt_label:
            continue

        src_id = _match_label_to_node(src_label, nodes, id_lookup, normalized_labels)
        tgt_id = _match_label_to_node(tgt_label, nodes, id_lookup, normalized_labels)
        if not src_id or not tgt_id or src_id == tgt_id:
            continue

        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        if confidence <= 0.0:
            confidence = 0.6
        confidence = round(_clamp(confidence, 0.05, 0.99), 3)

        src_node, tgt_node = id_lookup[src_id], id_lookup[tgt_id]
        direction = _direction(src_node["center"], tgt_node["center"])

        candidates.append({
            "source": src_id,
            "target": tgt_id,
            "direction": direction,
            "confidence": confidence,
            "has_arrowhead": True,  # VLM only reports edges it visually saw as directed
            "routing": "vlm",
            "coverage": confidence,
        })

    # keep the best-confidence edge per (source, target) pair — also
    # collapses accidental duplicates from loop-back / crossing connectors
    # the model may have described twice.
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in candidates:
        key = (e["source"], e["target"])
        if key not in best or e["confidence"] > best[key]["confidence"]:
            best[key] = e

    final = []
    for i, e in enumerate(sorted(best.values(), key=lambda x: (-x["confidence"], x["source"], x["target"])), start=1):
        e["id"] = f"E{i}"
        final.append(e)
    return final


# ── main entry point ─────────────────────────────────────────────────────────

def extract_diagram(host: str, model: str, image: np.ndarray, timeout: int = 180) -> Optional[Dict[str, Any]]:
    """Sends `image` (BGR numpy array) to the configured Ollama vision model
    and returns {"nodes": [...], "edges": [...]} in the internal project
    format, or None if the call/parse failed entirely — pipeline.py treats
    None (or empty nodes) as "fall back to classical CV".

    Retries the Ollama call once, with a stronger JSON-only reminder
    appended to the prompt, if the first response can't be parsed at all.
    """
    if image is None or getattr(image, "size", 0) == 0:
        logger.debug("extract_diagram called with empty/None image")
        return None

    img_b64 = _encode_image_b64(image)
    if not img_b64:
        return None

    prompt = build_vlm_extraction_prompt()
    data: Optional[Dict[str, Any]] = None

    for attempt in range(_OLLAMA_CHAT_RETRIES + 1):
        raw = _call_ollama_vision(host, model, img_b64, prompt, timeout=timeout)
        data = _safe_json_parse(raw)
        if data is not None:
            break
        if attempt < _OLLAMA_CHAT_RETRIES:
            logger.debug("VLM JSON parse failed (attempt %d) — retrying", attempt + 1)
            prompt = build_vlm_extraction_prompt() + _JSON_RETRY_REMINDER

    if not data or not isinstance(data, dict):
        logger.debug("VLM extraction gave up after retries — no usable JSON")
        return None

    raw_nodes = data.get("nodes")
    raw_edges = data.get("edges")
    if not isinstance(raw_nodes, list):
        return None
    if not isinstance(raw_edges, list):
        raw_edges = []

    nodes = _nodes_from_vlm(raw_nodes, image)
    if not nodes:
        return {"nodes": [], "edges": []}

    edges = _edges_from_vlm(raw_edges, nodes)

    for n in nodes:
        n.pop("_vlm_confidence", None)

    return {"nodes": nodes, "edges": edges}


# ── CV box refinement ────────────────────────────────────────────────────────

def _contour_candidate_near(binary: np.ndarray, bbox: Sequence[int], margin: int) -> Optional[Tuple[List[int], float]]:
    """Best matching binary-image contour box near `bbox`, plus its IoU
    against the original box. None if nothing plausible is found."""
    h, w = binary.shape[:2]
    x1, y1, x2, y2 = bbox
    rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
    rx2, ry2 = min(w, x2 + margin), min(h, y2 + margin)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = binary[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    orig_area = max(1, (x2 - x1) * (y2 - y1))
    orig_local = [x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1]

    best_bbox, best_score = None, 0.0
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < 0.25 * orig_area or area > 4.0 * orig_area:
            continue
        score = _iou(orig_local, [cx, cy, cx + cw, cy + ch])
        if score > best_score:
            best_score = score
            best_bbox = [cx + rx1, cy + ry1, cx + cw + rx1, cy + ch + ry1]

    if best_bbox is not None and best_score >= 0.30:
        return best_bbox, best_score
    return None


def _color_candidate_near(
    color_candidates_list: List[Dict[str, Any]], bbox: Sequence[int], margin: int
) -> Optional[Tuple[List[int], float]]:
    """Best matching colored-border box (from node_detector.color_candidates)
    near `bbox`, plus its IoU against the original box. Colored workflow
    boxes usually have a cleaner, more reliable outline than a generic
    binary-image contour, so this is tried as a second signal alongside
    the contour match and the higher-confidence one wins."""
    if not color_candidates_list:
        return None
    x1, y1, x2, y2 = bbox
    search_box = [x1 - margin, y1 - margin, x2 + margin, y2 + margin]

    best_bbox, best_score = None, 0.0
    for cand in color_candidates_list:
        cbbox = cand["bbox"]
        if _iou(search_box, cbbox) <= 0.0:
            continue  # not even in the search neighborhood
        score = _iou(bbox, cbbox)
        if score > best_score:
            best_score = score
            best_bbox = cbbox

    if best_bbox is not None and best_score >= 0.30:
        return list(best_bbox), best_score
    return None


def _blend_confidence(vlm_confidence: float, refinement_quality: Optional[float]) -> float:
    """Combines the VLM's self-reported confidence with how well the box
    could be geometrically confirmed against real CV evidence (contour or
    colored-border overlap). A good geometric match raises confidence
    toward the VLM's own estimate; no match at all pulls it down slightly
    since the box is unverified rather than wrong."""
    if refinement_quality is None:
        return round(_clamp(vlm_confidence * 0.92, 0.05, 0.99), 3)
    blended = 0.6 * vlm_confidence + 0.4 * refinement_quality
    return round(_clamp(blended, 0.05, 0.99), 3)


def snap_boxes_to_cv(
    nodes: List[Dict[str, Any]],
    color_bgr: np.ndarray,
    binary: np.ndarray,
    cfg: Any,
) -> List[Dict[str, Any]]:
    """Best-effort pixel-accurate refinement of each VLM node box using two
    independent CV signals — colored node borders (node_detector.color_
    candidates) and generic binary-image contours — searched in a margin
    proportional to the box size. Whichever signal overlaps the VLM box
    better wins; if neither clears the confidence floor the original VLM
    box is kept untouched. Final per-node confidence blends the VLM's own
    estimate with this geometric agreement, so a box that both the model
    and classical CV agree on scores higher than one only the model saw.

    Never removes or relabels a node — only tightens geometry and updates
    confidence/style for the (possibly) new box.
    """
    if not nodes or binary is None:
        return nodes

    try:
        color_cands = color_candidates(color_bgr, cfg) if cfg is not None else []
    except Exception as exc:
        logger.debug("color_candidates() failed during snap_boxes_to_cv: %s", exc)
        color_cands = []

    refined: List[Dict[str, Any]] = []
    for n in nodes:
        bbox = n.get("bbox")
        if not bbox or len(bbox) != 4:
            refined.append(n)
            continue

        n = dict(n)  # avoid mutating the caller's dict in place
        x1, y1, x2, y2 = bbox
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        margin = int(max(8, 0.25 * min(box_w, box_h)))

        color_match = _color_candidate_near(color_cands, bbox, margin)
        contour_match = None
        try:
            contour_match = _contour_candidate_near(binary, bbox, margin)
        except Exception as exc:
            logger.debug("contour refinement failed for node %s: %s", n.get("id"), exc)

        best_match = None
        if color_match and contour_match:
            best_match = color_match if color_match[1] >= contour_match[1] else contour_match
        else:
            best_match = color_match or contour_match

        vlm_confidence = n.get("_vlm_confidence", n.get("confidence", 0.6))
        if best_match is not None:
            snapped_bbox, quality = best_match
            sx1, sy1, sx2, sy2 = snapped_bbox
            n["bbox"] = [int(sx1), int(sy1), int(sx2), int(sy2)]
            n["center"] = [int((sx1 + sx2) / 2), int((sy1 + sy2) / 2)]
            n["style"] = _style_from_crop(color_bgr, n["bbox"])
            n["confidence"] = _blend_confidence(vlm_confidence, quality)
        else:
            n["confidence"] = _blend_confidence(vlm_confidence, None)

        n.pop("_vlm_confidence", None)
        refined.append(n)

    return refined
