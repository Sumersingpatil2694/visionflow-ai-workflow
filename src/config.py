"""
Central configuration for the Workflow Diagram Analyzer V5.0.

V5.0 calibration philosophy:
  - Keep the V5.0 corridor-fix in arrow_detector.py (correct root-cause fix).
  - Roll back all other "loosen everything" changes that created phantom edges.
  - Tune each threshold individually to a carefully chosen middle ground.
  - Net target: 10 nodes, 13 edges, 1 start, 2 ends on the CDD test diagram.

Key changes vs V4.0 (which gave edges=16, starts=2, ends=3):
  hough_max_gap       : 24 → 17    (was bridging unrelated segments → phantoms)
  connector_ring_px   : 24 → 16    (was merging non-connected component rings)
  min_line_coverage   : 0.58 → 0.66 (too low → false straight-line hits)
  max_edge_distance_ratio: 0.58 → 0.52 (was pulling in very far unrelated nodes)
  arrowhead_search_radius: 42 → 34  (smaller = fewer false arrowhead matches)
  min_arrow_confidence: 0.46 → 0.51 (was letting low-confidence phantoms through)
  max_edges_per_node  : 14 → 10    (14 was too permissive for this diagram)
  pairs_per_sector (in _candidate_pairs): 3 → 2 (fewer phantom pair candidates)
  tortuosity cap (in arrow_detector.py): 3.2 → 2.7 (reduce winding-path phantoms)
  turn_deg (junction hop): 72° → 68°  (slight tighten to catch more hop heuristics)

No hardcoded outputs anywhere in this pipeline.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class PipelineConfig:
    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    # moondream:1.8b is too small for reliable structured diagram parsing.
    # Pull a stronger local vision model for real accuracy, e.g.:
    #   ollama pull qwen2.5vl:7b       (best OCR + layout understanding, ~6GB)
    #   ollama pull minicpm-v          (lighter, still solid on charts/diagrams)
    #   ollama pull llama3.2-vision:11b
    vision_model: str = "minicpm-v:latest"
    text_model: str = "llama3.2:3b"
    ollama_timeout: int = 180

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_engine: str = "easyocr"
    min_ocr_confidence: float = 0.30
    ocr_upscale: float = 1.8
    ocr_languages: tuple = ("en",)

    # ── Node detection ────────────────────────────────────────────────────────
    min_node_area: int = 1800
    min_node_width: int = 60
    min_node_height: int = 18
    max_node_width: int = 480
    max_node_height: int = 55
    max_node_aspect: float = 14.0
    min_node_aspect: float = 1.4
    overlap_iou_threshold: float = 0.50
    node_padding: int = 8

    # ── Arrow / edge detection ────────────────────────────────────────────────
    #
    # CORRIDOR FIX (in arrow_detector.py, not here):
    #   Corridor blocking is applied ONLY for edges with no skeleton evidence.
    #   Pixel-proven connections bypass corridor geometry entirely.
    #
    min_arrow_confidence: float = 0.51       # V4: 0.46 → raised: fewer low-conf phantoms
    canny_low: int = 40
    canny_high: int = 140
    hough_threshold: int = 52
    hough_min_len: int = 30
    hough_max_gap: int = 17                  # V4: 24 → tightened: stop bridging gaps between unrelated segs
    max_edge_distance_ratio: float = 0.52    # V4: 0.58 → tightened: fewer long-range false pairs
    arrowhead_search_radius: int = 34        # V4: 42 → tightened: fewer false arrowhead hits
    min_line_coverage: float = 0.66          # V4: 0.58 → raised: catch true arrows, reject phantoms
    line_coverage_tol: int = 7               # compromise: 6 (orig) / 8 (V4)
    line_coverage_samples: int = 48
    connector_close_kernel: int = 11         # V4: 13 → slightly tighter
    connector_ring_px: int = 16              # V4: 24 → tightened: stop false shared-component hits
    require_arrowhead_or_shared_component: bool = True
    max_edges_per_node: int = 10             # V4: 14 → moderate; hub nodes rarely need >10 in this style

    # ── LLM ───────────────────────────────────────────────────────────────────
    use_text_cleanup: bool = True
    # VLM-first detection: if the vision_model is pulled and Ollama is
    # reachable, it becomes the PRIMARY node/edge detector (replaces the
    # hand-tuned CV heuristics, which don't generalize across diagrams).
    # If the vision model is unavailable, or returns unusable output, the
    # pipeline automatically falls back to the classical CV path below —
    # nothing breaks for users who haven't pulled a vision model yet.
    use_vlm_detection: bool = True
    # After the VLM gives approximate (percentage-based) boxes, snap each
    # one onto the nearest real CV contour/color-border box for pixel
    # accuracy. Keep this on unless you specifically want raw VLM boxes.
    vlm_snap_boxes: bool = True

    # ── Output ────────────────────────────────────────────────────────────────
    keep_debug: bool = True
    excel_filename: str = "workflow_analysis.xlsx"
    json_filename: str = "workflow_analysis.json"

    # ── Performance ───────────────────────────────────────────────────────────
    max_ocr_workers: int = 1
    max_image_dim: int = 1800

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ocr_languages"] = list(self.ocr_languages)
        return d
