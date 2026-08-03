"""
Main orchestrator V5.1.

run(image_path) -> dict:
    {
      "nodes":[...], "edges":[...],
      "start_states":[...], "end_states":[...],
      "statistics":{...}, "uncertainties":[...],
      "summary":"...", "observations":[...],
      "confidence": float,
      "debug": {...},                 # only if keep_debug
      "annotated_image_png": bytes,
      "source": {...}
    }

Everything here is derived purely from the uploaded image. There is no
"expected result" reconciliation step anywhere in this pipeline: nothing
external is allowed to relabel a node or inject an edge that wasn't found
in the pixels. If detection is wrong, fix node_detector.py / arrow_detector.py
directly rather than papering over it here.

V5.1 fix: nodes list now passed to infer_start_end so the graph_builder
can apply color-based start hints (green-style node = start state).
"""
from __future__ import annotations

from typing import Any, Dict, List
import io
import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig
from . import preprocess
from .ocr_engine import OCREngine
from .node_detector import detect_nodes
from .arrow_detector import detect_edges
from .graph_builder import (
    build_graph, infer_start_end, find_orphans, find_cycles,
    find_suspect_boundary_nodes, statistics,
)
from . import llm_cleanup
from . import vlm_extractor


class DiagramPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ocr = OCREngine(
            engine=config.ocr_engine,
            languages=config.ocr_languages,
            min_conf=config.min_ocr_confidence,
        )

    def run(self, image_path: str, progress_cb=None) -> Dict[str, Any]:
        def step(pct, msg):
            if progress_cb:
                progress_cb(pct, msg)

        step(2, "Loading image...")
        bgr = preprocess.load_and_normalize(image_path, max_dim=self.config.max_image_dim)

        step(10, "Enhancing image (denoise, CLAHE, sharpen, deskew)...")
        pre = preprocess.full_pipeline(bgr)

        detection_method = "classical_cv"
        ocr_items: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # ── VLM-first path ───────────────────────────────────────────────
        # A vision-LLM is far more robust to new/unseen diagram styles than
        # hand-tuned Hough/contour thresholds. Try it first; only fall back
        # to classical CV if Ollama or the vision model isn't available, or
        # if the model returns unusable output.
        if self.config.use_vlm_detection and vlm_extractor.vlm_available(
            self.config.ollama_host, self.config.vision_model
        ):
            step(20, f"Analyzing diagram with vision model ({self.config.vision_model})...")
            vlm_result = vlm_extractor.extract_diagram(
                self.config.ollama_host, self.config.vision_model, bgr,
                timeout=self.config.ollama_timeout,
            )
            if vlm_result and vlm_result["nodes"]:
                nodes = vlm_result["nodes"]
                edges = vlm_result["edges"]
                detection_method = "vlm"

                if self.config.vlm_snap_boxes:
                    step(35, "Snapping VLM boxes to pixel-accurate contours...")
                    nodes = vlm_extractor.snap_boxes_to_cv(nodes, bgr, pre["binary"], self.config)

                # A connected workflow needs at least (nodes - 1) edges. If the
                # VLM returned noticeably fewer than that, it likely missed some
                # (busy diagrams with crossing/converging arrows are hard for a
                # small local vision model) — run the classical tracer too and
                # merge the two edge sets instead of trusting the VLM alone.
                min_expected_edges = max(0, len(nodes) - 1)
                if len(edges) < min_expected_edges and len(nodes) >= 2:
                    step(60, "Few edges from VLM — supplementing with classical arrow tracer...")
                    classical_edges = detect_edges(pre["clahe"], nodes, self.config)
                    seen_pairs = {(e["source"], e["target"]) for e in edges}
                    for ce in classical_edges:
                        pair = (ce["source"], ce["target"])
                        if pair not in seen_pairs:
                            edges.append(ce)
                            seen_pairs.add(pair)
                    for i, e in enumerate(
                        sorted(edges, key=lambda x: (x["source"], x["target"])), start=1
                    ):
                        e["id"] = f"E{i}"

        # ── Classical CV fallback ────────────────────────────────────────
        if not nodes:
            step(25, f"Running OCR ({self.ocr.engine_name})...")
            ocr_items = self.ocr.read(pre["clahe"])

            step(45, "Detecting nodes (OCR + color borders + contour merge)...")
            nodes = detect_nodes(
                color_bgr=bgr,
                gray=pre["clahe"],
                binary=pre["binary"],
                ocr_items=ocr_items,
                ocr_engine=self.ocr,
                cfg=self.config,
            )

            step(68, "Detecting arrows (skeleton tracing + coverage scoring)...")
            edges = detect_edges(pre["clahe"], nodes, self.config)

        step(80, "Building directed graph...")
        g = build_graph(nodes, edges)

        # FIX V5.1: pass nodes so infer_start_end can use color hints
        starts, ends = infer_start_end(g, nodes)

        stats = statistics(nodes, edges, g)

        uncertainties: List[str] = []
        if detection_method == "classical_cv":
            if self.config.use_vlm_detection:
                uncertainties.append(
                    f"Vision model '{self.config.vision_model}' was unreachable or not pulled in "
                    f"Ollama — used the classical CV detector instead. Run "
                    f"`ollama pull {self.config.vision_model}` and retry for more robust detection."
                )
            if not self.ocr.available():
                uncertainties.append("OCR engine unavailable — labels may be missing.")
            if not ocr_items:
                uncertainties.append("No OCR text was confidently extracted from the full image; crop OCR / color-box fallback was used.")
        if not edges:
            uncertainties.append("No reliable connectors detected. Try higher-resolution screenshots.")
        orphans = find_orphans(g)
        if orphans:
            uncertainties.append(f"Isolated nodes with no incoming or outgoing edges: {', '.join(orphans)}.")
        cycles = find_cycles(g)
        if cycles:
            uncertainties.append(f"{len(cycles)} cycle(s) detected in the workflow.")
        suspects = find_suspect_boundary_nodes(g)
        if suspects:
            uncertainties.append(
                "Possible missing connector(s) near: " + ", ".join(suspects) +
                " — these look like a start/end only because no edge was detected to a nearby node."
            )
        summary_data = {"summary": "", "observations": []}
        if self.config.use_text_cleanup and llm_cleanup.ollama_available(self.config.ollama_host):
            step(86, "LLM label cleanup (Ollama)...")
            nodes = llm_cleanup.clean_labels(
                self.config.ollama_host, self.config.text_model, nodes,
                timeout=self.config.ollama_timeout,
            )
            step(92, "LLM summary generation...")
            summary_data = llm_cleanup.summarize(
                self.config.ollama_host, self.config.text_model, nodes, edges, stats,
                timeout=self.config.ollama_timeout,
            )
        else:
            summary_data["summary"] = self._fallback_summary(nodes, edges, stats)

        step(95, "Annotating debug image...")
        annotated_png = self._annotate(bgr, nodes, edges)

        step(100, "Done.")

        confidence = stats["overall_confidence"]
        result: Dict[str, Any] = {
            "nodes": nodes,
            "edges": edges,
            "start_states": starts,
            "end_states": ends,
            "statistics": stats,
            "uncertainties": uncertainties,
            "summary": summary_data.get("summary", ""),
            "observations": summary_data.get("observations", []),
            "confidence": confidence,
            "detection_method": detection_method,
            "source": {
                "ocr_engine": self.ocr.engine_name if detection_method == "classical_cv" else None,
                "vision_model": self.config.vision_model if detection_method == "vlm" else None,
                "text_model": self.config.text_model if self.config.use_text_cleanup else None,
                "skew_angle_deg": pre.get("skew_angle", 0.0),
            },
            "annotated_image_png": annotated_png,
        }
        if self.config.keep_debug:
            result["debug"] = {
                "ocr_item_count": len(ocr_items),
                "sample_ocr_items": ocr_items[:15],
                "config": self.config.to_dict(),
            }
        return result

    def _fallback_summary(self, nodes, edges, stats) -> str:
        if not nodes:
            return "No nodes detected. Consider using a higher-resolution screenshot."
        parts = [
            f"Detected {len(nodes)} node(s) and {len(edges)} directional connection(s).",
            f"Start state(s): {stats['start_count']}, end state(s): {stats['end_count']}.",
        ]
        if stats.get("cycle_count"):
            parts.append(f"Workflow contains {stats['cycle_count']} loop(s).")
        parts.append(f"Overall confidence: {int(stats['overall_confidence']*100)}%.")
        return " ".join(parts)

    def _annotate(self, bgr: np.ndarray, nodes, edges) -> bytes:
        img = bgr.copy()
        for n in nodes:
            x1, y1, x2, y2 = n["bbox"]
            conf = n.get("confidence", 0.0)
            style = n.get("style", "")
            if style == "green":
                color = (0, 190, 0)
            elif style == "blue":
                color = (255, 120, 0)
            else:
                color = (0, 180, 0) if conf >= 0.6 else (0, 165, 255) if conf >= 0.4 else (0, 0, 220)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{n['id']} {n.get('label','')[:28]}"
            cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        for e in edges:
            src = next((n for n in nodes if n["id"] == e["source"]), None)
            tgt = next((n for n in nodes if n["id"] == e["target"]), None)
            if not src or not tgt:
                continue
            p1 = tuple(src["center"])
            p2 = tuple(tgt["center"])
            color = (255, 90, 0) if e.get("has_arrowhead") else (200, 200, 200)
            cv2.arrowedLine(img, p1, p2, color, 2, tipLength=0.03)
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else b""
