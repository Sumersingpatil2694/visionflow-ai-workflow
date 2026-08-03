"""
VisionFlow AI — prompts.py
Prompt templates for the local Ollama models.

Two use cases now live here:
  1) Vision-LLM diagram extraction (vlm_extractor.py) — geometry + labels,
     used as the PRIMARY detector when a vision model is available.
  2) Text-LLM cleanup (llm_cleanup.py) — label normalization + summary only,
     never used for geometry.
"""
import json
from typing import Any, Dict, List


def build_vlm_extraction_prompt() -> str:
    """Strict JSON-only prompt for the vision model (MiniCPM-V / Qwen2.5-VL /
    Llama-Vision, via Ollama). No prose, no markdown — just the schema below,
    with bounding boxes expressed as percentages of image width/height so the
    prompt stays resolution-independent."""
    return """You are the vision-extraction engine for VisionFlow AI, a precise computer-vision
tool for workflow / flowchart diagrams.

Look at the image and identify every node (box/shape with text) and every
directed connector (arrow) between nodes.

Return ONLY valid JSON. No explanation. No markdown fences. No comments.
No trailing text before or after the JSON object.

Respond with EXACTLY this schema (the numbers below are placeholders showing the
expected TYPE only — never copy them literally, always compute your own real value
for each node/edge):
{"nodes":[{"label":"<text>","bbox":[x_percent,y_percent,w_percent,h_percent],"shape":"<shape>","style":"<color>","confidence":<0.0-1.0>}],"edges":[{"from":"<label>","to":"<label>","confidence":<0.0-1.0>}]}

Rules:
- "bbox" is [x, y, width, height] as PERCENTAGES of the full image (0-100 floats), where x,y is the top-left corner of the node's box.
- "label" is the visible text inside the node, verbatim.
- "shape" is one of: rectangle, rounded_rectangle, diamond, ellipse, circle, parallelogram.
- "style" is the dominant fill/border color of the node in plain words (e.g. green, blue, red, yellow, gray, white, none).
- "confidence" is a REAL number reflecting how certain YOU are, from 0 to 1 — never leave it at 0 and never copy an example value.
  A node/edge you can read and see clearly should score 0.8-0.99. Only use a low value (below 0.3) when the
  text or connector is genuinely blurry, cut off, or ambiguous. A confidence of exactly 0.0 is almost never correct
  and will be treated as an error.
- "from" and "to" in "edges" MUST use the exact "label" text of the source and target nodes (not ids).
- Only include an edge if you can see an arrow/line connecting the two nodes, with the arrowhead (if any) pointing from "from" to "to".
- Do not invent nodes or edges that are not visibly present in the image.
- If you are unsure about a node or edge, still include it but lower its confidence value accordingly (do not set it to 0).

Return ONLY the JSON object described above."""


def build_label_cleanup_prompt(nodes: List[Dict[str, Any]]) -> str:
    payload = [{"id": n["id"], "label": n["label"]} for n in nodes]
    return f"""You are a text-normalization assistant for workflow diagrams.

Below is a list of node labels extracted by OCR from a real workflow screenshot.
The labels may have OCR noise (broken casing, glued words, stray punctuation).
Your job:
  - Fix obvious OCR mistakes ONLY when you are highly confident.
  - Preserve real domain words (e.g. "CDD", "UAR", "RFI") exactly as-is.
  - DO NOT invent new labels. DO NOT remove or add items.
  - Keep the SAME number of items with the SAME ids.

Return ONLY valid JSON of the shape:
{{"nodes":[{{"id":"N1","label":"<clean label>"}}, ...]}}

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def build_summary_prompt(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
    node_payload = [{"id": n["id"], "label": n["label"]} for n in nodes]
    edge_payload = [{"from": e["source"], "to": e["target"]} for e in edges]
    return f"""You are a workflow analyst. Given the extracted nodes, edges and statistics
of a real workflow diagram, write:
  1) A concise 2-4 sentence summary of what the workflow does.
  2) 2-6 bullet-point observations (branches, loops, terminal states).

Do NOT invent nodes or edges that are not in the input.
Return ONLY valid JSON:
{{"summary":"...", "observations":["...","..."]}}

Nodes:
{json.dumps(node_payload, ensure_ascii=False)}
Edges:
{json.dumps(edge_payload, ensure_ascii=False)}
Stats:
{json.dumps(stats, ensure_ascii=False)}
"""
