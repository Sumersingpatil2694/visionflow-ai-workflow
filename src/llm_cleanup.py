"""
Local Ollama LLM cleanup.

Two calls, both OPTIONAL and safe to fail:
  1) Label normalization (fix OCR noise, keep ids stable).
  2) Summary + observations.

Never used for geometry / node detection / edge inference.
Any failure -> pipeline continues with classical results.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import requests

from .prompts import build_label_cleanup_prompt, build_summary_prompt


def _ollama_generate(host: str, model: str, prompt: str, timeout: int = 120) -> Optional[str]:
    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        return r.json().get("response", "")
    except Exception:
        return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # try direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to find the first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def clean_labels(host: str, model: str, nodes: List[Dict[str, Any]], timeout: int = 120) -> List[Dict[str, Any]]:
    if not nodes:
        return nodes
    prompt = build_label_cleanup_prompt(nodes)
    raw = _ollama_generate(host, model, prompt, timeout=timeout)
    data = _extract_json(raw or "")
    if not data or "nodes" not in data:
        return nodes
    lookup = {item.get("id"): item.get("label") for item in data["nodes"] if isinstance(item, dict)}
    changed = 0
    for n in nodes:
        new_lbl = lookup.get(n["id"])
        if isinstance(new_lbl, str) and new_lbl.strip() and new_lbl.strip() != n["label"]:
            n["original_label"] = n["label"]
            n["label"] = new_lbl.strip()
            changed += 1
    if changed:
        # small confidence bump for successfully-cleaned labels
        for n in nodes:
            if "original_label" in n:
                n["confidence"] = round(min(0.99, n.get("confidence", 0.5) + 0.05), 3)
    return nodes


def summarize(host: str, model: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], stats: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    prompt = build_summary_prompt(nodes, edges, stats)
    raw = _ollama_generate(host, model, prompt, timeout=timeout)
    data = _extract_json(raw or "")
    if not data:
        return {"summary": "", "observations": []}
    return {
        "summary": (data.get("summary") or "").strip(),
        "observations": [o for o in (data.get("observations") or []) if isinstance(o, str)],
    }


def ollama_available(host: str, timeout: int = 4) -> bool:
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False
