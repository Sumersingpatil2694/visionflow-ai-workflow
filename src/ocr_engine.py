"""
OCR engine wrapper. Runs on the whole image AND per-node crops.
Handles engine fallback (easyocr <-> paddleocr -> tesseract CLI).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
import os
import re
import shutil
import subprocess
import tempfile

try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None


class OCREngine:
    def __init__(self, engine: str = "easyocr", languages: Tuple[str, ...] = ("en",), min_conf: float = 0.3):
        self.engine_name = engine
        self.languages = list(languages)
        self.min_conf = min_conf
        self.reader = None
        self._paddle = None
        self._tesseract_bin = shutil.which("tesseract")
        self._init_engine()

    def _init_engine(self) -> None:
        wanted = self.engine_name.lower()
        if wanted in ("easyocr", "auto") and easyocr is not None:
            try:
                self.reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
                self.engine_name = "easyocr"
                return
            except Exception:
                self.reader = None
        if wanted in ("paddleocr", "auto") and PaddleOCR is not None:
            try:
                self._paddle = PaddleOCR(use_angle_cls=True, lang=self.languages[0], show_log=False)
                self.engine_name = "paddleocr"
                return
            except Exception:
                self._paddle = None
        if self._tesseract_bin:
            self.engine_name = "tesseract"
            return
        self.engine_name = "unavailable"

    def available(self) -> bool:
        return self.reader is not None or self._paddle is not None or bool(self._tesseract_bin)

    def read(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            return []
        if self.reader is not None:
            return self._read_easyocr(image)
        if self._paddle is not None:
            return self._read_paddle(image)
        if self._tesseract_bin:
            return self._read_tesseract(image)
        return []

    def _clean_text(self, text: str) -> str:
        text = (text or "").replace("\u2014", "-").replace("\u2013", "-")
        text = re.sub(r"^[^A-Za-z0-9]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Strip small leading OCR garbage like "w?" or "v!"
        text = re.sub(r"^(?:[A-Za-z]?[^A-Za-z0-9\s-]{1,3}\s+)+", "", text).strip()
        return text

    def _read_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            results = self.reader.readtext(image)
        except Exception:
            return []
        items: List[Dict[str, Any]] = []
        for box, text, conf in results:
            if conf is None or conf < self.min_conf:
                continue
            text = self._clean_text(text)
            if not text:
                continue
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            items.append({
                "text": text,
                "confidence": round(float(conf), 3),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "center": [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))],
            })
        return items

    def _read_paddle(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            result = self._paddle.ocr(image, cls=True)
        except Exception:
            return []
        items: List[Dict[str, Any]] = []
        if not result:
            return items
        for line in result[0] or []:
            box = line[0]
            text, conf = line[1]
            if conf is None or conf < self.min_conf:
                continue
            text = self._clean_text(text)
            if not text:
                continue
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            items.append({
                "text": text,
                "confidence": round(float(conf), 3),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "center": [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))],
            })
        return items

    def _prep_for_tesseract(self, image: np.ndarray, line_mode: bool = False) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        if line_mode:
            return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)

    def _run_tesseract(self, image: np.ndarray, psm: int, tsv: bool = False) -> str:
        if not self._tesseract_bin:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, image)
            cmd = [self._tesseract_bin, tmp_path, "stdout", "--psm", str(psm)]
            if tsv:
                cmd.append("tsv")
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            return out.stdout or ""
        except Exception:
            return ""
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def _read_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        prepared = self._prep_for_tesseract(image, line_mode=False)
        raw = self._run_tesseract(prepared, psm=11, tsv=True)
        if raw.count("\n") <= 1:
            raw = self._run_tesseract(prepared, psm=6, tsv=True)
        lines = raw.splitlines()
        items: List[Dict[str, Any]] = []
        for ln in lines[1:]:
            parts = ln.split("\t")
            if len(parts) < 12:
                continue
            left, top, width, height, conf, text = parts[6], parts[7], parts[8], parts[9], parts[10], parts[11]
            try:
                conf_f = float(conf) / 100.0 if float(conf) > 1.0 else float(conf)
            except Exception:
                conf_f = -1.0
            text = self._clean_text(text)
            if conf_f < self.min_conf or not text:
                continue
            try:
                x = int(float(left)); y = int(float(top)); w = int(float(width)); h = int(float(height))
            except Exception:
                continue
            # Scale back because we 2x-upscaled before OCR.
            bbox = [x // 2, y // 2, (x + w) // 2, (y + h) // 2]
            items.append({
                "text": text,
                "confidence": round(float(conf_f), 3),
                "bbox": bbox,
                "center": [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)],
            })
        return items

    def read_crop(self, image: np.ndarray, bbox: List[int], upscale: float = 1.5, pad: int = 4) -> Optional[str]:
        """OCR on a padded, upscaled crop. Returns joined text or None."""
        if not self.available():
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        crop = image[y1:y2, x1:x2]
        if upscale and upscale > 1.0:
            crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

        # Native engines first
        if self.reader is not None or self._paddle is not None:
            items = self.read(crop)
            if items:
                items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
                text = self._clean_text(" ".join(it["text"] for it in items).strip())
                return text or None

        # Tesseract line OCR fallback
        if self._tesseract_bin:
            prep = self._prep_for_tesseract(crop, line_mode=True)
            raw = self._run_tesseract(prep, psm=7, tsv=False)
            text = self._clean_text(" ".join(raw.split()))
            return text or None
        return None
