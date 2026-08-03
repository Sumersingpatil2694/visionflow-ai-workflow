# VisionFlow AI 🧭

**VisionFlow AI** is a Streamlit app that turns an image of a workflow / flowchart diagram into structured, editable data — nodes, edges, and a clean summary — with no hardcoded outputs. Everything is derived directly from the uploaded pixels, using a hybrid **Vision-LLM + classical computer vision** pipeline.

> Engine version: **Workflow Diagram Analyzer V5.1**

## Table of Contents

- [Features](#features)
- [How It Works (Pipeline Architecture)](#how-it-works-pipeline-architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Optional: Enable LLM Cleanup](#optional-enable-llm-cleanup)
- [Usage](#usage)
- [Output Schema](#output-schema)
- [Configuration & Tuning](#configuration--tuning)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap Ideas](#roadmap-ideas)
- [License](#license)

## Features

- 📤 Upload a diagram image (flowchart, process map, BPMN-style diagram, etc.)
- 🧠 **VLM-first detection**: uses a local vision-language model (via Ollama) to read the diagram like a human would, falling back to classical CV (Hough transforms + contour analysis) when a vision model isn't available or returns too few edges
- 🔎 Automatic detection of **nodes** (shapes/boxes) and **edges** (arrows/connectors), including start/end state inference (with color-based hints, e.g. green = start)
- 🎯 **Pixel-accurate box snapping**: VLM-proposed node boxes are snapped to real contour boundaries for precision
- 🔤 OCR-based text extraction from nodes (EasyOCR), used both standalone and to clean up VLM labels
- 🖼️ Annotated overlay preview showing detected nodes, arrows, and directionality
- 📊 Live statistics: node count, edge count, start/end points, orphan nodes, cycles, and warnings/uncertainties
- 🧠 Optional local LLM cleanup (via [Ollama](https://ollama.com)) for label normalization and a natural-language summary — fully optional; if Ollama is unreachable, the pipeline silently continues with classical results
- ⬇️ Export results as **Excel**, **JSON**, or the **annotated image**
- ⚡ Real-time progress bar with step-by-step status while the pipeline runs
- 🧩 No hardcoded/expected outputs anywhere — every result is derived purely from the uploaded pixels; nothing external "corrects" a detection after the fact

## How It Works (Pipeline Architecture)

The pipeline (`src/pipeline.py`) runs as a sequence of stages, each reporting progress back to the Streamlit UI:

1. **Load & normalize** — image is loaded and resized to a max dimension (`preprocess.load_and_normalize`)
2. **Enhance** — denoise, CLAHE contrast enhancement, sharpening, and deskew (`preprocess.full_pipeline`)
3. **Detection (VLM-first, CV-fallback)**:
   - If `use_vlm_detection` is enabled and a vision model is reachable on Ollama, the diagram is sent to the vision model (`vlm_extractor.extract_diagram`) to propose nodes + edges directly
   - Proposed node boxes are optionally **snapped** to pixel-accurate contours (`vlm_extractor.snap_boxes_to_cv`)
   - If the VLM returns noticeably fewer edges than expected for a connected graph (`< nodes - 1`), the classical arrow tracer runs as well and its edges are **merged in** — busy diagrams with crossing/converging arrows are hard for small local vision models to fully resolve alone
   - If no vision model is available, the pipeline falls back entirely to classical CV: `node_detector.py` (contour/shape based) + `arrow_detector.py` (Hough-line based, with a "corridor fix" to avoid phantom edges)
4. **OCR** — EasyOCR extracts text from each detected node region (`ocr_engine.py`)
5. **Graph building** — nodes + edges are assembled into a graph (`graph_builder.build_graph`), with:
   - Start/end state inference (`infer_start_end`, using color hints)
   - Orphan node detection (`find_orphans`)
   - Cycle detection (`find_cycles`)
   - Boundary/suspect node flagging (`find_suspect_boundary_nodes`)
   - Aggregate statistics (`statistics`)
6. **Optional LLM cleanup** (`llm_cleanup.py`) — two independent, safe-to-fail calls to a local Ollama text model:
   - Label normalization (fixes OCR noise, keeps node IDs stable)
   - Natural-language summary + observations
   - Never used for geometry, node detection, or edge inference — only cosmetic cleanup on top of already-detected structure
7. **Rendering & export** — an annotated overlay image is generated, and results can be exported as Excel (`excel_export.py`), JSON, or the annotated PNG

## Project Structure

```
VisionFlow AI/
├── app.py                   # Streamlit UI: upload, progress bar, previews, exports
├── requirements.txt         # Python dependencies
├── run_app.bat               # One-click venv setup + launch (Windows)
├── Gemini_1.png              # Sample/reference asset
├── src/
│   ├── __init__.py
│   ├── config.py             # PipelineConfig dataclass — all tunable thresholds
│   ├── pipeline.py           # DiagramPipeline orchestrator (see architecture above)
│   ├── preprocess.py         # Image load, denoise, CLAHE, sharpen, deskew
│   ├── node_detector.py      # Classical CV node/shape detection
│   ├── arrow_detector.py     # Classical CV arrow/edge detection (Hough-based)
│   ├── graph_builder.py      # Graph assembly, start/end inference, cycles, stats
│   ├── ocr_engine.py         # EasyOCR wrapper for text extraction
│   ├── vlm_extractor.py      # Vision-LLM (Ollama) node/edge extraction + box snapping
│   ├── llm_cleanup.py        # Optional local LLM label cleanup + summary generation
│   ├── prompts.py            # Prompt templates for VLM/LLM calls
│   └── excel_export.py       # Converts pipeline results to an Excel workbook
└── tests/
    └── test_arrow_detector.py # Unit tests for arrow/edge detection
```

## Project Structure

```
VisionFlow AI/
├── app.py                  # Streamlit UI entry point
├── requirements.txt        # Python dependencies
├── run_app.bat             # One-click setup + launch (Windows)
├── src/
│   ├── config.py           # Central pipeline configuration/tuning
│   ├── pipeline.py          # Orchestrates the full diagram-analysis pipeline
│   ├── preprocess.py        # Image preprocessing
│   ├── node_detector.py     # Node/shape detection
│   ├── arrow_detector.py    # Arrow/edge detection
│   ├── graph_builder.py     # Builds the node-edge graph
│   ├── ocr_engine.py        # OCR text extraction (EasyOCR)
│   ├── vlm_extractor.py     # Vision-language model extraction helpers
│   ├── llm_cleanup.py       # Optional local LLM (Ollama) label cleanup + summary
│   ├── prompts.py           # LLM prompt templates
│   └── excel_export.py      # Excel export logic
└── tests/
    └── test_arrow_detector.py
```

## Setup

### Requirements
- Python 3.11+ (also tested compatible with 3.14 bytecode present)
- OS: Windows, macOS, or Linux
- (Optional but recommended) [Ollama](https://ollama.com) running locally, for VLM-based detection and LLM-based label cleanup/summaries — the app works without it via the classical CV fallback

Core dependencies (see `requirements.txt` for exact versions):

| Package         | Purpose                                  |
|-----------------|-------------------------------------------|
| `streamlit`     | Web UI                                    |
| `opencv-python` | Classical CV — preprocessing, contours, Hough line detection |
| `numpy`         | Array/image math                          |
| `pillow`        | Image I/O                                 |
| `requests`      | HTTP calls to local Ollama API            |
| `easyocr`       | OCR text extraction                       |
| `torch` / `torchvision` | Backend for EasyOCR                |
| `openpyxl`      | Excel export                              |
| `networkx`      | Graph operations (cycles, orphans)        |
| `scikit-image`  | Additional image processing utilities     |

### Install & Run (Windows)

Just double-click / run:

```bat
run_app.bat
```

This script will:
1. Create a `.venv` virtual environment (if it doesn't already exist)
2. Activate it
3. Upgrade `pip`
4. Install everything in `requirements.txt`
5. Launch `streamlit run app.py`

### Install & Run (manual / macOS / Linux)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Optional: Enable VLM Detection + LLM Cleanup

VisionFlow AI can use a local vision-language model (through Ollama) for much more robust detection on diagram styles it hasn't been hand-tuned for, plus an optional text model for cleanup/summaries.

1. Install [Ollama](https://ollama.com)
2. Pull a vision model (choose based on your hardware):

```bash
ollama pull minicpm-v:latest        # default in config.py — lighter, solid on charts/diagrams
ollama pull qwen2.5vl:7b            # stronger OCR + layout understanding (~6GB)
ollama pull llama3.2-vision:11b     # alternative larger vision model
```

3. Pull a lightweight text model for cleanup/summaries:

```bash
ollama pull llama3.2:3b
```

4. Make sure Ollama is running (`ollama serve`, or the desktop app) — it should be reachable at `http://localhost:11434` by default (configurable in `src/config.py`)

**Note:** `moondream:1.8b` is intentionally avoided as the default vision model — it's too small for reliable structured diagram parsing.

If Ollama isn't running or a model isn't pulled, the app automatically and silently falls back to:
- Classical CV (Hough/contour-based) for node & edge detection
- No LLM cleanup — raw OCR labels and rule-based summary only

## Usage

1. Launch the app (see [Setup](#setup))
2. Upload a diagram image in the sidebar (PNG/JPG of a flowchart, workflow, or process diagram)
3. Watch the real-time progress bar as the pipeline runs through preprocessing → detection → OCR → graph building → (optional) LLM cleanup
4. Review:
   - The **annotated overlay** (detected nodes + arrows drawn on your image)
   - **Statistics**: node/edge counts, start/end states, orphan nodes, cycles
   - **Warnings/uncertainties** flagged by the pipeline
   - The **summary** and observations (if LLM cleanup is enabled)
5. Download results as:
   - 📊 **Excel** (`.xlsx`) — structured node/edge tables
   - 🧾 **JSON** — full raw pipeline output
   - 🖼️ **Annotated image** (`.png`)

## Output Schema

The pipeline's `run()` function (`src/pipeline.py`) returns a dictionary shaped like:

```json
{
  "nodes": [ { "id": "...", "label": "...", "bbox": [x, y, w, h], "type": "..." } ],
  "edges": [ { "source": "...", "target": "...", "confidence": 0.0 } ],
  "start_states": ["..."],
  "end_states": ["..."],
  "statistics": { "node_count": 0, "edge_count": 0, "orphans": 0, "cycles": 0 },
  "uncertainties": ["..."],
  "summary": "...",
  "observations": ["..."],
  "confidence": 0.0,
  "annotated_image_png": "<bytes>",
  "source": { "detection_method": "vlm | classical_cv", "..." : "..." }
}
```

This is the same structure exported to JSON via the download button, and flattened into rows for the Excel export.

## Configuration & Tuning

All tunable parameters live in `src/config.py` as a single `PipelineConfig` dataclass. Notable groups:

- **Ollama**: `ollama_host`, `vision_model`, `text_model`, `ollama_timeout`
- **OCR**: `ocr_engine`, `min_ocr_confidence`, `ocr_upscale`, `ocr_languages`
- **Node detection**: `min_node_area`, `min/max_node_width/height`, `max_node_aspect`, `overlap_iou_threshold`, `node_padding`
- **Arrow/edge detection**: Hough-transform gap tolerances, arrowhead search radius, minimum arrow confidence, max edges per node, tortuosity cap, junction-hop turn angle

The config file itself documents a **V5.0 calibration history** — a record of threshold changes made to fix phantom-edge issues on a reference test diagram (target: 10 nodes, 13 edges, 1 start, 2 ends). If you're tuning for a new diagram style, adjust these values incrementally and re-test rather than changing many at once.

## Testing

```bash
pytest tests/
```

Currently covers `tests/test_arrow_detector.py`. Contributions of additional test coverage (node detection, graph building, VLM extraction with mocked Ollama responses) are welcome.

## Troubleshooting

| Issue | Likely cause / fix |
|---|---|
| App falls back to classical CV even though Ollama is installed | Check Ollama is running (`ollama serve`) and reachable at the host in `config.py` (`http://localhost:11434` by default); confirm the model was pulled with `ollama list` |
| Detection is noisy / phantom edges appear | Tune `arrow_detector.py`-related thresholds in `config.py` (see [Configuration & Tuning](#configuration--tuning)) |
| OCR misses text or garbles labels | Increase `ocr_upscale` or lower `min_ocr_confidence` in `config.py`; ensure the source image has good resolution/contrast |
| `torch`/`easyocr` install is slow or large | Expected — these pull GPU/CPU-heavible ML backends; use a CPU-only torch wheel if you don't have a GPU |
| `run_app.bat` does nothing on Mac/Linux | It's Windows-only; use the manual setup steps instead |

## Roadmap Ideas

- [ ] Support for additional diagram export formats (Mermaid, DrawIO/XML)
- [ ] Batch processing of multiple diagrams
- [ ] Confidence-based manual correction UI for nodes/edges
- [ ] Docker packaging for one-command setup
- [ ] CI pipeline running `pytest` on push

## Contributing

Issues and pull requests are welcome. Please open an issue describing the diagram style or bug before submitting large changes to the detection thresholds in `config.py`, since they were tuned carefully against a reference test case.
