"""
VisionFlow AI — Streamlit UI (Workflow Diagram Analyzer engine V5.1)

- Upload a diagram
- Real-time progress bar
- Preview + annotated overlay
- Statistics + summary + warnings
- Download: Excel, JSON, Annotated image
- No hardcoded outputs, everything is derived from the uploaded pixels.
"""
import json
import tempfile
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from src.pipeline import DiagramPipeline
from src.config import PipelineConfig
from src.excel_export import export_to_bytes as excel_export_bytes
from src.llm_cleanup import ollama_available


def render_table(rows: list) -> None:
    """Render a list of dicts as a plain HTML table."""
    if not rows:
        st.caption("No data.")
        return
    cols = list(rows[0].keys())
    thead = "".join(f"<th>{c}</th>" for c in cols)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{'' if r.get(c) is None else r.get(c)}</td>" for c in cols)
        trs.append(f"<tr>{tds}</tr>")
    html = f"""
    <div style="max-height:420px; overflow:auto; border:1px solid rgba(148,163,184,0.25); border-radius:10px;">
    <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
      <thead style="position:sticky; top:0; background:#1f2937; color:#fff;">
        <tr>{thead}</tr>
      </thead>
      <tbody>
        {''.join(trs)}
      </tbody>
    </table>
    </div>
    <style>
      table td, table th {{ padding:6px 10px; border-bottom:1px solid rgba(148,163,184,0.18); text-align:left; }}
      table tbody tr:nth-child(odd) {{ background: rgba(255,255,255,0.03); }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)


st.set_page_config(
    page_title="VisionFlow AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #111827 40%, #0f172a 100%);
        color: #e5e7eb;
    }
    .main-card {
        background: rgba(17, 24, 39, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        box-shadow: 0 20px 45px rgba(0,0,0,0.25);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.9));
        border: 1px solid rgba(96,165,250,0.22);
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
    }
    .hero {
        padding: 1.3rem 1.4rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(37,99,235,0.32), rgba(16,185,129,0.18));
        border: 1px solid rgba(96,165,250,0.24);
        margin-bottom: 1rem;
    }
    .small-note { color: #cbd5e1; font-size: 0.93rem; }
    .pill {
        display:inline-block; padding:2px 10px; border-radius:999px;
        background: rgba(59,130,246,0.18); border:1px solid rgba(96,165,250,0.35);
        color:#dbeafe; font-size:0.82rem; margin-right:6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None
if "excel_bytes" not in st.session_state:
    st.session_state.excel_bytes = None
if "annotated_bytes" not in st.session_state:
    st.session_state.annotated_bytes = None

st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0; font-size:2rem;">🧭 VisionFlow AI <span class="pill">V5.1</span></h1>
        <p style="margin:0.45rem 0 0 0; color:#dbeafe; font-size:1.03rem;">
            Local-first pipeline — <b>OpenCV + OCR + local Ollama</b>. Accurate edge detection. Excel + JSON + annotated PNG downloads.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("100% local. Ollama is optional but improves labels & summary.")

    _defaults = PipelineConfig()  # single source of truth for slider defaults

    ollama_host = st.text_input("Ollama host", value=_defaults.ollama_host)

    st.divider()
    st.subheader("🧠 Vision detection")
    use_vlm = st.checkbox(
        "Use vision LLM as primary detector (recommended)",
        value=_defaults.use_vlm_detection,
        help="Reads the whole diagram with a local vision model instead of hand-tuned "
             "contour/Hough thresholds. Falls back to classical CV automatically if the "
             "model isn't pulled or Ollama isn't reachable."
    )
    vision_model = st.text_input(
        "Vision LLM (Ollama)", value=_defaults.vision_model,
        help="Needs to be pulled first: `ollama pull qwen2.5vl:7b` (best accuracy, ~6GB) "
             "or `ollama pull minicpm-v` (lighter). moondream is too small for reliable results."
    )
    vlm_snap = st.checkbox(
        "Snap vision-model boxes to pixel-accurate contours", value=_defaults.vlm_snap_boxes,
        help="The vision model gives approximate box positions; this tightens them onto the "
             "real detected borders when a good match is found."
    )
    text_model = st.text_input("Text LLM (Ollama)", value=_defaults.text_model,
                               help="Light model (~1GB RAM) used for label cleanup + summary.")

    st.divider()
    st.subheader("Classical CV fallback")
    st.caption("Only used if the vision model above is unavailable.")
    min_conf = st.slider(
        "Minimum OCR confidence", 0.0, 1.0, _defaults.min_ocr_confidence, 0.05,
        help="Lower = more text detected but may include noise."
    )
    arrow_threshold = st.slider(
        "Arrow confidence threshold", 0.0, 1.0, _defaults.min_arrow_confidence, 0.02,
        help="Higher = fewer false arrows. Default 0.46 balances recall and precision."
    )
    strict_edges = st.checkbox(
        "Strict edge gate (recommended)",
        value=_defaults.require_arrowhead_or_shared_component,
        help="Only keep arrows with a traced stroke OR clear arrowhead. Prevents over-detection."
    )
    max_deg = st.slider(
        "Max edges per node", 1, 20, _defaults.max_edges_per_node,
        help="Cap on combined in+out degree per node. Hub nodes (e.g. 'Review in Progress') "
             "can legitimately need 8-10 connections. Raise this if real edges are being dropped."
    )
    use_text_llm = st.checkbox("Use local LLM for label cleanup + summary", value=True)
    enable_debug = st.checkbox("Show debug artifacts", value=True)

    st.divider()
    if st.button("🔌 Check Ollama connection"):
        ok = ollama_available(ollama_host)
        (st.success if ok else st.warning)(f"Ollama {'reachable' if ok else 'NOT reachable'} at {ollama_host}")

    st.divider()
    st.subheader("💡 Tips for best results")
    st.markdown(
        """
        - Use **screenshots**, not photos of a screen.
        - Higher resolution = more accurate detection.
        - Pull a real vision model first — `ollama pull qwen2.5vl:7b` — moondream is too
          weak for reliable structured extraction.
        - The vision LLM reads the whole diagram at once; classical CV settings below only
          matter when it falls back or when filling in missing arrows.
        - LLM cleanup only normalizes labels — never invents or removes nodes/edges.
        - Green-highlighted nodes are automatically detected as start states.
        - If edges are missing, try lowering **Arrow confidence threshold**.
        - If too many false edges appear, enable **Strict edge gate**.
        """
    )

# ---------- main columns
left, right = st.columns([1.05, 1.15], gap="large")

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload diagram")
    uploaded = st.file_uploader(
        "Supported: PNG, JPG, JPEG, WEBP, BMP",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded diagram", use_container_width=True)

        if st.button("🔍 Analyze diagram", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            config = PipelineConfig(
                ollama_host=ollama_host.strip(),
                vision_model=vision_model.strip(),
                text_model=text_model.strip(),
                min_ocr_confidence=float(min_conf),
                min_arrow_confidence=float(arrow_threshold),
                use_text_cleanup=bool(use_text_llm),
                use_vlm_detection=bool(use_vlm),
                vlm_snap_boxes=bool(vlm_snap),
                keep_debug=bool(enable_debug),
                require_arrowhead_or_shared_component=bool(strict_edges),
                max_edges_per_node=int(max_deg),
            )

            pipeline = DiagramPipeline(config)

            progress = st.progress(0, text="Starting...")
            def cb(pct, msg):
                progress.progress(min(100, int(pct)), text=msg)

            t0 = time.time()
            try:
                result = pipeline.run(temp_path, progress_cb=cb)
                elapsed = time.time() - t0
                st.session_state.result = result
                st.session_state.excel_bytes = excel_export_bytes(result)
                st.session_state.annotated_bytes = result.get("annotated_image_png", b"")
                method = result.get("detection_method", "classical_cv")
                method_label = "vision LLM" if method == "vlm" else "classical CV (fallback)"
                st.success(
                    f"Analysis complete in {elapsed:.1f}s • detector: {method_label} • "
                    f"overall confidence {int(result['confidence']*100)}%"
                )
                if method == "classical_cv" and use_vlm:
                    st.info(
                        f"Vision model **{vision_model}** wasn't reachable, so classical CV ran instead. "
                        f"Pull it with `ollama pull {vision_model}` and click **Check Ollama connection** "
                        f"to verify, then re-analyze."
                    )
            except Exception as exc:
                st.exception(exc)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("🧾 Structured output")
    result = st.session_state.result

    if result is None:
        st.info("Upload a workflow image and click **Analyze diagram**.")
    else:
        stats = result.get("statistics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Nodes",      stats.get("node_count", 0))
        c2.metric("Edges",      stats.get("edge_count", 0))
        c3.metric("Starts",     stats.get("start_count", 0))
        c4.metric("Ends",       stats.get("end_count", 0))
        c5.metric("Confidence", f"{int(result.get('confidence', 0.0)*100)}%")

        tab_labels = ["🖼️ Annotated", "📋 Summary", "📊 Data", "⚠️ Warnings", "⬇️ Download"]
        tab1, tab2, tab3, tab4, tab6 = st.tabs(tab_labels)

        with tab1:
            if st.session_state.annotated_bytes:
                st.image(st.session_state.annotated_bytes, caption="Detected nodes & connectors", use_container_width=True)
            else:
                st.write("Annotated image unavailable.")

        with tab2:
            st.markdown("### Extracted summary")
            st.write(result.get("summary") or "_No summary available._")
            obs = result.get("observations", [])
            if obs:
                st.markdown("### Observations")
                for o in obs:
                    st.write(f"- {o}")
            st.markdown("### Start / End states")
            # Resolve node IDs to labels for display
            id2lbl = {n["id"]: n["label"] for n in result.get("nodes", [])}
            starts_display = ", ".join(
                f"{sid} · {id2lbl.get(sid, '')}" for sid in result.get("start_states", [])
            ) or "—"
            ends_display = ", ".join(
                f"{eid} · {id2lbl.get(eid, '')}" for eid in result.get("end_states", [])
            ) or "—"
            st.write(f"**Start:** {starts_display}")
            st.write(f"**End:** {ends_display}")

        with tab3:
            st.markdown("#### Nodes")
            render_table(
                [
                    {
                        "ID": n["id"],
                        "Label": n["label"],
                        "Shape": n["shape"],
                        "Style": n.get("style", ""),
                        "Conf.": n["confidence"],
                        "Source": n.get("source", ""),
                        "Center": f"({n['center'][0]}, {n['center'][1]})",
                    }
                    for n in result.get("nodes", [])
                ]
            )
            st.markdown("#### Connections")
            id2lbl = {n["id"]: n["label"] for n in result.get("nodes", [])}
            render_table(
                [
                    {
                        "ID": e["id"],
                        "From": f"{e['source']} · {id2lbl.get(e['source'], '')}",
                        "To": f"{e['target']} · {id2lbl.get(e['target'], '')}",
                        "Direction": e.get("direction", ""),
                        "Routing": e.get("routing", ""),
                        "Conf.": e.get("confidence", 0.0),
                        "Arrowhead": "Yes" if e.get("has_arrowhead") else "No",
                    }
                    for e in result.get("edges", [])
                ]
            )
            st.markdown("#### Statistics")
            st.json(stats)

        with tab4:
            uncs = result.get("uncertainties", [])
            if not uncs:
                st.success("No warnings — every element passed the confidence checks.")
            else:
                for u in uncs:
                    st.warning(u)
            low_nodes = [n for n in result.get("nodes", []) if n.get("confidence", 1.0) < 0.45]
            low_edges = [e for e in result.get("edges", []) if e.get("confidence", 1.0) < 0.40]
            if low_nodes:
                st.markdown("**Low-confidence nodes:**")
                for n in low_nodes:
                    st.write(f"- {n['id']} — '{n['label']}' (conf {n['confidence']})")
            if low_edges:
                st.markdown("**Low-confidence edges:**")
                for e in low_edges:
                    st.write(f"- {e['id']} {e['source']} → {e['target']} (conf {e['confidence']})")

        with tab6:
            st.markdown("Download the analysis results below:")
            colA, colB, colC = st.columns(3)
            if st.session_state.excel_bytes:
                colA.download_button(
                    "⬇️ Excel (.xlsx)",
                    data=st.session_state.excel_bytes,
                    file_name="workflow_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            json_payload = {k: v for k, v in result.items() if k != "annotated_image_png"}
            json_bytes = json.dumps(json_payload, indent=2, ensure_ascii=False).encode("utf-8")
            colB.download_button(
                "⬇️ JSON",
                data=json_bytes,
                file_name="workflow_analysis.json",
                mime="application/json",
                use_container_width=True,
            )
            if st.session_state.annotated_bytes:
                colC.download_button(
                    "⬇️ Annotated PNG",
                    data=st.session_state.annotated_bytes,
                    file_name="workflow_annotated.png",
                    mime="image/png",
                    use_container_width=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p class="small-note">V5.1 — modular pipeline (preprocess → OCR → nodes → arrows → graph → LLM cleanup). '
    'Corridor blocking now skeleton-aware. Green node = start hint. No hardcoded outputs.</p>',
    unsafe_allow_html=True,
)
