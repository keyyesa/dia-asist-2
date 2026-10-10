"""🧪 Lab Report page: extract and classify diabetes-related lab values."""
import os

import streamlit as st

import config
from src import documents, orchestrator
from views import _common as C
from views.chat import report_table, level_icon, GOOD, BAD

SAMPLE_DIR = os.path.join(config.BASE_DIR, "sample_lab_reports")


def _sample_files() -> list:
    try:
        return sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".txt", ".pdf", ".docx")))
    except OSError:
        return []


def _load_sample() -> None:
    name = st.session_state.get("sample_choice")
    try:
        with open(os.path.join(SAMPLE_DIR, name), "rb") as f:
            C.set_report(documents.extract_text(name, f.read(), page_markers=False), name)
    except (OSError, ValueError, TypeError):
        C.flash("error", "The sample report could not be loaded.")


def _text_changed() -> None:
    st.session_state.report_text = st.session_state.report_text_widget
    st.session_state.analysis = None


def render() -> None:
    C.page_header("🧪", "Lab Report Analysis",
                  "Upload a report; values are extracted and classified with fixed reference ranges.")
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Report")
        uploaded = st.file_uploader("Upload PDF, TXT or DOCX", type=["pdf", "txt", "docx"],
                                    key=f"lab_uploader_{st.session_state.uploader_n}")
        if uploaded:
            C.load_report_from_upload(uploaded)
        samples = _sample_files()
        if samples:
            st.selectbox("Sample reports", samples, key="sample_choice")
            st.button("📄 Load sample report", on_click=_load_sample, **C.wide(st.button))

        st.session_state.setdefault("report_text_widget", st.session_state.report_text)
        st.text_area("Report text (editable, or paste your own)", height=260, key="report_text_widget",
                     on_change=_text_changed)
        st.session_state.report_text = st.session_state.report_text_widget
        st.caption("Reports stay in this session and are not added to the knowledge base.")

    with right:
        st.subheader("Results")
        text = st.session_state.report_text.strip()
        if not text:
            st.info("Upload a report, paste text, or load the sample to begin.", icon="🧪")
            return
        if st.button("🔬 Analyze report", type="primary"):
            with st.spinner("Extracting lab values…"):
                try:
                    st.session_state.analysis = orchestrator.analyze_report_cached(text)
                except Exception as exc:  # noqa: BLE001
                    st.session_state.analysis = None
                    st.error(str(exc))
        result = st.session_state.analysis
        if result is None:
            return
        if not result:
            st.warning("No recognised diabetes-related values were found in this report.")
            return
        labels = [i["classification"] for i in result.values()]
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 In range", sum(l in GOOD for l in labels))
        m2.metric("🟡 Borderline", sum(l not in GOOD and l not in BAD and level_icon(l) == "🟡" for l in labels))
        m3.metric("🔴 Abnormal", sum(l in BAD for l in labels))
        st.dataframe(report_table(result), hide_index=True, **C.wide(st.dataframe))
        st.caption("A single value is not a diagnosis. Ranges follow common ADA cut-offs.")
        if st.button("💬 Ask questions about this report", **C.wide(st.button)):
            C.goto("chat")
