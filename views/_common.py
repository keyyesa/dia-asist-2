"""Shared UI helpers: state, cached loaders, training runner, sidebar, small widgets."""
from __future__ import annotations

import hashlib
import inspect
from datetime import datetime

import streamlit as st

import config
from src import documents, llm_client, vectorstore
from src.baseline_rag import BaselineRAGPipeline
from src.orchestrator import AgenticRAGPipeline

MODE_LABELS = {"fast": "⚡ Fast (1 LLM call, streamed)", "full": "🧠 Full agentic (optimize + verify)"}

CSS = """
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.dia-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem;}
.dia-pill {display:inline-block; padding: 1px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;}
.dia-ok {background: rgba(46,160,67,.18); color: #2ea043;}
.dia-warn {background: rgba(210,153,34,.20); color: #d29922;}
.dia-bad {background: rgba(248,81,73,.18); color: #f85149;}
.dia-muted {background: rgba(128,128,128,.18); color: gray;}
[data-testid="stSidebar"] .stButton button {text-align: left;}
</style>
"""


def wide(fn) -> dict:
    """Kwargs that make a widget fill its container, across old and new Streamlit versions."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    width = params.get("width")
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def pill(text: str, kind: str = "muted") -> str:
    return f'<span class="dia-pill dia-{kind}">{text}</span>'


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def fmt_time(iso: str | None) -> str:
    if not iso:
        return "never"
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d %b %Y, %H:%M")
    except ValueError:
        return iso


# ----------------------------------------------------------------------------- state
def init_state() -> None:
    defaults = {
        "messages": [], "report_text": "", "report_name": None, "report_sig": None,
        "mode": "fast", "top_k": min(max(config.TOP_K, 1), 10), "auto_train": True,
        "uploader_n": 0, "run_train": None, "flash": [], "last_train": None,
        "kb_fail": None, "compare_history": [], "analysis": None, "pages": {}, "last_upload_sig": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def flash(level: str, message: str) -> None:
    """Queue a message that is shown at the top of the next run (survives st.rerun)."""
    st.session_state.flash.append((level, message))


def show_flash() -> None:
    queued, st.session_state.flash = st.session_state.flash, []
    for level, message in queued:
        getattr(st, level, st.info)(message, icon={"success": "✅", "warning": "⚠️", "error": "🚫"}.get(level, "ℹ️"))


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    st.title(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)


def goto(page_key: str) -> None:
    page = st.session_state.pages.get(page_key)
    if page is not None:
        st.switch_page(page)


# ----------------------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading the trained knowledge base…", max_entries=2)
def _load_kb(token: str, embedding_model: str):
    return vectorstore.KnowledgeBase()


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_pipelines(token: str, embedding_model: str):
    kb = _load_kb(token, embedding_model)
    return AgenticRAGPipeline(kb=kb), BaselineRAGPipeline(kb=kb)


def get_pipelines():
    """(agentic, baseline) for the CURRENT trained index, or (None, None) if unavailable.
    The cache key is the index version, so a new training run is picked up automatically."""
    token = vectorstore.index_token()
    if token is None:
        return None, None
    fail = st.session_state.get("kb_fail")
    if fail and fail[0] == token and fail[1] == config.EMBEDDING_MODEL:
        return None, None  # already failed for this exact index; don't retry on every rerun
    try:
        pipelines = _load_pipelines(token, config.EMBEDDING_MODEL)
        st.session_state.kb_fail = None
        return pipelines
    except Exception as exc:  # noqa: BLE001
        st.session_state.kb_fail = (token, config.EMBEDDING_MODEL, str(exc))
        return None, None


def get_kb():
    agentic, _ = get_pipelines()
    return agentic.kb if agentic else None


def kb_load_error() -> str | None:
    fail = st.session_state.get("kb_fail")
    return fail[2] if fail else None


def retry_kb_load() -> None:
    st.session_state.kb_fail = None


def kb_status() -> dict:
    return vectorstore.get_training_status()


def kb_badge(status: dict) -> tuple[str, str]:
    """(streamlit level, one-line description) summarising knowledge-base state."""
    n_docs = len(status["documents"])
    if not status["index_exists"] and n_docs == 0:
        return "error", "No documents yet. Upload some to get started."
    if not status["index_exists"]:
        return "warning", f"Not trained yet: {n_docs} document(s) waiting."
    if status["needs_full_retrain"]:
        return "warning", "Settings changed: a full retrain is needed."
    if status["legacy"]:
        return "warning", "Index from an older version: click Train once to refresh it."
    if status["pending"]:
        return "warning", f"{status['n_pending']} change(s) not trained yet."
    return "success", f"Trained: {n_docs} document(s), {status['n_chunks']} chunks."


@st.cache_data(ttl=15, show_spinner=False)
def _llm_health(backend: str, url: str, model: str, key_set: bool) -> dict:
    return llm_client.check_connection()


def llm_health() -> dict:
    return _llm_health(config.LLM_BACKEND, config.OLLAMA_BASE_URL, config.OLLAMA_MODEL
                       if config.LLM_BACKEND == "ollama" else config.GEMINI_MODEL, config.gemini_key_is_set())


# ----------------------------------------------------------------------------- training
def request_training(full: bool = False) -> None:
    st.session_state.run_train = "full" if full else "incremental"


def handle_training_request() -> None:
    """Runs a requested training with a live progress bar (called once per run from app.py)."""
    mode = st.session_state.get("run_train")
    if not mode:
        return
    st.session_state.run_train = None
    with st.status("Training the knowledge base…", expanded=True) as box:
        bar = st.progress(0.0, text="Starting…")
        try:
            summary = vectorstore.train(
                force_full=(mode == "full"),
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        except vectorstore.TrainingBusyError as exc:
            box.update(label=str(exc), state="error")
            return
        except Exception as exc:  # noqa: BLE001
            box.update(label="Training failed", state="error", expanded=True)
            st.error(f"{exc}")
            return
        box.update(label="Training complete", state="complete", expanded=False)

    st.session_state.last_train = summary
    st.session_state.kb_fail = None
    if summary["n_chunks"] == 0:
        flash("warning", "No readable documents found, so the knowledge base is empty. Upload documents first.")
    elif not summary["changed"]:
        flash("info", "Everything was already trained, nothing to do.")
    else:
        parts = []
        if summary["added"]:
            parts.append(f"{len(summary['added'])} file(s) trained")
        if summary["removed"]:
            parts.append(f"{len(summary['removed'])} removed")
        flash("success", f"Training complete: {', '.join(parts) or 'index rebuilt'} → "
                         f"{summary['n_chunks']} chunks ready ({summary['seconds']}s). The chatbot now answers from this data.")
    for name, reason in summary.get("skipped", []):
        flash("warning", f"Skipped '{name}': {reason}")
    st.rerun()


# ----------------------------------------------------------------------------- reports
def load_report_from_upload(uploaded) -> bool:
    """Extract text from an uploaded lab report once (cached by name+size). Returns True if usable."""
    sig = (uploaded.name, uploaded.size)
    if st.session_state.last_upload_sig == sig:  # already processed: don't overwrite later edits/samples
        return bool(st.session_state.report_text)
    try:
        text = documents.extract_text(uploaded.name, uploaded.getvalue(), page_markers=False)
    except ValueError as exc:
        st.error(str(exc))
        return False
    if not text.strip():
        st.warning("No text could be extracted. Scanned images need OCR first; try a text-based PDF or TXT.")
        return False
    set_report(text, uploaded.name, sig)
    st.session_state.last_upload_sig = sig
    return True


def set_report(text: str, name: str | None, sig=None) -> None:
    st.session_state.report_text = text
    st.session_state.report_name = name
    st.session_state.report_sig = sig or ("pasted", hashlib.sha1(text.encode()).hexdigest())
    st.session_state["report_text_widget"] = text
    st.session_state.analysis = None


def clear_report() -> None:
    st.session_state.report_text = ""
    st.session_state.report_name = None
    st.session_state.report_sig = None
    st.session_state.last_upload_sig = None
    st.session_state["report_text_widget"] = ""
    st.session_state.analysis = None
    st.session_state.uploader_n += 1


# ----------------------------------------------------------------------------- sidebar
def render_sidebar() -> None:
    status = kb_status()
    level, text = kb_badge(status)
    health = llm_health()
    with st.sidebar:
        st.divider()
        st.markdown("##### 🧭 Quick controls")
        st.radio("Response mode", list(MODE_LABELS), format_func=MODE_LABELS.get, key="mode",
                 help="Fast skips query rewriting and verification for much quicker answers. "
                      "Full runs every agent for the most careful answer.")
        st.slider("Sources to retrieve (top-k)", 1, 10, key="top_k",
                  help="How many chunks of trained knowledge are given to the model.")

        st.markdown("##### 📚 Knowledge base")
        getattr(st, level)(text, icon={"success": "✅", "warning": "⚠️", "error": "📭"}[level])
        if status["pending"] or not status["index_exists"]:
            if status["documents"] and st.button("🚀 Train now", type="primary", key="sb_train", **wide(st.button)):
                request_training(full=status["needs_full_retrain"])

        st.markdown("##### 🤖 Language model")
        icon = "🟢" if health["ok"] else "🔴"
        st.caption(f"{icon} **{health['backend'].title()}** · `{health['model']}`")
        if not health["ok"]:
            st.caption(health["detail"])

        if st.session_state.messages and st.button("🧹 Clear chat", key="sb_clear", **wide(st.button)):
            st.session_state.messages = []
            st.rerun()
        st.caption("Research prototype. Not medical advice.")
