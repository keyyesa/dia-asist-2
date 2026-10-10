"""🏠 Dashboard: system status at a glance and a guided quick start."""
import streamlit as st

import config
from views import _common as C


def render() -> None:
    C.page_header("🩺", "DiaAssist AI", "Agentic-RAG clinical decision support for diabetes. A research prototype, "
                                         "not a substitute for professional medical advice.")
    status = C.kb_status()
    health = C.llm_health()
    pipeline, _ = C.get_pipelines()
    trained = [d for d in status["documents"] if d["status"] == "trained"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", f"{len(trained)}/{len(status['documents'])} trained")
    c2.metric("Knowledge chunks", status["n_chunks"] if status["n_chunks"] is not None
              else (pipeline.kb.n_chunks if pipeline else 0))
    c3.metric("Last trained", C.fmt_time(status["trained_at"]))
    c4.metric("Language model", "Online" if health["ok"] else "Offline",
              help=health["detail"])

    st.subheader("Quick start")
    ready_kb = pipeline is not None
    steps = [
        (ready_kb and not status["pending"], "Train the knowledge base",
         "Upload PDF/TXT/MD/DOCX guidelines, then press Train.", "kb", "📚 Open Knowledge Base"),
        (health["ok"], "Connect the language model",
         health["detail"] if not health["ok"] else f"{health['backend'].title()} · {health['model']} is ready.",
         "settings", "⚙️ Open Settings"),
        (ready_kb and health["ok"], "Ask questions",
         "Answers are written from your trained documents, with sources.", "chat", "💬 Open Chat"),
    ]
    for i, (done, title, hint, page, label) in enumerate(steps, 1):
        with st.container(border=True):
            a, b = st.columns([5, 1.4], vertical_alignment="center")
            a.markdown(f"{'✅' if done else '⬜'} **{i}. {title}**  \n{hint}")
            if b.button(label, key=f"qs_{page}", **C.wide(st.button)):
                C.goto(page)

    if C.kb_load_error():
        st.error(f"Could not load the trained knowledge base: {C.kb_load_error()}")
        st.button("🔁 Try again", on_click=C.retry_kb_load)

    st.subheader("What you can do")
    a, b, c = st.columns(3)
    with a.container(border=True):
        st.markdown("**💬 Chat**\n\nAsk diabetes questions. Choose ⚡ Fast or 🧠 Full mode in the sidebar.")
        st.page_link(st.session_state.pages["chat"], label="Open", icon="➡️")
    with b.container(border=True):
        st.markdown("**🧪 Lab reports**\n\nExtract HbA1c, glucose, lipids… and classify them against reference ranges.")
        st.page_link(st.session_state.pages["lab"], label="Open", icon="➡️")
    with c.container(border=True):
        st.markdown("**⚖️ Compare**\n\nSee Agentic-RAG vs. a traditional RAG baseline side by side.")
        st.page_link(st.session_state.pages["compare"], label="Open", icon="➡️")

    with st.expander("⚡ Why answers are fast"):
        st.markdown(
            "- Documents are **trained once** into a vector index; each question only searches that index.\n"
            "- **Fast mode** uses a single streamed LLM call instead of up to five.\n"
            f"- The Ollama model stays loaded in memory for {config.OLLAMA_KEEP_ALIVE} between questions.\n"
            "- Repeated questions and lab reports are **cached**."
        )
