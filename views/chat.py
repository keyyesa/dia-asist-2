"""💬 Chat page: ask questions against the trained knowledge base."""
import streamlit as st
import pandas as pd

from src.agents import router
from views import _common as C

EXAMPLES = [
    "What is HbA1c and what do the different ranges mean?",
    "What are the symptoms and treatment of hypoglycemia?",
    "How is type 2 diabetes diagnosed?",
    "Which lifestyle changes help manage blood sugar?",
]

GOOD = {"normal", "optimal", "acceptable", "below_diagnostic_threshold"}
BAD = {"diabetes_range", "high", "very_high", "macroalbuminuria", "elevated"}


def level_icon(label: str) -> str:
    if label in GOOD:
        return "🟢"
    if label in BAD:
        return "🔴"
    if label in ("unknown", "unit_not_supported"):
        return "⚪"
    return "🟡"


def report_table(interpretation: dict) -> pd.DataFrame:
    rows = []
    for param, info in interpretation.items():
        rows.append({
            "Parameter": param,
            "Value": info.get("value"),
            "Unit": info.get("unit") or "",
            "Result": f"{level_icon(info['classification'])} {info['classification'].replace('_', ' ')}",
            "Lab range": info.get("lab_range", ""),
            "Reported as": info.get("reported_as", ""),
        })
    return pd.DataFrame(rows)


def make_meta(result: dict) -> dict:
    return {
        "mode": result["mode"], "time": result["response_time_sec"], "cached": result["cached"],
        "route": result["route"], "optimized_query": result["optimized_query"],
        "verified": result["verified"], "status": result["verification_status"],
        "issues": result["verification_issues"], "report": result["report_interpretation"],
        "sources": [{"source": c["source"], "score": c.get("score", 0.0), "content": c["content"]}
                    for c in result["retrieved_chunks"]],
    }


def render_meta(meta: dict, key: str) -> None:
    bits = ["⚡ Fast" if meta["mode"] == "fast" else "🧠 Full", f"{meta['time']}s"]
    if meta["cached"]:
        bits.append("instant (cached)")
    bits.append(f"{len(meta['sources'])} source(s)")
    if meta["mode"] == "full":
        bits.append({"ok": "✅ verified", "corrected": "🛠️ corrected by verifier",
                     "flagged": "⚠️ flagged by verifier"}.get(meta["status"], "❓ verification incomplete"))
    st.caption(" · ".join(bits))
    with st.expander("Sources & details"):
        if meta["report"]:
            st.markdown("**Lab report interpretation**")
            st.dataframe(report_table(meta["report"]), hide_index=True, **C.wide(st.dataframe))
        if meta["sources"]:
            st.markdown("**Retrieved from the trained knowledge base**")
            for i, s in enumerate(meta["sources"], 1):
                st.markdown(f"{i}. `{s['source']}` · relevance {s['score']:.2f}")
                text = s["content"]
                st.caption(text[:350] + ("…" if len(text) > 350 else ""))
        else:
            st.caption("No knowledge-base passages were used for this answer.")
        st.caption(f"Route: {meta['route']}"
                   + (f" · Optimized query: {meta['optimized_query']}" if meta["mode"] == "full" else ""))
        if meta["issues"]:
            st.warning("Verifier notes: " + "; ".join(meta["issues"]))


NO_REPORT_NOTICE = ("No lab report is attached to this chat, so I can only answer from the trained guidelines. "
                    "Attach your report with the 📎 box above (documents trained in the Knowledge Base are shared "
                    "guideline knowledge, not your personal report).")


def transcript() -> str:
    lines = ["# DiaAssist AI chat\n"]
    for m in st.session_state.messages:
        lines.append(f"**{'You' if m['role'] == 'user' else 'DiaAssist'}:** {m['content']}\n")
    return "\n".join(lines)


def report_bar() -> None:
    name = st.session_state.report_name
    label = f"📎 Lab report: {name}" if st.session_state.report_text else "📎 Attach a lab report (optional)"
    with st.expander(label, expanded=False):
        uploaded = st.file_uploader("PDF, TXT or DOCX", type=["pdf", "txt", "docx"],
                                    key=f"chat_report_{st.session_state.uploader_n}")
        if uploaded:
            C.load_report_from_upload(uploaded)
        if st.session_state.report_text:
            st.success(f"Attached: {st.session_state.report_name or 'pasted text'}. Your questions will use it.")
            st.button("Remove report", key="chat_rm_report", on_click=C.clear_report)
        st.caption("Reports are analysed once per session and are NOT added to the shared knowledge base.")


def answer(prompt: str, pipeline) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        notice = NO_REPORT_NOTICE if (not st.session_state.report_text and router.refers_to_own_report(prompt)) else None
        if notice:
            st.info(notice, icon="📎")
        box = st.status("Working…", expanded=False)
        placeholder = st.empty()
        buf = []

        def on_stage(_key, message):
            box.update(label=message)

        def on_token(piece):
            buf.append(piece)
            placeholder.markdown("".join(buf) + "▌")

        try:
            result = pipeline.run(
                prompt, report_text=st.session_state.report_text or None,
                mode=st.session_state.mode, top_k=st.session_state.top_k,
                on_stage=on_stage, on_token=on_token,
            )
        except Exception as exc:  # noqa: BLE001 - show any failure in the chat instead of a stack trace
            box.update(label="Something went wrong", state="error", expanded=True)
            placeholder.empty()
            message = f"⚠️ {exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message, "error": True,
                                              "notice": notice})
            return
        placeholder.markdown(result["response"])
        box.update(label=f"Done in {result['response_time_sec']}s" + (" (cached)" if result["cached"] else ""),
                   state="complete")
        meta = make_meta(result)
        render_meta(meta, key=f"m{len(st.session_state.messages)}")
        st.session_state.messages.append({"role": "assistant", "content": result["response"], "meta": meta,
                                          "notice": notice})


def render() -> None:
    C.page_header("💬", "Ask DiaAssist", "Answers come from the documents you have trained, with sources shown.")
    status = C.kb_status()
    pipeline, _ = C.get_pipelines()

    if pipeline is None:
        if status["index_exists"] and C.kb_load_error():
            st.error(f"The trained knowledge base could not be loaded: {C.kb_load_error()}")
            st.button("🔁 Try again", on_click=C.retry_kb_load)
        elif status["documents"]:
            st.warning("The chatbot has not been trained yet. Train it on your documents first.", icon="🧠")
            if st.button("🚀 Train now", type="primary"):
                C.request_training()
                st.rerun()
        else:
            st.info("No knowledge yet. Upload documents on the Knowledge Base page, then train.", icon="📭")
            st.page_link(st.session_state.pages["kb"], label="Go to Knowledge Base", icon="📚")
        return

    if status["pending"]:
        c1, c2 = st.columns([4, 1])
        c1.warning(f"{status['n_pending'] or 'Some'} document change(s) are not trained yet. "
                   "Answers only use what was already trained.", icon="⏳")
        if c2.button("🚀 Train now", type="primary", key="chat_train"):
            C.request_training(full=status["needs_full_retrain"])
            st.rerun()

    report_bar()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("notice"):
                st.info(msg["notice"], icon="📎")
            if msg.get("error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
                if msg.get("meta"):
                    render_meta(msg["meta"], key=f"h{i}")

    pending_prompt = None
    if not st.session_state.messages:
        st.markdown("##### Try a question")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex{i}", **C.wide(st.button)):
                pending_prompt = example

    if st.session_state.messages:
        st.download_button("⬇️ Export chat (.md)", transcript(), "diaassist_chat.md", "text/markdown")

    prompt = st.chat_input("Ask a diabetes-related question…") or pending_prompt
    if prompt:
        answer(prompt, pipeline)
