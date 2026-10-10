"""📚 Knowledge Base page: upload documents, TRAIN, manage and test what the bot has learned."""
import streamlit as st

import config
from src import documents, vectorstore
from views import _common as C

STATUS_PILLS = {
    "trained": ("✔ trained", "ok"),
    "new": ("● new: not trained", "warn"),
    "modified": ("● changed: not trained", "warn"),
}


def handle_uploads() -> None:
    files = st.file_uploader(
        "Drop PDF, TXT, MD or DOCX files here",
        type=[e.lstrip(".") for e in config.SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        key=f"kb_uploader_{st.session_state.uploader_n}",
    )
    st.caption("For guidelines and reference material. Your personal lab report goes in Chat (📎) or the "
               "Lab Reports page instead.")
    st.checkbox("Train automatically after upload", key="auto_train",
                help="On: uploaded files are trained straight away. Off: they wait until you press Train.")
    if not files:
        return
    saved, errors, patient_reports = [], [], []
    for f in files:
        try:
            name, outcome = vectorstore.add_document_bytes(f.name, f.getvalue())
            saved.append((name, outcome))
            if documents.looks_like_patient_report(f.name, f.getvalue()):
                patient_reports.append(name)
        except ValueError as exc:
            errors.append(str(exc))
    for name in patient_reports:
        C.flash("warning", f"'{name}' looks like a personal lab report. It was added to the shared knowledge base, "
                           "but questions about 'my report' only work when you attach it in Chat (📎) or on the "
                           "Lab Reports page. Delete it below if you don't want it in the shared knowledge.")
    changed = [n for n, o in saved if o != "unchanged"]
    for err in errors:
        C.flash("error", err)
    if changed:
        C.flash("success", f"Saved {len(changed)} file(s): {', '.join(changed)}.")
        if st.session_state.auto_train:
            C.request_training()
    elif saved:
        C.flash("info", "Those files are already in the knowledge base (identical content).")
    st.session_state.uploader_n += 1  # resets the uploader so files are not processed twice
    st.rerun()


def documents_table(status: dict) -> None:
    docs = status["documents"]
    if not docs and not status["removed"]:
        st.info("No documents yet. Upload some above.", icon="📭")
        return
    head = st.columns([4, 2, 2, 1, 1])
    for col, title in zip(head, ["Document", "Status", "Size", "Chunks", ""]):
        col.markdown(f"**{title}**")
    for d in docs:
        label, kind = STATUS_PILLS.get(d["status"], (d["status"], "muted"))
        c = st.columns([4, 2, 2, 1, 1])
        c[0].markdown(f"📄 {d['name']}")
        c[1].markdown(C.pill(label, kind), unsafe_allow_html=True)
        c[2].caption(C.human_size(d["size"]))
        c[3].caption(str(d["chunks"] if d["chunks"] is not None else "-"))
        with c[4].popover("🗑️", help="Delete this document"):
            st.write(f"Delete **{d['name']}** from the knowledge base?")
            if st.button("Yes, delete", key=f"del_{d['name']}", type="primary"):
                vectorstore.delete_document(d["name"])
                C.flash("success", f"Deleted {d['name']}.")
                if st.session_state.auto_train:
                    C.request_training()
                st.rerun()
    for name in status["removed"]:
        c = st.columns([4, 2, 2, 1, 1])
        c[0].markdown(f"~~{name}~~")
        c[1].markdown(C.pill("● deleted: still trained", "warn"), unsafe_allow_html=True)
        c[2].caption("removed on next Train")


def retrieval_test() -> None:
    kb = C.get_kb()
    if kb is None:
        return
    with st.expander("🔎 Test what the bot has learned (no LLM needed)"):
        query = st.text_input("Search the trained knowledge", key="kb_test_query",
                              placeholder="e.g. metformin side effects")
        if query.strip():
            try:
                hits = kb.search(query, k=st.session_state.top_k)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
            if not hits:
                st.caption("Nothing found.")
            for i, h in enumerate(hits, 1):
                st.markdown(f"**{i}. {h['source']}** · relevance {h['score']:.2f}")
                st.caption(h["content"][:500] + ("…" if len(h["content"]) > 500 else ""))


def render() -> None:
    C.page_header("📚", "Knowledge Base", "Upload documents, train the chatbot on them, and manage what it knows.")
    status = C.kb_status()
    level, text = C.kb_badge(status)
    getattr(st, level)(text, icon={"success": "✅", "warning": "⏳", "error": "📭"}[level])

    st.subheader("1 · Upload documents")
    handle_uploads()

    st.subheader("2 · Train")
    st.caption("Training reads your documents, splits them into chunks, and builds the searchable index the "
               "chatbot answers from. Only new or changed files are processed, so it stays quick.")
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("🚀 Train", type="primary", disabled=not status["documents"] and not status["removed"],
                 help="Train on new/changed documents.", **C.wide(st.button)):
        C.request_training(full=status["needs_full_retrain"])
        st.rerun()
    if c2.button("♻️ Full retrain", disabled=not status["documents"],
                 help="Rebuild the whole index from scratch.", **C.wide(st.button)):
        C.request_training(full=True)
        st.rerun()

    last = st.session_state.last_train
    if last and last.get("changed"):
        st.caption(f"Last run: {len(last['added'])} trained · {len(last['removed'])} removed · "
                   f"{last['unchanged']} reused · {last['n_chunks']} chunks · {last['seconds']}s")

    st.subheader("3 · Documents")
    documents_table(status)
    st.divider()
    retrieval_test()
