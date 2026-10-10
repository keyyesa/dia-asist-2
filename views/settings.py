"""⚙️ Settings page: model, retrieval and maintenance controls."""
import streamlit as st

import config
from src import llm_client, vectorstore
from views import _common as C


def _models_for_select(current: str):
    models = llm_client.list_ollama_models(timeout=1.5) or []
    options = list(dict.fromkeys(([current] if current else []) + models))
    return options, bool(models)


def render() -> None:
    C.page_header("⚙️", "Settings", "Choose the language model, tune retrieval, and maintain the knowledge base.")
    tab_llm, tab_kb, tab_maint = st.tabs(["🤖 Language model", "📐 Retrieval & training", "🧹 Maintenance"])

    with tab_llm:
        backend = st.radio("Backend", ["ollama", "gemini"], horizontal=True,
                           index=0 if config.LLM_BACKEND == "ollama" else 1,
                           format_func=lambda b: {"ollama": "🔒 Ollama (offline)", "gemini": "☁️ Gemini (cloud)"}[b])
        updates = {"LLM_BACKEND": backend}
        if backend == "ollama":
            options, found = _models_for_select(config.OLLAMA_MODEL)
            if found:
                model = st.selectbox("Installed Ollama model", options, index=options.index(config.OLLAMA_MODEL)
                                     if config.OLLAMA_MODEL in options else 0)
            else:
                st.caption("Could not list models (is Ollama running?). Type the model name instead.")
                model = st.text_input("Ollama model", config.OLLAMA_MODEL)
            url = st.text_input("Ollama URL", config.OLLAMA_BASE_URL)
            keep = st.text_input("Keep model loaded for", config.OLLAMA_KEEP_ALIVE,
                                 help="e.g. 30m, 2h, or -1 to keep it loaded forever. Avoids slow reloads.")
            updates.update(OLLAMA_MODEL=model, OLLAMA_BASE_URL=url, OLLAMA_KEEP_ALIVE=keep)
        else:
            key = st.text_input("Gemini API key", config.GEMINI_API_KEY if config.gemini_key_is_set() else "",
                                type="password")
            gmodel = st.text_input("Gemini model", config.GEMINI_MODEL)
            updates.update(GEMINI_API_KEY=key, GEMINI_MODEL=gmodel)

        c1, c2, c3 = st.columns(3)
        apply_clicked = c1.button("✅ Apply", type="primary", **C.wide(st.button))
        save_clicked = c2.button("💾 Apply & save to .env", **C.wide(st.button))
        test_clicked = c3.button("🔌 Test connection", **C.wide(st.button))
        if apply_clicked or save_clicked or test_clicked:
            try:
                config.update_settings(updates, persist=save_clicked)
                C._llm_health.clear()
                if save_clicked:
                    st.success("Saved to .env.")
                elif apply_clicked:
                    st.success("Applied (until the app restarts). Use “Apply & save” to keep it.")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            else:
                if test_clicked:
                    with st.spinner("Testing (this also loads the model into memory)…"):
                        health = llm_client.check_connection()
                        if not health["ok"]:
                            st.error(health["detail"])
                        else:
                            try:
                                llm_client.test_generation()
                                st.success(f"{health['backend'].title()} · {health['model']} responded correctly.")
                            except Exception as exc:  # noqa: BLE001
                                st.error(str(exc))

    with tab_kb:
        st.caption("Chunk settings apply the next time you train. Changing them triggers a full retrain.")
        c1, c2 = st.columns(2)
        size = c1.number_input("Chunk size (characters)", 100, 4000, config.CHUNK_SIZE, step=50)
        overlap = c2.number_input("Chunk overlap", 0, 1000, config.CHUNK_OVERLAP, step=10)
        embed = st.text_input("Embedding model", config.EMBEDDING_MODEL,
                              help="A sentence-transformers model name. Needs internet the first time only.")
        st.caption("The number of sources per answer (top-k) is adjusted live in the sidebar.")
        b1, b2 = st.columns(2)
        apply_kb = b1.button("✅ Apply", key="kb_apply", type="primary", **C.wide(st.button))
        save_kb = b2.button("💾 Apply & save to .env", key="kb_save", **C.wide(st.button))
        if apply_kb or save_kb:
            try:
                config.update_settings({"CHUNK_SIZE": size, "CHUNK_OVERLAP": overlap,
                                        "EMBEDDING_MODEL": embed}, persist=save_kb)
                st.success("Saved to .env." if save_kb else "Applied.")
                if vectorstore.get_training_status()["needs_full_retrain"]:
                    st.warning("These settings differ from the trained index. Press Train (a full retrain runs).",
                               icon="♻️")
            except (ValueError, OSError) as exc:
                st.error(str(exc))

    with tab_maint:
        agentic, _ = C.get_pipelines()
        if st.button("🧹 Clear response cache", disabled=agentic is None):
            agentic.clear_cache()
            st.success("Response cache cleared.")
        if agentic is not None:
            st.caption(f"{agentic.cache_size} cached answer(s).")
        if st.button("♻️ Full retrain now", disabled=not C.kb_status()["documents"]):
            C.request_training(full=True)
            st.rerun()
        if st.button("🔁 Reload model & index"):
            C.retry_kb_load()
            st.cache_resource.clear()
            st.rerun()
        st.divider()
        st.caption(f"Documents folder: `{config.RAW_DOCS_DIR}`  \nIndex folder: `{config.FAISS_INDEX_DIR}`")
