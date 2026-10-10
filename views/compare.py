"""⚖️ Compare page: Agentic-RAG vs the traditional RAG baseline."""
import pandas as pd
import streamlit as st

from views import _common as C


def render() -> None:
    C.page_header("⚖️", "Compare Agentic vs Baseline RAG",
                  "Run the same question through both pipelines and compare answer and speed.")
    agentic, baseline = C.get_pipelines()
    if agentic is None:
        st.info("Train the knowledge base first.", icon="📭")
        st.page_link(st.session_state.pages["kb"], label="Go to Knowledge Base", icon="📚")
        return
    st.caption("The agentic pipeline always runs in Full mode here, with the cache bypassed for honest timings.")

    with st.form("compare_form"):
        query = st.text_input("Question to compare", placeholder="e.g. How is gestational diabetes diagnosed?")
        go = st.form_submit_button("⚖️ Compare", type="primary")

    if go:
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running both pipelines…"):
                try:
                    a = agentic.run(query, mode="full", top_k=st.session_state.top_k, use_cache=False)
                    b = baseline.run(query, k=st.session_state.top_k)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                    return
            st.session_state.compare_history.append({
                "Question": query, "Agentic (s)": a["response_time_sec"], "Baseline (s)": b["response_time_sec"],
                "Agentic verified": {True: "yes", False: "no", None: "-"}[a["verified"]],
            })
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 🧠 Agentic-RAG")
                st.write(a["response"])
                st.caption(f"Sources: {', '.join(dict.fromkeys(a['retrieved_sources'])) or 'none'} · {a['response_time_sec']}s")
            with col_b:
                st.markdown("#### 📄 Traditional RAG")
                st.write(b["response"])
                st.caption(f"Sources: {', '.join(dict.fromkeys(b['retrieved_sources'])) or 'none'} · {b['response_time_sec']}s")

    if st.session_state.compare_history:
        st.subheader("This session")
        st.dataframe(pd.DataFrame(st.session_state.compare_history), hide_index=True, **C.wide(st.dataframe))
