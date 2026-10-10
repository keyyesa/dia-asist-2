"""📊 Evaluation page: run the proposal's evaluation and view/download the results."""
import streamlit as st

from evaluation import evaluate
from views import _common as C


def render() -> None:
    C.page_header("📊", "Evaluation", "Run the test questions through both pipelines and compare the metrics.")
    agentic, baseline = C.get_pipelines()
    if agentic is None:
        st.info("Train the knowledge base first.", icon="📭")
        st.page_link(st.session_state.pages["kb"], label="Go to Knowledge Base", icon="📚")
        return
    questions = evaluate.load_questions()
    with st.expander(f"Test questions ({len(questions)})"):
        for q in questions:
            st.markdown(f"- {q['question']}")
    st.caption("Each question runs the full agentic pipeline plus the baseline, so this can take a while "
               "on CPU-only machines.")

    limit = st.slider("Number of questions to run", 1, len(questions), min(3, len(questions)))
    if st.button("▶️ Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Starting…")

        def progress(done, total, question):
            bar.progress(done / max(total, 1), text=f"Question {min(done + 1, total)}/{total}: {question}"
                         if question else "Finished")

        df = evaluate.run_evaluation(agentic=agentic, baseline=baseline, progress_cb=progress,
                                     limit=limit, verbose=False)
        st.session_state["eval_df"] = df

    df = st.session_state.get("eval_df")
    if df is not None:
        summary = evaluate.summarize(df)
        if summary:
            cols = st.columns(3)
            for i, (k, v) in enumerate(summary.items()):
                cols[i % 3].metric(k, v)
        failed = df[df["error"] != ""]
        if len(failed):
            st.error(f"{len(failed)} question(s) failed. First error: {failed.iloc[0]['error']}")
        st.dataframe(df, hide_index=True, **C.wide(st.dataframe))
        st.download_button("⬇️ Download results.csv", df.to_csv(index=False), "results.csv", "text/csv")
