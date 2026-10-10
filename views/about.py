"""ℹ️ About page."""
import streamlit as st

from views import _common as C

DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8f4f5", fontname="Helvetica"];
  Q [label="Question\\n(+ lab report)"]; O [label="Query\\nOptimizer"]; R [label="Router"];
  K [label="Knowledge\\nRetrieval\\n(FAISS)"]; L [label="Report\\nAnalysis"];
  G [label="Response\\nGeneration"]; V [label="Verification"]; A [label="Answer", fillcolor="#d9f2dd"];
  Q -> O -> R; R -> K; R -> L; K -> G; L -> G; G -> V -> A;
}
"""


def render() -> None:
    C.page_header("ℹ️", "About DiaAssist AI")
    st.markdown(
        "**DiaAssist AI** is a multi-agent Agentic-RAG clinical decision support prototype for diabetes. "
        "It answers from documents **you train it on**, can interpret lab reports, and runs fully offline "
        "with a local model (Ollama)."
    )
    st.graphviz_chart(DOT)
    st.markdown(
        "**Fast mode** uses only retrieval + one streamed LLM call. **Full mode** adds the Query Optimizer, "
        "Router and Verification agents.\n\n"
        "**Training** = reading your documents, chunking them, embedding the chunks and storing them in a FAISS "
        "index. Only new or changed documents are processed.\n\n"
        "⚠️ This is a research/academic prototype. It is not a medical device and must not be used for real "
        "clinical decisions without qualified professional oversight."
    )
