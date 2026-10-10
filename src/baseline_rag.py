"""
Traditional (non-agentic) RAG pipeline used as the comparison baseline
required by the project's evaluation plan: single embed -> retrieve -> generate,
no query optimization, routing, or verification step.
"""
import time

import config
from src.vectorstore import KnowledgeBase
from src import llm_client

SYSTEM_PROMPT = """You are a diabetes information assistant. Answer the user's question using
only the provided context. Cite sources inline like [Source: <source_name>]. If the context is
insufficient, say so. End with a disclaimer that this is not a replacement for professional
medical advice."""


class BaselineRAGPipeline:
    def __init__(self, kb=None):
        self.kb = kb or KnowledgeBase()

    def run(self, query: str, k: int = None) -> dict:
        t0 = time.time()
        query = (query or "").strip()
        if not query:
            raise ValueError("Please enter a question.")
        chunks = self.kb.search(query, k=k or config.TOP_K)
        context = "\n\n".join(f"[Source: {c['source']}]\n{c['content']}" for c in chunks)
        prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        response = llm_client.generate(prompt, system_instruction=SYSTEM_PROMPT, temperature=0.3)
        return {
            "query": query,
            "retrieved_sources": [c["source"] for c in chunks],
            "retrieved_chunks": chunks,
            "response": response,
            "response_time_sec": round(time.time() - t0, 2),
        }
