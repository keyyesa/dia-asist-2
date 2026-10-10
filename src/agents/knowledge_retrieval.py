"""Knowledge Retrieval Agent: semantic search over the trained FAISS knowledge base."""
import config


def retrieve(kb, query: str, k: int = None) -> list:
    k = k or config.TOP_K
    return kb.search(query, k=k)
