"""Query Optimizer Agent: rewrites the raw user question for better retrieval."""
from src import llm_client

SYSTEM_PROMPT = """You are the Query Optimizer Agent in a diabetes clinical decision support system.
Rewrite the user's question into a clear, specific, retrieval-friendly query.
Rules:
- Keep standard medical abbreviations as-is (HbA1c, FBG, RBG, OGTT, LDL, HDL).
- Fix grammar, spelling, and ambiguity.
- Do not change the medical meaning or add new clinical claims.
- Return ONLY the rewritten query text, nothing else."""


def optimize(raw_query: str) -> str:
    prompt = f"Original query: {raw_query}\n\nRewritten query:"
    result = llm_client.generate(prompt, system_instruction=SYSTEM_PROMPT, temperature=0.1)
    return result if result else raw_query
