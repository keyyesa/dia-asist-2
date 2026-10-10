"""Router Agent: decides which downstream path a request needs."""
import re
from src import llm_client

SYSTEM_PROMPT = """You are the Router Agent in a diabetes clinical decision support system.
Decide which processing path the request needs, given the query and whether a lab report was uploaded.
Respond with exactly one label and nothing else:
KNOWLEDGE_ONLY  -> the question is general medical/guideline knowledge and does not depend on the uploaded report
REPORT_ONLY     -> the question is purely about interpreting the uploaded lab report values
BOTH            -> the question needs both guideline knowledge AND the uploaded report"""

# Questions that clearly refer to the user's own report ("is my report normal?", "what is my RBG
# from the report"). Decided by rule, so a small LLM can never route them away from the report.
PERSONAL_RE = re.compile(
    r"\b(my|mine|this|the|our|attached|uploaded)\s+(lab\s+)?(report|reports|result|results|labs|"
    r"test|tests|values|readings?|numbers|levels?)\b|\bfrom\s+(the|my|this)\s+report\b|\bin\s+(the|my)\s+report\b"
    r"|\b(is|are)\s+(it|they)\s+normal\b",
    re.IGNORECASE,
)


def refers_to_own_report(query: str) -> bool:
    return bool(PERSONAL_RE.search(query or ""))


def route(query: str, has_report: bool) -> str:
    # No report uploaded -> nothing to route, no LLM call needed.
    if not has_report:
        return "KNOWLEDGE_ONLY"
    if refers_to_own_report(query):
        return "BOTH"

    prompt = f"Lab report uploaded: YES\nQuery: {query}\n\nLabel:"
    try:
        result = llm_client.generate(prompt, system_instruction=SYSTEM_PROMPT, temperature=0.0)
    except Exception:  # noqa: BLE001
        return "BOTH"
    result = (result or "").strip().upper()
    if "REPORT_ONLY" in result:
        return "REPORT_ONLY"
    if "KNOWLEDGE_ONLY" in result:
        return "KNOWLEDGE_ONLY"
    # Unclear/garbled output: safest default when a report exists is to use everything,
    # so the uploaded report is never silently ignored.
    return "BOTH"
