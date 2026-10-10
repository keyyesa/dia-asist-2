"""
Post-Processing Verification Agent.

Two layers:
  1. Rule-based citation check (deterministic): every [Source: ...] in the response must
     match a source that was actually retrieved (or the uploaded lab report).
  2. LLM claim check: are the factual claims supported by the context?
     If problems are found, a second call rewrites the response to fix them.

If the verifier itself fails (bad output / API error) the result is reported honestly as
verified=False with verification_status="failed" -- it is never silently marked verified.
"""
import re
from src import llm_client
from src.agents.json_utils import extract_json

CHECK_PROMPT = """You are the Post-Processing Verification Agent for a diabetes clinical
decision support system.
You are given a generated response and the source context it was supposed to rely on.
Check:
1. Does every factual claim have support in the context?
2. Is there any hallucinated or unsupported medical claim (numbers, thresholds, drug names)?
Return ONLY a JSON object with EXACTLY these keys:
{"verified": true or false, "issues": ["short description of each problem, empty list if none"]}"""

CORRECT_PROMPT = """You are the Post-Processing Verification Agent for a diabetes clinical
decision support system.
Rewrite the response so that every claim is supported by the provided context.
Remove or fix any unsupported or incorrect statement listed in the issues.
Keep valid [Source: ...] citations and keep the closing medical disclaimer.
Return ONLY the corrected response text, nothing else."""

CITATION_RE = re.compile(r"\[Source:\s*([^\]]+)\]")
REPORT_ALIASES = {"lab report", "uploaded lab report", "report", "lab report interpretation"}


def _format_report(report_interpretation) -> str:
    if not report_interpretation:
        return ""
    lines = [f"- {p}: {i['value']} {i.get('unit') or ''} -> {i['classification']}"
             + (f" (lab range {i['lab_range']})" if i.get("lab_range") else "")
             for p, i in report_interpretation.items()]
    return "[Source: Lab Report]\nLab Report Interpretation:\n" + "\n".join(lines)


def check_citations(response: str, context_chunks: list, has_report: bool = False) -> list:
    """Deterministic check: cited source names must exist among retrieved sources."""
    known = {c["source"].strip().lower() for c in context_chunks}
    if has_report:
        known |= REPORT_ALIASES
    issues = []
    cited = [m.split(",")[0].strip() for m in CITATION_RE.findall(response)]
    for name in dict.fromkeys(cited):
        if name.lower() not in known:
            issues.append(f"Citation to '{name}' does not match any retrieved source.")
    return issues


def _correct(response: str, context_text: str, issues: list):
    prompt = (f"Original response:\n{response}\n\nIssues found:\n- " + "\n- ".join(issues) +
              f"\n\nSource context:\n{context_text}\n\nCorrected response:")
    try:
        corrected = llm_client.generate(prompt, system_instruction=CORRECT_PROMPT, temperature=0.0)
    except Exception:  # noqa: BLE001
        return None
    corrected = (corrected or "").strip()
    return corrected if len(corrected) >= 30 else None


def verify(response: str, context_chunks: list, report_interpretation: dict = None,
           report_text: str = None) -> dict:
    has_report = bool(report_interpretation) or bool(report_text and report_text.strip())
    issues = check_citations(response, context_chunks, has_report=has_report)

    context_parts = [f"[Source: {c['source']}]\n{c['content']}" for c in context_chunks]
    report_block = _format_report(report_interpretation)
    if report_block:
        context_parts.append(report_block)
    if report_text and report_text.strip():
        context_parts.append("[Source: Lab Report]\nRaw report text:\n" + report_text.strip()[:3500])
    context_text = "\n\n".join(context_parts) if context_parts else "(no context was available)"

    prompt = f"Generated response:\n{response}\n\nSource context:\n{context_text}\n\nVerify now."
    try:
        raw = llm_client.generate(prompt, system_instruction=CHECK_PROMPT, temperature=0.0)
    except Exception:  # noqa: BLE001
        raw = None
    parsed = extract_json(raw)

    if "verified" not in parsed:
        return {
            "verified": False,
            "issues": issues + ["Verification could not be completed; response is unverified."],
            "corrected_response": response,
            "verification_status": "failed",
        }

    llm_ok = parsed["verified"]
    if isinstance(llm_ok, str):
        llm_ok = llm_ok.strip().lower() == "true"
    llm_issues = parsed.get("issues") or []
    if not isinstance(llm_issues, list):
        llm_issues = [str(llm_issues)]
    issues += [str(i) for i in llm_issues if i]

    if bool(llm_ok) and not issues:
        return {"verified": True, "issues": [], "corrected_response": response, "verification_status": "ok"}

    corrected = _correct(response, context_text, issues or ["Unsupported claims reported by verifier."])
    return {
        "verified": False,
        "issues": issues,
        "corrected_response": corrected or response,
        "verification_status": "corrected" if corrected else "flagged",
    }
