"""Response Generation Agent: produces the final answer from retrieved context and/or the lab report."""
from src import llm_client
from src.agents import report_analysis

SYSTEM_PROMPT = """You are the Response Generation Agent for DiaAssist AI, a diabetes clinical
decision support system.
Generate a clear, structured, evidence-based answer using ONLY the provided context
(retrieved guideline excerpts and/or the uploaded lab report).
Rules:
- Cite sources inline like [Source: <source_name>] right after any claim drawn from the guideline context.
  Facts taken from the uploaded lab report are cited as [Source: Lab Report].
- When a lab report is provided, the user's questions such as "is my report normal?" or "what is my X?"
  are about THAT report. Answer from the "Lab report values" list and the raw report text. For each value
  state the number, its unit, how it classifies, and the lab's own reference range if one is given.
  Clearly flag anything abnormal or borderline. If everything is in range, say so plainly.
- If the user asks about a test that appears in "Not found in this report", say clearly that the report
  does not contain that test. Never guess or invent a value.
- A single value is not a diagnosis; say so when you judge normal/abnormal.
- Do not invent facts that are not present in the context. If the context is insufficient to
  answer fully, say so explicitly.
- Write in clear prose paragraphs, avoid unnecessary jargon.
- Always end with a short disclaimer that this is decision support only and does not replace
  the judgment of a qualified healthcare professional."""

MAX_REPORT_CHARS = 3500


def format_report_values(report_interpretation: dict) -> str:
    lines = []
    for param, info in report_interpretation.items():
        line = f"- {param}: {info['value']} {info.get('unit') or ''} -> {info['classification'].replace('_', ' ')}"
        if info.get("lab_range"):
            line += f" (lab reference range: {info['lab_range']})"
        lines.append(line)
    return "\n".join(lines)


def build_report_block(report_interpretation: dict, report_text: str = None) -> str:
    """Everything the model needs to know about the uploaded report ('' if there is none)."""
    has_text = bool(report_text and report_text.strip())
    if not report_interpretation and not has_text:
        return ""
    parts = ["Uploaded lab report:"]
    if report_interpretation:
        parts.append("Lab report values (classified with fixed reference ranges):\n"
                     + format_report_values(report_interpretation))
        missing = report_analysis.missing_parameters(report_interpretation)
        if missing:
            parts.append("Not found in this report: " + ", ".join(missing) + ".")
    else:
        parts.append("No standard diabetes values could be read automatically; use the raw text below.")
    if has_text:
        text = report_text.strip()
        if len(text) > MAX_REPORT_CHARS:
            text = text[:MAX_REPORT_CHARS] + "\n[...report truncated...]"
        parts.append("Raw report text (may be messy):\n" + text)
    return "\n\n".join(parts)


def generate_response(query: str, retrieved_chunks: list, report_interpretation: dict = None,
                      on_token=None, report_text: str = None) -> str:
    """on_token(piece) is called for every streamed piece of text (optional)."""
    context_parts = [f"[Source: {c['source']}]\n{c['content']}" for c in retrieved_chunks]
    context_text = "\n\n".join(context_parts) if context_parts else "No guideline context was retrieved."
    report_block = build_report_block(report_interpretation, report_text)

    prompt = f"""User query: {query}

Retrieved guideline context:
{context_text}

{report_block}

Generate the response now."""

    if on_token is None:
        return llm_client.generate(prompt, system_instruction=SYSTEM_PROMPT, temperature=0.3)

    pieces = []
    for piece in llm_client.generate_stream(prompt, system_instruction=SYSTEM_PROMPT, temperature=0.3):
        pieces.append(piece)
        on_token(piece)
    return "".join(pieces).strip()
