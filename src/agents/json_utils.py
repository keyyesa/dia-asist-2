"""Small helper shared by agents that ask the LLM for JSON."""
import json
import re


def extract_json(raw) -> dict:
    """Pull the first {...} object out of an LLM reply (handles ```json fences and extra prose).
    Returns {} if nothing parseable is found."""
    if not raw:
        return {}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
