"""Central configuration. Values come from .env (see .env.example) and can be
changed at runtime from the Settings page via update_settings()."""
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)


def _get_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _get_path(name: str, default: str) -> str:
    raw = (os.getenv(name) or default).strip() or default
    raw = os.path.expanduser(raw)
    # Relative paths are resolved against the project folder, so the app works
    # no matter which directory you start it from.
    return raw if os.path.isabs(raw) else os.path.normpath(os.path.join(BASE_DIR, raw))


# "ollama" = fully offline local LLM, "gemini" = cloud API
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()

# Ollama settings (used when LLM_BACKEND=ollama)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
# Keep the model loaded in RAM between questions (big speed-up: no reload per query).
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
OLLAMA_NUM_CTX = _get_int("OLLAMA_NUM_CTX", 4096, minimum=512)

# Gemini settings (used when LLM_BACKEND=gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Knowledge base / retrieval
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
FAISS_INDEX_DIR = _get_path("FAISS_INDEX_DIR", "knowledge_base/faiss_index")
RAW_DOCS_DIR = _get_path("RAW_DOCS_DIR", "data/knowledge_base_docs")
CHUNK_SIZE = _get_int("CHUNK_SIZE", 800, minimum=100)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 120, minimum=0)
TOP_K = _get_int("TOP_K", 4, minimum=1)
EMBED_BATCH_SIZE = _get_int("EMBED_BATCH_SIZE", 32, minimum=1)
RESPONSE_CACHE_SIZE = _get_int("RESPONSE_CACHE_SIZE", 128, minimum=0)

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")
PLACEHOLDER_KEY = "your_gemini_api_key_here"

_SETTABLE = {
    "LLM_BACKEND": str, "OLLAMA_MODEL": str, "OLLAMA_BASE_URL": str,
    "OLLAMA_KEEP_ALIVE": str, "OLLAMA_NUM_CTX": int,
    "GEMINI_API_KEY": str, "GEMINI_MODEL": str,
    "EMBEDDING_MODEL": str, "CHUNK_SIZE": int, "CHUNK_OVERLAP": int, "TOP_K": int,
}


def gemini_key_is_set() -> bool:
    key = (GEMINI_API_KEY or "").strip()
    return bool(key) and key != PLACEHOLDER_KEY


def update_settings(updates: dict, persist: bool = False) -> None:
    """Validate and apply settings at runtime; optionally write them to .env."""
    clean = {}
    for key, value in updates.items():
        if key not in _SETTABLE:
            raise KeyError(f"Unknown setting: {key}")
        if _SETTABLE[key] is int:
            try:
                clean[key] = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a whole number.") from None
        else:
            clean[key] = str(value).strip()

    if "LLM_BACKEND" in clean:
        clean["LLM_BACKEND"] = clean["LLM_BACKEND"].lower()
        if clean["LLM_BACKEND"] not in ("ollama", "gemini"):
            raise ValueError("LLM_BACKEND must be 'ollama' or 'gemini'.")
    if "OLLAMA_BASE_URL" in clean:
        clean["OLLAMA_BASE_URL"] = clean["OLLAMA_BASE_URL"].rstrip("/")
    for key, minimum in (("CHUNK_SIZE", 100), ("TOP_K", 1), ("OLLAMA_NUM_CTX", 512), ("CHUNK_OVERLAP", 0)):
        if key in clean and clean[key] < minimum:
            raise ValueError(f"{key} must be at least {minimum}.")
    size = clean.get("CHUNK_SIZE", CHUNK_SIZE)
    overlap = clean.get("CHUNK_OVERLAP", CHUNK_OVERLAP)
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    module_globals = globals()
    for key, value in clean.items():
        module_globals[key] = value
        os.environ[key] = str(value)
    if persist:
        _write_env(clean)


def _write_env(values: dict) -> None:
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    remaining = dict(values)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in remaining:
            out.append(f"{key}={_quote(remaining.pop(key))}")
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={_quote(value)}")
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _quote(value) -> str:
    text = str(value)
    return f'"{text}"' if any(c in text for c in " #") else text
