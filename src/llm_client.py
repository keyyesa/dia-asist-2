"""
LLM wrapper used by every agent in the pipeline. Two backends, selected via LLM_BACKEND:

  - "ollama" (default): fully offline, calls a local Ollama server. No API key and no
    internet needed once the model is pulled.
  - "gemini": cloud API, needs GEMINI_API_KEY.

Speed features: the Ollama model is kept loaded in memory between questions
(OLLAMA_KEEP_ALIVE), a persistent HTTP session is reused, and generate_stream() yields
tokens as they are produced so the UI can show the answer immediately.
"""
import json
import time

import requests

import config

_session = requests.Session()
_TIMEOUT = (5, 600)  # (connect, read) seconds


class LLMError(RuntimeError):
    """Raised for any LLM backend failure, with a human-readable fix-it message."""


def _ollama_payload(prompt, system_instruction, temperature, stream):
    return {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_instruction or "",
        "stream": stream,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_ctx": config.OLLAMA_NUM_CTX},
    }


def _ollama_fix_hint() -> str:
    return (f"Checklist: is Ollama running (`ollama serve`)? Have you pulled the model "
            f"(`ollama pull {config.OLLAMA_MODEL}`)? Is OLLAMA_BASE_URL ({config.OLLAMA_BASE_URL}) correct?")


def _ollama_error(resp) -> LLMError:
    try:
        detail = resp.json().get("error") or resp.text
    except ValueError:
        detail = resp.text
    return LLMError(f"Ollama returned HTTP {resp.status_code}: {str(detail)[:300]}\n{_ollama_fix_hint()}")


def _generate_ollama(prompt, system_instruction, temperature, max_retries):
    url = f"{config.OLLAMA_BASE_URL}/api/generate"
    payload = _ollama_payload(prompt, system_instruction, temperature, stream=False)
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = _session.post(url, json=payload, timeout=_TIMEOUT)
        except requests.ConnectionError as exc:  # Ollama is not running: retrying will not help
            raise LLMError(f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. {_ollama_fix_hint()}") from exc
        except requests.RequestException as exc:
            last_error = exc
        else:
            if resp.status_code == 200:
                return (resp.json().get("response") or "").strip()
            if resp.status_code in (400, 404):  # bad request / model not pulled: permanent
                raise _ollama_error(resp)
            last_error = _ollama_error(resp)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    raise LLMError(f"Ollama call failed after {max_retries} attempts: {last_error}\n{_ollama_fix_hint()}")


def _stream_ollama(prompt, system_instruction, temperature):
    url = f"{config.OLLAMA_BASE_URL}/api/generate"
    payload = _ollama_payload(prompt, system_instruction, temperature, stream=True)
    try:
        resp = _session.post(url, json=payload, timeout=_TIMEOUT, stream=True)
    except requests.ConnectionError as exc:
        raise LLMError(f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. {_ollama_fix_hint()}") from exc
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request failed: {exc}\n{_ollama_fix_hint()}") from exc
    with resp:
        if resp.status_code != 200:
            raise _ollama_error(resp)
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if data.get("error"):
                raise LLMError(f"Ollama error: {data['error']}\n{_ollama_fix_hint()}")
            piece = data.get("response") or ""
            if piece:
                yield piece
            if data.get("done"):
                break


def _gemini_model(system_instruction):
    if not config.gemini_key_is_set():
        raise LLMError("GEMINI_API_KEY is not set. Add it in Settings (or .env), or switch to the Ollama backend.")
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise LLMError("google-generativeai is not installed: pip install google-generativeai") from exc
    genai.configure(api_key=config.GEMINI_API_KEY)
    return genai.GenerativeModel(config.GEMINI_MODEL, system_instruction=system_instruction)


def _generate_gemini(prompt, system_instruction, temperature, max_retries):
    model = _gemini_model(system_instruction)
    last_error = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, generation_config={"temperature": temperature})
            return (response.text or "").strip() if response else ""
        except Exception as exc:  # noqa: BLE001 - includes blocked-content ValueError
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise LLMError(f"Gemini API call failed after {max_retries} attempts: {last_error}")


def _stream_gemini(prompt, system_instruction, temperature):
    model = _gemini_model(system_instruction)
    try:
        for chunk in model.generate_content(prompt, generation_config={"temperature": temperature}, stream=True):
            try:
                text = chunk.text
            except ValueError:  # chunk without text (e.g. safety block)
                continue
            if text:
                yield text
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Gemini streaming failed: {exc}") from exc


def generate(prompt: str, system_instruction: str = None, temperature: float = 0.3,
             max_retries: int = 3) -> str:
    if config.LLM_BACKEND == "ollama":
        return _generate_ollama(prompt, system_instruction, temperature, max_retries)
    if config.LLM_BACKEND == "gemini":
        return _generate_gemini(prompt, system_instruction, temperature, max_retries)
    raise LLMError(f"Unknown LLM_BACKEND '{config.LLM_BACKEND}'. Use 'ollama' or 'gemini'.")


def generate_stream(prompt: str, system_instruction: str = None, temperature: float = 0.3):
    """Yield the answer piece by piece as the model produces it."""
    if config.LLM_BACKEND == "ollama":
        yield from _stream_ollama(prompt, system_instruction, temperature)
    elif config.LLM_BACKEND == "gemini":
        yield from _stream_gemini(prompt, system_instruction, temperature)
    else:
        raise LLMError(f"Unknown LLM_BACKEND '{config.LLM_BACKEND}'. Use 'ollama' or 'gemini'.")


def list_ollama_models(timeout: float = 2.0):
    """Names of locally installed Ollama models, or None if the server is unreachable."""
    try:
        resp = _session.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
    except (requests.RequestException, ValueError):
        return None


def _model_installed(wanted: str, installed: list) -> bool:
    names = set(installed)
    return wanted in names or (":" not in wanted and f"{wanted}:latest" in names)


def check_connection() -> dict:
    """Quick health check -> {ok, backend, model, detail, models}."""
    if config.LLM_BACKEND == "ollama":
        models = list_ollama_models()
        if models is None:
            return {"ok": False, "backend": "ollama", "model": config.OLLAMA_MODEL, "models": [],
                    "detail": f"Ollama is not reachable at {config.OLLAMA_BASE_URL}. Start it with `ollama serve`."}
        if not _model_installed(config.OLLAMA_MODEL, models):
            return {"ok": False, "backend": "ollama", "model": config.OLLAMA_MODEL, "models": models,
                    "detail": f"Model '{config.OLLAMA_MODEL}' is not installed. Run `ollama pull {config.OLLAMA_MODEL}`."}
        return {"ok": True, "backend": "ollama", "model": config.OLLAMA_MODEL, "models": models,
                "detail": "Ollama is running and the model is installed."}
    if config.LLM_BACKEND == "gemini":
        ok = config.gemini_key_is_set()
        return {"ok": ok, "backend": "gemini", "model": config.GEMINI_MODEL, "models": [],
                "detail": "API key configured (not verified)." if ok else "GEMINI_API_KEY is not set."}
    return {"ok": False, "backend": config.LLM_BACKEND, "model": "", "models": [],
            "detail": f"Unknown LLM_BACKEND '{config.LLM_BACKEND}'."}


def test_generation() -> str:
    """Tiny real generation (also pre-loads the model into memory). Raises LLMError on failure."""
    return generate("Reply with the single word: ready", temperature=0.0, max_retries=1)
