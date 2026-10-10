"""
Agentic-RAG orchestrator: wires the agents together into the pipeline described in the
project proposal:
    Query Optimizer -> Router -> (Knowledge Retrieval and/or Report Analysis)
    -> Response Generation -> Verification.

Two modes:
  "full" - every agent runs (up to 4-5 LLM calls): best quality, slowest.
  "fast" - skips the Query Optimizer, Router and Verification agents: retrieval from the
           trained index + ONE LLM call, streamed token by token. Much faster.

Repeated questions are answered instantly from an in-memory cache, and lab-report
analysis is cached per report, so the same report is never re-analysed.
"""
import copy
import hashlib
import threading
import time
from collections import OrderedDict

import config
from src.vectorstore import KnowledgeBase
from src.agents import query_optimizer, router, knowledge_retrieval, report_analysis, response_generation, verification

MODES = ("fast", "full")

_report_cache = OrderedDict()
_report_lock = threading.Lock()
_REPORT_CACHE_MAX = 32


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def analyze_report_cached(report_text: str) -> dict:
    """analyze_report() with a per-report cache (an LLM extraction call is expensive)."""
    key = _hash(report_text)
    with _report_lock:
        if key in _report_cache:
            _report_cache.move_to_end(key)
            return copy.deepcopy(_report_cache[key])
    result = report_analysis.analyze_report(report_text)
    if result:  # an empty result may just mean the LLM was busy/offline: do not remember it
        with _report_lock:
            _report_cache[key] = copy.deepcopy(result)
            while len(_report_cache) > _REPORT_CACHE_MAX:
                _report_cache.popitem(last=False)
    return result


def _emit(callback, key, message):
    if callback:
        callback(key, message)


class AgenticRAGPipeline:
    def __init__(self, kb=None):
        # Pass a shared KnowledgeBase to avoid loading the index twice (e.g. app + baseline).
        self.kb = kb or KnowledgeBase()
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def run(self, query: str, report_text: str = None, mode: str = "full", top_k: int = None,
            on_stage=None, on_token=None, use_cache: bool = True) -> dict:
        """on_stage(key, message) reports progress; on_token(piece) streams the draft answer."""
        t0 = time.time()
        query = (query or "").strip()
        if not query:
            raise ValueError("Please enter a question.")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        top_k = top_k or config.TOP_K
        has_report = bool(report_text and report_text.strip())

        cache_key = (mode, " ".join(query.lower().split()), _hash(report_text) if has_report else "", top_k)
        if use_cache and config.RESPONSE_CACHE_SIZE > 0:
            with self._lock:
                hit = self._cache.get(cache_key)
                if hit is not None:
                    self._cache.move_to_end(cache_key)
                    result = copy.deepcopy(hit)
                    result.update(cached=True, response_time_sec=round(time.time() - t0, 3))
                    return result

        full = mode == "full"
        if full:
            _emit(on_stage, "optimize", "Optimizing your question…")
            optimized_query = query_optimizer.optimize(query)
            _emit(on_stage, "route", "Choosing the processing path…")
            path = router.route(optimized_query, has_report)
        else:
            optimized_query = query
            path = "BOTH" if has_report else "KNOWLEDGE_ONLY"

        retrieved_chunks, report_interp, report_for_answer = [], None, None
        if path in ("KNOWLEDGE_ONLY", "BOTH"):
            _emit(on_stage, "retrieve", "Searching the trained knowledge base…")
            retrieved_chunks = knowledge_retrieval.retrieve(self.kb, optimized_query, k=top_k)

        if path in ("REPORT_ONLY", "BOTH") and has_report:
            _emit(on_stage, "analyze_report", "Analysing the lab report…")
            report_interp = analyze_report_cached(report_text)
            report_for_answer = report_text  # the answer step also sees the raw report text

        _emit(on_stage, "generate", "Writing the answer…")
        draft = response_generation.generate_response(
            optimized_query, retrieved_chunks, report_interp, on_token=on_token,
            report_text=report_for_answer,
        )

        if full:
            _emit(on_stage, "verify", "Verifying the answer against the sources…")
            # The verifier also sees the report interpretation, so lab-value statements are not
            # wrongly flagged as "unsupported" when no guideline chunks were retrieved.
            vr = verification.verify(draft, retrieved_chunks, report_interp, report_text=report_for_answer)
            final_response = vr.get("corrected_response", draft)
            verified, status, issues = vr.get("verified", False), vr.get("verification_status", "unknown"), vr.get("issues", [])
        else:
            final_response, verified, status, issues = draft, None, "skipped", []

        result = {
            "query": query,
            "mode": mode,
            "optimized_query": optimized_query,
            "route": path,
            "retrieved_sources": [c["source"] for c in retrieved_chunks],
            "retrieved_chunks": retrieved_chunks,
            "report_interpretation": report_interp,
            "response": final_response,
            "verified": verified,
            "verification_status": status,
            "verification_issues": issues,
            "cached": False,
            "response_time_sec": round(time.time() - t0, 2),
        }

        if use_cache and config.RESPONSE_CACHE_SIZE > 0 and status != "failed" and final_response:
            with self._lock:
                self._cache[cache_key] = copy.deepcopy(result)
                while len(self._cache) > config.RESPONSE_CACHE_SIZE:
                    self._cache.popitem(last=False)
        return result
