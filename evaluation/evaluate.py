"""
Evaluation script: runs the Agentic-RAG pipeline and the traditional RAG
baseline over the same test set and reports the comparison metrics called
for in the project proposal (retrieval/citation proxies, response time).

Usage:
    python evaluation/evaluate.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.orchestrator import AgenticRAGPipeline
from src.baseline_rag import BaselineRAGPipeline

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_questions.json")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")


def keyword_overlap_score(response: str, keywords: list):
    if not keywords:
        return None
    resp_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
    return round(hits / len(keywords), 2)


def citation_present(response: str) -> bool:
    return "[Source:" in response


def load_questions(test_file: str = TEST_FILE) -> list:
    with open(test_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(test_file: str = TEST_FILE, output_csv: str = OUTPUT_CSV, agentic=None, baseline=None,
                   progress_cb=None, limit: int = None, verbose: bool = True) -> pd.DataFrame:
    """progress_cb(done, total, question) is called before each question (optional)."""
    test_cases = load_questions(test_file)
    if limit:
        test_cases = test_cases[:limit]
    agentic = agentic or AgenticRAGPipeline()
    baseline = baseline or BaselineRAGPipeline(kb=agentic.kb)

    rows = []
    for i, case in enumerate(test_cases, 1):
        q = case["question"]
        keywords = case.get("expected_keywords", [])
        if verbose:
            print(f"[{i}/{len(test_cases)}] {q}")
        if progress_cb:
            progress_cb(i - 1, len(test_cases), q)
        row = {"question": q}
        try:
            a = agentic.run(q, mode="full", use_cache=False)
            b = baseline.run(q)
            row.update({
                "agentic_response_time_s": a["response_time_sec"],
                "baseline_response_time_s": b["response_time_sec"],
                "agentic_keyword_score": keyword_overlap_score(a["response"], keywords),
                "baseline_keyword_score": keyword_overlap_score(b["response"], keywords),
                "agentic_citation_present": citation_present(a["response"]),
                "baseline_citation_present": citation_present(b["response"]),
                "agentic_verified": a["verified"],
                "agentic_num_sources": len(a["retrieved_sources"]),
                "baseline_num_sources": len(b["retrieved_sources"]),
                "error": "",
            })
        except Exception as exc:  # noqa: BLE001 - one failing question must not abort the run
            row["error"] = str(exc)[:300]
        rows.append(row)

    df = pd.DataFrame(rows)
    try:
        df.to_csv(output_csv, index=False)
    except OSError:
        pass
    if progress_cb:
        progress_cb(len(test_cases), len(test_cases), "")
    return df


def summarize(df: pd.DataFrame) -> dict:
    ok = df[df["error"] == ""] if "error" in df else df
    if ok.empty:
        return {}
    return {
        "Avg agentic keyword score": round(ok["agentic_keyword_score"].mean(), 2),
        "Avg baseline keyword score": round(ok["baseline_keyword_score"].mean(), 2),
        "Agentic citation rate": f"{ok['agentic_citation_present'].mean():.0%}",
        "Baseline citation rate": f"{ok['baseline_citation_present'].mean():.0%}",
        "Avg agentic time (s)": round(ok["agentic_response_time_s"].mean(), 2),
        "Avg baseline time (s)": round(ok["baseline_response_time_s"].mean(), 2),
    }


if __name__ == "__main__":
    result = run_evaluation()
    print("\n=== Summary ===")
    for key, value in summarize(result).items():
        print(f"{key}: {value}")
    failed = int((result["error"] != "").sum())
    if failed:
        print(f"\n{failed} question(s) failed - see the 'error' column in results.csv")
    print(f"\nDetailed results saved to {OUTPUT_CSV}")
