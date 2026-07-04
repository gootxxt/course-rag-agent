from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from course_rag_agent.agent_graph import run_agent  # noqa: E402
from course_rag_agent.config import settings  # noqa: E402
from course_rag_agent.knowledge_base import KnowledgeBase  # noqa: E402


DEFAULT_EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "qa_eval.jsonl"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "eval" / "reports"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return samples


def _source_matches(source: str, expected_sources: list[str]) -> bool:
    normalized = source.replace("\\", "/").lower()
    return any(expected.lower().replace("\\", "/") in normalized for expected in expected_sources)


def _hit_sources(state: dict[str, Any], top_k: int) -> list[str]:
    return [str(hit.get("source", "")) for hit in state.get("retrieval_hits", [])[:top_k]]


def _retrieval_recall_at_k(state: dict[str, Any], expected_sources: list[str], top_k: int) -> float | None:
    if not expected_sources:
        return None
    sources = _hit_sources(state, top_k)
    return 1.0 if any(_source_matches(source, expected_sources) for source in sources) else 0.0


def _retrieval_precision_at_k(state: dict[str, Any], expected_sources: list[str], top_k: int) -> float | None:
    if not expected_sources:
        return None
    sources = _hit_sources(state, top_k)
    if not sources:
        return 0.0
    relevant = sum(1 for source in sources if _source_matches(source, expected_sources))
    return relevant / min(top_k, len(sources))


def _citation_coverage(state: dict[str, Any]) -> float | None:
    citations = state.get("citations", [])
    if not citations:
        return None
    retrieval_keys = {
        (str(hit.get("source", "")), str(hit.get("chunk_id", "")))
        for hit in state.get("retrieval_hits", [])
    }
    if not retrieval_keys:
        return 0.0
    valid = sum(
        1
        for citation in citations
        if (str(citation.get("source", "")), str(citation.get("chunk_id", ""))) in retrieval_keys
    )
    return valid / len(citations)


def _keyword_hit(answer: str, expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None
    normalized = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in normalized)
    return hits / len(expected_keywords)


def _trace_summary(tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {
        "node",
        "action",
        "decision",
        "next_node",
        "reason",
        "latency_ms",
        "top_score",
        "hit_count",
        "fusion_strategy",
        "self_check_passed",
        "retry_count",
        "generation_retry_count",
    }
    return [{key: item[key] for key in keys if key in item} for item in tool_trace]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * 0.95))
    return sorted_values[index]


def _mean_optional(values: list[float | None]) -> float | None:
    actual = [value for value in values if value is not None]
    if not actual:
        return None
    return sum(actual) / len(actual)


def _has_rewrite(state: dict[str, Any]) -> bool:
    return any(item.get("node") == "rewrite_node" for item in state.get("tool_trace", []))


def _rewrite_success(state: dict[str, Any]) -> bool:
    if not _has_rewrite(state):
        return False
    if not state.get("refused", False):
        return True
    return any(
        item.get("node") == "relevance_check_node" and item.get("decision") == "compress"
        for item in state.get("tool_trace", [])
    )


def _case_failures(
    sample: dict[str, Any],
    state: dict[str, Any],
    *,
    recall: float | None,
    citation_coverage: float | None,
    keyword_score: float | None,
) -> list[str]:
    failures: list[str] = []
    should_refuse = bool(sample.get("should_refuse", False))
    refused = bool(state.get("refused", False))
    if refused != should_refuse:
        failures.append("refusal_mismatch")
    if recall == 0.0 and not should_refuse:
        failures.append("expected_source_not_recalled")
    if citation_coverage == 0.0 and not refused and sample.get("category") != "stats":
        failures.append("invalid_citations")
    if state.get("self_check_passed") is False and not refused:
        failures.append("self_check_failed")
    if keyword_score is not None and keyword_score < 0.5 and not refused:
        failures.append("answer_keywords_missing")
    return failures


def run_eval(eval_file: Path, report_dir: Path, top_k: int, limit: int | None = None) -> dict[str, Any]:
    samples = _read_jsonl(eval_file)
    if limit is not None:
        samples = samples[:limit]

    kb = KnowledgeBase(settings)
    case_results: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    latencies: list[float] = []
    recall_values: list[float | None] = []
    precision_values: list[float | None] = []
    citation_values: list[float | None] = []
    keyword_values: list[float | None] = []
    refusal_correct = 0
    answer_count = 0
    refusal_count = 0
    self_check_passes = 0
    self_check_total = 0
    rewrite_triggered = 0
    rewrite_successes = 0

    for sample in samples:
        start = time.perf_counter()
        state = dict(run_agent(sample["query"], kb, settings))
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        latencies.append(latency_ms)

        refused = bool(state.get("refused", False))
        if refused:
            refusal_count += 1
        else:
            answer_count += 1

        should_refuse = bool(sample.get("should_refuse", False))
        if refused == should_refuse:
            refusal_correct += 1

        expected_sources = list(sample.get("expected_sources", []))
        recall = _retrieval_recall_at_k(state, expected_sources, top_k)
        precision = _retrieval_precision_at_k(state, expected_sources, top_k)
        citation_cov = _citation_coverage(state)
        keyword_score = _keyword_hit(str(state.get("answer", "")), list(sample.get("expected_answer_keywords", [])))
        recall_values.append(recall)
        precision_values.append(precision)
        citation_values.append(citation_cov)
        keyword_values.append(keyword_score)

        if state.get("self_check_passed") is not None:
            self_check_total += 1
            if state.get("self_check_passed") is True:
                self_check_passes += 1

        if _has_rewrite(state):
            rewrite_triggered += 1
            if _rewrite_success(state):
                rewrite_successes += 1

        failures = _case_failures(
            sample,
            state,
            recall=recall,
            citation_coverage=citation_cov,
            keyword_score=keyword_score,
        )
        case_result = {
            "id": sample.get("id"),
            "category": sample.get("category"),
            "query": sample.get("query"),
            "should_refuse": should_refuse,
            "refused": refused,
            "reason": state.get("reason"),
            "recall_at_k": recall,
            "precision_at_k": precision,
            "citation_coverage": citation_cov,
            "keyword_score": keyword_score,
            "self_check_passed": state.get("self_check_passed"),
            "rewrite_triggered": _has_rewrite(state),
            "rewrite_success": _rewrite_success(state) if _has_rewrite(state) else None,
            "latency_ms": latency_ms,
            "top_score": state.get("top_score"),
            "answer_preview": str(state.get("answer", ""))[:300],
            "failures": failures,
        }
        case_results.append(case_result)

        if failures:
            failed_cases.append(
                {
                    "id": sample.get("id"),
                    "query": sample.get("query"),
                    "expected": {
                        "should_refuse": should_refuse,
                        "expected_sources": expected_sources,
                        "expected_answer_keywords": sample.get("expected_answer_keywords", []),
                    },
                    "actual": {
                        "refused": refused,
                        "reason": state.get("reason"),
                        "answer_preview": str(state.get("answer", ""))[:500],
                        "recall_at_k": recall,
                        "precision_at_k": precision,
                        "citation_coverage": citation_cov,
                        "self_check_passed": state.get("self_check_passed"),
                    },
                    "failures": failures,
                    "tool_trace": _trace_summary(state.get("tool_trace", [])),
                }
            )

    total = len(samples)
    report = {
        "config": {
            "eval_file": str(eval_file),
            "top_k": top_k,
            "fusion_strategy": settings.fusion_strategy,
            "score_threshold": settings.score_threshold,
            "max_retries": settings.max_retries,
            "max_generation_retries": settings.max_generation_retries,
        },
        "summary": {
            "total": total,
            "answer_count": answer_count,
            "refusal_count": refusal_count,
            "refusal_accuracy": refusal_correct / total if total else 0.0,
            "recall_at_k": _mean_optional(recall_values),
            "precision_at_k": _mean_optional(precision_values),
            "citation_coverage": _mean_optional(citation_values),
            "answer_keyword_hit_rate": _mean_optional(keyword_values),
            "self_check_pass_rate": self_check_passes / self_check_total if self_check_total else None,
            "rewrite_trigger_rate": rewrite_triggered / total if total else 0.0,
            "rewrite_success_rate": rewrite_successes / rewrite_triggered if rewrite_triggered else None,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "p95_latency_ms": _p95(latencies),
            "failed_case_count": len(failed_cases),
        },
        "cases": case_results,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "eval_report.json"
    failed_path = report_dir / "failed_cases.jsonl"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with failed_path.open("w", encoding="utf-8") as f:
        for item in failed_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return report


def _print_summary(report: dict[str, Any], report_dir: Path) -> None:
    summary = report["summary"]
    print("Eval Summary")
    print("============")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print()
    print("Saved:")
    print(f"- {report_dir / 'eval_report.json'}")
    print(f"- {report_dir / 'failed_cases.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local Agentic RAG system.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    report = run_eval(args.eval_file, args.report_dir, args.top_k, args.limit)
    _print_summary(report, args.report_dir)


if __name__ == "__main__":
    main()
