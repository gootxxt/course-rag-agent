from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    original_query: str
    query: str
    rewritten_query: str | None
    retry_count: int
    max_retries: int
    rewrite_history: list[dict[str, Any]]
    self_check_passed: bool | None
    self_check_reason: str | None
    unsupported_claims: list[str]
    citation_coverage: float | None
    groundedness_score: float | None
    generation_retry_count: int
    max_generation_retries: int
    intent: str
    tool_name: str | None
    tool_args: dict[str, Any]
    tool_result: dict[str, Any]
    retrieval_hits: list[dict[str, Any]]
    top_score: float | None
    hit_count: int
    compressed_context: str | None
    citation_map: dict[str, dict[str, Any]]
    relevance_passed: bool
    answer: str
    citations: list[dict[str, Any]]
    refused: bool
    reason: str | None
    tool_trace: list[dict[str, Any]]
