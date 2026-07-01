from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    tool_name: str | None
    tool_args: dict[str, Any]
    tool_result: dict[str, Any]
    retrieval_hits: list[dict[str, Any]]
    top_score: float | None
    hit_count: int
    compressed_context: str | None
    relevance_passed: bool
    answer: str
    citations: list[dict[str, Any]]
    refused: bool
    reason: str | None
    tool_trace: list[dict[str, Any]]
