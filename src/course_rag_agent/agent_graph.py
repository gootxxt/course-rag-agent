from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from .agent_state import AgentState
from .config import Settings, settings
from .knowledge_base import KnowledgeBase
from .tools import ToolExecutor


STATS_KEYWORDS = (
    "统计",
    "多少文档",
    "多少资料",
    "多少chunk",
    "知识库",
    "stats",
    "documents",
    "chunks",
)
DEFAULT_CONTEXT_BUDGET = 4500


def _classify_intent(query: str) -> str:
    normalized = query.strip().lower()
    if not normalized:
        return "unknown"
    if any(keyword in normalized for keyword in STATS_KEYWORDS):
        return "stats"
    return "ask"


def _next_after_intent(intent: str) -> str:
    if intent == "stats":
        return "stats"
    if intent == "ask":
        return "retrieve"
    return "refuse"


def _route_after_intent(state: AgentState) -> str:
    return _next_after_intent(state.get("intent", "unknown"))


def _next_after_relevance(relevance_passed: bool, refused: bool) -> str:
    if relevance_passed and not refused:
        return "compress"
    return "final"


def _route_after_relevance(state: AgentState) -> str:
    return _next_after_relevance(
        bool(state.get("relevance_passed", False)),
        bool(state.get("refused", False)),
    )


def _trace(
    state: AgentState,
    *,
    node: str,
    action: str,
    latency_ms: float | None = None,
    **extra: Any,
) -> list[dict[str, Any]]:
    trace = list(state.get("tool_trace", []))
    item: dict[str, Any] = {"node": node, "action": action}
    if latency_ms is not None:
        item["latency_ms"] = latency_ms
    item.update({key: value for key, value in extra.items() if value is not None})
    trace.append(item)
    return trace


def _summarize_query(query: str, limit: int = 80) -> str:
    query = query.replace("\n", " ").strip()
    return query if len(query) <= limit else query[:limit] + "..."


def _build_citation(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": hit.get("source", ""),
        "doc_id": hit.get("doc_id", ""),
        "chunk_id": hit.get("chunk_id", ""),
        "chunk_index": int(hit.get("chunk_index", 0)),
        "start_pos": int(hit.get("start_pos", 0)),
        "end_pos": int(hit.get("end_pos", 0)),
    }


def _compress_hits(
    hits: list[dict[str, Any]],
    *,
    char_budget: int = DEFAULT_CONTEXT_BUDGET,
) -> tuple[str, list[dict[str, Any]], int]:
    sorted_hits = sorted(hits, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    seen: set[str] = set()
    context_blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    used_chars = 0

    for hit in sorted_hits:
        chunk_id = str(hit.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)

        text = str(hit.get("text", "")).strip()
        if not text:
            continue

        citation = _build_citation(hit)
        header = (
            f"[{len(citations) + 1}] source={citation['source']} "
            f"chunk={citation['chunk_index']} pos={citation['start_pos']}-{citation['end_pos']} "
            f"score={float(hit.get('score', 0.0)):.4f}"
        )
        block = f"{header}\n{text}"
        remaining = char_budget - used_chars
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining]

        context_blocks.append(block)
        citations.append(citation)
        used_chars += len(block)

    return "\n\n".join(context_blocks), citations, len(citations)


def _fallback_answer_from_context(query: str, context: str, citations: list[dict[str, Any]]) -> str:
    lines = [
        "LLM is not configured. The following answer is an evidence summary from retrieved context.",
        f"Question: {query}",
        "",
    ]
    preview = context.replace("\n", " ").strip()
    if len(preview) > 800:
        preview = preview[:800] + "..."
    lines.append(preview)
    if citations:
        lines.append("")
        lines.append("Citations:")
        for citation in citations:
            lines.append(
                f"- {citation['source']}#{citation['chunk_index']}:"
                f"{citation['start_pos']}-{citation['end_pos']}"
            )
    return "\n".join(lines)


def _generate_answer_from_context(
    query: str,
    context: str,
    citations: list[dict[str, Any]],
    app_settings: Settings,
) -> str:
    if not app_settings.openai_api_key:
        return _fallback_answer_from_context(query, context, citations)

    client = OpenAI(api_key=app_settings.openai_api_key, base_url=app_settings.openai_base_url)
    citation_lines = [
        f"[{i}] {item['source']}#{item['chunk_index']}:{item['start_pos']}-{item['end_pos']}"
        for i, item in enumerate(citations, start=1)
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a course-material QA assistant. Answer only from the provided context. "
                "If the context is insufficient, say the materials are insufficient. "
                "Cite evidence using bracket ids like [1], [2]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                f"Available citations:\n" + "\n".join(citation_lines)
            ),
        },
    ]
    response = client.chat.completions.create(
        model=app_settings.openai_model,
        messages=messages,
        temperature=app_settings.llm_temperature,
    )
    return response.choices[0].message.content or ""


def build_agent_graph(kb: KnowledgeBase | None = None, app_settings: Settings = settings):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is required. Run: pip install -r requirements.txt") from exc

    executor = ToolExecutor(kb, app_settings)

    def intent_node(state: AgentState) -> AgentState:
        start = time.perf_counter()
        query = state.get("query", "")
        intent = _classify_intent(query)
        next_node = _next_after_intent(intent)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "intent": intent,
            "tool_trace": _trace(
                state,
                node="intent_node",
                action="classify_intent",
                input_summary=_summarize_query(query),
                decision=intent,
                next_node=next_node,
                reason=f"intent={intent}",
                latency_ms=latency_ms,
            ),
        }

    def stats_node(state: AgentState) -> AgentState:
        start = time.perf_counter()
        result = executor.execute("stats", {})
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "tool_name": "stats",
            "tool_args": {},
            "tool_result": result,
            "tool_trace": _trace(
                state,
                node="stats_node",
                action="stats",
                next_node="final",
                output_summary={
                    "documents": result.get("documents", 0),
                    "chunks": result.get("chunks", 0),
                    "bm25_chunks": result.get("bm25_chunks", 0),
                },
                latency_ms=latency_ms,
            ),
        }

    def refuse_node(state: AgentState) -> AgentState:
        return {
            "tool_name": None,
            "tool_args": {},
            "answer": "Current Agent supports course-material QA and knowledge-base stats only.",
            "citations": [],
            "refused": True,
            "reason": "unsupported_intent",
            "tool_trace": _trace(
                state,
                node="refuse_node",
                action="refuse",
                decision="unsupported_intent",
                next_node="final",
                reason=f"intent={state.get('intent', 'unknown')}",
            ),
        }

    def retrieve_node(state: AgentState) -> AgentState:
        query = state.get("query", "")
        tool_args = {"query": query, "top_k": app_settings.top_k, "mode": "hybrid"}
        start = time.perf_counter()
        result = executor.execute("rag_search", tool_args)
        hits = result.get("hits", [])
        hit_count = len(hits)
        top_score = max((float(hit.get("score", 0.0)) for hit in hits), default=None)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "tool_name": "rag_search",
            "tool_args": tool_args,
            "tool_result": result,
            "retrieval_hits": hits,
            "hit_count": hit_count,
            "top_score": top_score,
            "tool_trace": _trace(
                state,
                node="retrieve_node",
                action="rag_search",
                next_node="relevance_check",
                input_summary={"query": _summarize_query(query), **tool_args},
                hit_count=hit_count,
                top_score=top_score,
                latency_ms=latency_ms,
            ),
        }

    def relevance_check_node(state: AgentState) -> AgentState:
        start = time.perf_counter()
        hits = state.get("retrieval_hits", [])
        hit_count = int(state.get("hit_count", len(hits)))
        top_score = state.get("top_score")
        threshold = app_settings.score_threshold
        passed = hit_count > 0 and top_score is not None and float(top_score) >= threshold
        next_node = _next_after_relevance(passed, not passed)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if not passed:
            return {
                "relevance_passed": False,
                "refused": True,
                "reason": "low_relevance",
                "answer": "I could not find enough relevant evidence in the local knowledge base.",
                "citations": [],
                "tool_trace": _trace(
                    state,
                    node="relevance_check_node",
                    action="check_relevance",
                    decision="refuse",
                    next_node=next_node,
                    reason=f"top_score={top_score}, threshold={threshold}, hit_count={hit_count}",
                    hit_count=hit_count,
                    top_score=top_score,
                    threshold=threshold,
                    latency_ms=latency_ms,
                ),
            }

        return {
            "relevance_passed": True,
            "refused": False,
            "reason": None,
            "tool_trace": _trace(
                state,
                node="relevance_check_node",
                action="check_relevance",
                decision="compress",
                next_node=next_node,
                reason=f"top_score={top_score} >= threshold={threshold}",
                hit_count=hit_count,
                top_score=top_score,
                threshold=threshold,
                latency_ms=latency_ms,
            ),
        }

    def compress_node(state: AgentState) -> AgentState:
        start = time.perf_counter()
        context, citations, kept_count = _compress_hits(state.get("retrieval_hits", []))
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "compressed_context": context,
            "citations": citations,
            "tool_trace": _trace(
                state,
                node="compress_node",
                action="compress_context",
                next_node="generate",
                input_summary={"hit_count": state.get("hit_count", 0)},
                output_summary={"kept_count": kept_count, "context_chars": len(context)},
                latency_ms=latency_ms,
            ),
        }

    def generate_node(state: AgentState) -> AgentState:
        context = state.get("compressed_context") or ""
        citations = state.get("citations", [])
        if not context.strip():
            return {
                "answer": "I could not build enough context from the retrieved evidence.",
                "refused": True,
                "reason": "empty_context",
                "tool_trace": _trace(
                    state,
                    node="generate_node",
                    action="skip_generate",
                    decision="refuse",
                    next_node="final",
                    reason="empty_context",
                ),
            }

        start = time.perf_counter()
        answer = _generate_answer_from_context(state.get("query", ""), context, citations, app_settings)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "answer": answer,
            "refused": False,
            "reason": None,
            "tool_trace": _trace(
                state,
                node="generate_node",
                action="generate_answer",
                next_node="final",
                input_summary={"context_chars": len(context), "citation_count": len(citations)},
                output_summary={"answer_chars": len(answer)},
                latency_ms=latency_ms,
            ),
        }

    def final_node(state: AgentState) -> AgentState:
        start = time.perf_counter()
        intent = state.get("intent")
        result = state.get("tool_result", {})

        if intent == "stats":
            answer = (
                f"当前知识库共有 {result.get('documents', 0)} 个文档、"
                f"{result.get('chunks', 0)} 个 chunks，BM25 索引中有 "
                f"{result.get('bm25_chunks', 0)} 个 chunks。"
            )
            refused = False
            reason = None
            citations: list[dict[str, Any]] = []
        else:
            answer = state.get("answer", "")
            refused = bool(state.get("refused", False))
            reason = state.get("reason")
            citations = state.get("citations", [])

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "answer": answer,
            "citations": citations,
            "refused": refused,
            "reason": reason,
            "tool_trace": _trace(
                state,
                node="final_node",
                action="finalize",
                decision="refused" if refused else "answered",
                reason=reason,
                latency_ms=latency_ms,
            ),
        }

    graph = StateGraph(AgentState)
    graph.add_node("intent", intent_node)
    graph.add_node("stats", stats_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("relevance_check", relevance_check_node)
    graph.add_node("compress", compress_node)
    graph.add_node("generate", generate_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("intent")
    graph.add_conditional_edges(
        "intent",
        _route_after_intent,
        {
            "stats": "stats",
            "retrieve": "retrieve",
            "refuse": "refuse",
        },
    )
    graph.add_edge("stats", "final")
    graph.add_edge("refuse", "final")
    graph.add_edge("retrieve", "relevance_check")
    graph.add_conditional_edges(
        "relevance_check",
        _route_after_relevance,
        {
            "compress": "compress",
            "final": "final",
        },
    )
    graph.add_edge("compress", "generate")
    graph.add_edge("generate", "final")
    graph.add_edge("final", END)
    return graph.compile()


def run_agent(query: str, kb: KnowledgeBase | None = None, app_settings: Settings = settings) -> AgentState:
    graph = build_agent_graph(kb, app_settings)
    return graph.invoke({"query": query, "tool_trace": []})

