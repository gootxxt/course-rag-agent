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


def _next_after_relevance(relevance_passed: bool, refused: bool, retry_count: int, max_retries: int) -> str:
    if relevance_passed and not refused:
        return "compress"
    if not refused and retry_count < max_retries:
        return "rewrite"
    return "final"


def _route_after_relevance(state: AgentState) -> str:
    return _next_after_relevance(
        bool(state.get("relevance_passed", False)),
        bool(state.get("refused", False)),
        int(state.get("retry_count", 0)),
        int(state.get("max_retries", settings.max_retries)),
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


def _summarize_hits_for_rewrite(hits: list[dict[str, Any]], limit: int = 3) -> str:
    summaries: list[str] = []
    for hit in hits[:limit]:
        text = str(hit.get("text", "")).replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:220] + "..."
        summaries.append(
            f"- source={hit.get('source', '')}, score={float(hit.get('score', 0.0)):.4f}, text={text}"
        )
    return "\n".join(summaries) if summaries else "No useful retrieval hits."


def _clean_rewritten_query(text: str, fallback_query: str) -> str:
    cleaned = text.strip().strip('"').strip("'").replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned or fallback_query


def _fallback_rewrite_query(original_query: str, old_query: str, hits: list[dict[str, Any]]) -> str:
    terms: list[str] = []
    for hit in hits[:2]:
        source = str(hit.get("source", "")).replace("_", " ")
        for token in source.replace(".", " ").replace("-", " ").split():
            if len(token) >= 4 and token.lower() not in {item.lower() for item in terms}:
                terms.append(token)
            if len(terms) >= 4:
                break
        if len(terms) >= 4:
            break

    base_query = original_query or old_query
    if not terms:
        return base_query
    return f"{base_query} {' '.join(terms)}"


def _rewrite_query(
    *,
    original_query: str,
    old_query: str,
    hits: list[dict[str, Any]],
    reason: str,
    retry_count: int,
    app_settings: Settings,
) -> str:
    if not app_settings.openai_api_key:
        return _fallback_rewrite_query(original_query, old_query, hits)

    client = OpenAI(api_key=app_settings.openai_api_key, base_url=app_settings.openai_base_url)
    hit_summary = _summarize_hits_for_rewrite(hits)
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite user questions into better search queries for a local course-material RAG system. "
                "Preserve the user's original intent. Do not broaden the question. "
                "Prefer concrete technical terms, aliases, and likely document keywords. "
                "Return only one rewritten query, without explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{original_query}\n\n"
                f"Current search query:\n{old_query}\n\n"
                f"Failure reason:\n{reason}\n\n"
                f"Retry attempt:\n{retry_count + 1}\n\n"
                f"Low-relevance retrieval summary:\n{hit_summary}\n\n"
                "Rewrite the search query. Keep it specific and faithful to the original question."
            ),
        },
    ]
    response = client.chat.completions.create(
        model=app_settings.openai_model,
        messages=messages,
        temperature=0.0,
    )
    content = response.choices[0].message.content or ""
    return _clean_rewritten_query(content, old_query)


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
        query = state.get("original_query") or state.get("query", "")
        intent = _classify_intent(query)
        next_node = _next_after_intent(intent)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "original_query": query,
            "query": state.get("query", query),
            "retry_count": int(state.get("retry_count", 0)),
            "max_retries": int(state.get("max_retries", app_settings.max_retries)),
            "rewrite_history": list(state.get("rewrite_history", [])),
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
        query = state.get("rewritten_query") or state.get("query") or state.get("original_query", "")
        retry_count = int(state.get("retry_count", 0))
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
            "query": query,
            "tool_trace": _trace(
                state,
                node="retrieve_node",
                action="rag_search",
                next_node="relevance_check",
                input_summary={"query": _summarize_query(query), **tool_args},
                hit_count=hit_count,
                top_score=top_score,
                retry_count=retry_count,
                latency_ms=latency_ms,
            ),
        }

    def relevance_check_node(state: AgentState) -> AgentState:
        start = time.perf_counter()
        hits = state.get("retrieval_hits", [])
        hit_count = int(state.get("hit_count", len(hits)))
        top_score = state.get("top_score")
        threshold = app_settings.score_threshold
        retry_count = int(state.get("retry_count", 0))
        max_retries = int(state.get("max_retries", app_settings.max_retries))
        passed = hit_count > 0 and top_score is not None and float(top_score) >= threshold
        should_refuse = (not passed) and retry_count >= max_retries
        next_node = _next_after_relevance(passed, should_refuse, retry_count, max_retries)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if not passed:
            reason = "low_relevance_after_retry" if should_refuse else "low_relevance"
            answer = (
                "I could not find enough relevant evidence in the local knowledge base after query rewrite."
                if should_refuse
                else ""
            )
            return {
                "relevance_passed": False,
                "refused": should_refuse,
                "reason": reason,
                "answer": answer,
                "citations": [],
                "tool_trace": _trace(
                    state,
                    node="relevance_check_node",
                    action="check_relevance",
                    decision="refuse" if should_refuse else "rewrite",
                    next_node=next_node,
                    reason=f"top_score={top_score}, threshold={threshold}, hit_count={hit_count}, retry_count={retry_count}/{max_retries}",
                    hit_count=hit_count,
                    top_score=top_score,
                    threshold=threshold,
                    retry_count=retry_count,
                    max_retries=max_retries,
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
                retry_count=retry_count,
                max_retries=max_retries,
                latency_ms=latency_ms,
            ),
        }

    def rewrite_node(state: AgentState) -> AgentState:
        start = time.perf_counter()
        original_query = state.get("original_query") or state.get("query", "")
        old_query = state.get("query") or original_query
        retry_count = int(state.get("retry_count", 0))
        max_retries = int(state.get("max_retries", app_settings.max_retries))
        reason = state.get("reason") or "low_relevance"
        hits = state.get("retrieval_hits", [])

        new_query = _rewrite_query(
            original_query=original_query,
            old_query=old_query,
            hits=hits,
            reason=reason,
            retry_count=retry_count,
            app_settings=app_settings,
        )
        new_retry_count = retry_count + 1
        history = list(state.get("rewrite_history", []))
        history.append(
            {
                "old_query": old_query,
                "new_query": new_query,
                "reason": reason,
                "retry_count": new_retry_count,
                "top_score": state.get("top_score"),
                "hit_count": state.get("hit_count", 0),
            }
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "query": new_query,
            "rewritten_query": new_query,
            "retry_count": new_retry_count,
            "max_retries": max_retries,
            "rewrite_history": history,
            "refused": False,
            "reason": "rewritten_query",
            "tool_trace": _trace(
                state,
                node="rewrite_node",
                action="rewrite_query",
                decision="retry_retrieve",
                next_node="retrieve",
                old_query=_summarize_query(old_query),
                new_query=_summarize_query(new_query),
                reason=reason,
                retry_count=new_retry_count,
                max_retries=max_retries,
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
        answer = _generate_answer_from_context(
            state.get("original_query") or state.get("query", ""),
            context,
            citations,
            app_settings,
        )
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
    graph.add_node("rewrite", rewrite_node)
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
            "rewrite": "rewrite",
            "final": "final",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("compress", "generate")
    graph.add_edge("generate", "final")
    graph.add_edge("final", END)
    return graph.compile()


def run_agent(query: str, kb: KnowledgeBase | None = None, app_settings: Settings = settings) -> AgentState:
    graph = build_agent_graph(kb, app_settings)
    return graph.invoke(
        {
            "original_query": query,
            "query": query,
            "retry_count": 0,
            "max_retries": max(0, app_settings.max_retries),
            "rewrite_history": [],
            "tool_trace": [],
        }
    )
