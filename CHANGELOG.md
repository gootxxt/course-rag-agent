# Changelog

## Unreleased

Implemented:

- Added query rewrite node for low-relevance retrieval.
- Added retry loop from `relevance_check` to `rewrite` back to `retrieve`.
- Added `retry_count`, `max_retries`, `rewritten_query`, and `rewrite_history` state fields.
- Added loop guard so low-relevance queries are refused after retry limit.
- Added Phase 3 documentation and demo commands.
- Added rule-based self-check node after answer generation.
- Added self-check routing for pass, regenerate, and final refusal.
- Added self-check state fields and unit tests for citation validation.
- Added configurable hybrid retrieval fusion strategies.
- Added Reciprocal Rank Fusion support.
- Standardized agent retrieval hits with vector, BM25, final score, and rank fields.
- Upgraded context compression with citation map and compression diagnostics.
- Added local evaluation dataset and script for retrieval, refusal, citation, self-check, rewrite, and latency metrics.

## v0.4.0 - LangGraph Conditional Routing Agent

Current local milestone.

Implemented:

- Local RAG knowledge base for txt, md, and pdf files.
- Chroma vector store with metadata and chunk deduplication.
- BM25 keyword retrieval.
- Hybrid retrieval with vector and BM25 score fusion.
- OpenAI-compatible LLM generation.
- CLI commands for import, update, delete, stats, ask, and agent demo.
- FastAPI endpoints for document operations, ask, stats, and agent execution.
- ToolExecutor layer with function-calling style tool schemas.
- LangGraph AgentState.
- Phase 1 visible RAG workflow: retrieve, relevance check, compress, generate, final.
- Phase 2 conditional routing with `add_conditional_edges`.
- Structured `tool_trace` with node decisions and next-node records.

Not yet implemented:

- Query rewrite.
- Retry loop.
- LLM self-check.
- Benchmark and ablation experiments.

## Suggested Future Versions

- `v0.5.0`: query rewrite and retry loop.
- `v0.6.0`: self-check node.
- `v0.7.0`: benchmark, Recall@K, Precision@K, refusal rate, latency metrics.
- `v1.0.0`: resume-ready final version with README, architecture docs, and interview notes.
