from __future__ import annotations

from .bm25_store import BM25Store
from .embeddings import EmbeddingModel
from .schemas import RetrievalHit
from .vector_store import ChromaVectorStore


def _bm25_confidence(score: float) -> float:
    score = max(score, 0.0)
    return score / (score + 3.0) if score > 0 else 0.0


class HybridRetriever:
    def __init__(
        self,
        embeddings: EmbeddingModel,
        vector_store: ChromaVectorStore,
        bm25_store: BM25Store,
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        vector_weight: float = 0.65,
        score_threshold: float = 0.25,
        mode: str = "hybrid",
    ) -> list[RetrievalHit]:
        mode = mode.lower()
        vector_hits: list[RetrievalHit] = []
        bm25_hits: list[RetrievalHit] = []

        if mode in {"vector", "hybrid"}:
            query_embedding = self.embeddings.encode([query])[0]
            vector_hits = self.vector_store.search(query_embedding, candidate_k)
        if mode in {"bm25", "hybrid"}:
            bm25_hits = self.bm25_store.search(query, candidate_k)

        if mode == "vector":
            return [hit for hit in vector_hits[:top_k] if hit.score >= score_threshold]
        if mode == "bm25":
            for hit in bm25_hits:
                hit.score = _bm25_confidence(hit.bm25_score)
            return [hit for hit in bm25_hits[:top_k] if hit.score >= score_threshold]

        by_id: dict[str, RetrievalHit] = {}
        vector_scores = {hit.chunk_id: hit.vector_score for hit in vector_hits}
        bm25_scores = {hit.chunk_id: hit.bm25_score for hit in bm25_hits}

        for hit in vector_hits + bm25_hits:
            if hit.chunk_id not in by_id:
                by_id[hit.chunk_id] = hit
        for chunk_id, hit in by_id.items():
            hit.vector_score = vector_scores.get(chunk_id, 0.0)
            hit.bm25_score = bm25_scores.get(chunk_id, 0.0)
            hit.score = (
                vector_weight * hit.vector_score
                + (1.0 - vector_weight) * _bm25_confidence(hit.bm25_score)
            )

        ranked = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
        return [hit for hit in ranked[:top_k] if hit.score >= score_threshold]
