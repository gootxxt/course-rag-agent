from __future__ import annotations

from .bm25_store import BM25Store
from .embeddings import EmbeddingModel
from .schemas import RetrievalHit
from .vector_store import ChromaVectorStore


def _bm25_confidence(score: float) -> float:
    score = max(score, 0.0)
    return score / (score + 3.0) if score > 0 else 0.0


def _clone_hit(hit: RetrievalHit) -> RetrievalHit:
    return hit.model_copy(deep=True)


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
        self.last_diagnostics: dict[str, int | str] = {}

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        vector_weight: float = 0.65,
        fusion_strategy: str = "weighted_score_fusion",
        rrf_k: int = 60,
        score_threshold: float = 0.25,
        mode: str = "hybrid",
    ) -> list[RetrievalHit]:
        mode = mode.lower()
        fusion_strategy = fusion_strategy.lower()
        vector_hits: list[RetrievalHit] = []
        bm25_hits: list[RetrievalHit] = []

        if mode in {"vector", "hybrid"}:
            query_embedding = self.embeddings.encode([query])[0]
            vector_hits = self.vector_store.search(query_embedding, candidate_k)
        if mode in {"bm25", "hybrid"}:
            bm25_hits = self.bm25_store.search(query, candidate_k)

        if mode == "vector":
            ranked = []
            for rank, hit in enumerate(vector_hits, start=1):
                item = _clone_hit(hit)
                item.bm25_score = None
                item.final_score = float(item.vector_score or 0.0)
                item.score = item.final_score
                item.rank = rank
                ranked.append(item)
            filtered = [hit for hit in ranked[:top_k] if hit.final_score >= score_threshold]
            self.last_diagnostics = {
                "fusion_strategy": "vector_only",
                "vector_hit_count": len(vector_hits),
                "bm25_hit_count": 0,
                "merged_hit_count": len(ranked),
                "final_hit_count": len(filtered),
            }
            return filtered
        if mode == "bm25":
            ranked = []
            for rank, hit in enumerate(bm25_hits, start=1):
                item = _clone_hit(hit)
                item.vector_score = None
                item.final_score = _bm25_confidence(float(item.bm25_score or 0.0))
                item.score = item.final_score
                item.rank = rank
                ranked.append(item)
            filtered = [hit for hit in ranked[:top_k] if hit.final_score >= score_threshold]
            self.last_diagnostics = {
                "fusion_strategy": "bm25_only",
                "vector_hit_count": 0,
                "bm25_hit_count": len(bm25_hits),
                "merged_hit_count": len(ranked),
                "final_hit_count": len(filtered),
            }
            return filtered

        if fusion_strategy == "rrf_fusion":
            ranked = self._rrf_fusion(vector_hits, bm25_hits, rrf_k=rrf_k)
        else:
            ranked = self._weighted_score_fusion(vector_hits, bm25_hits, vector_weight=vector_weight)

        filtered = [hit for hit in ranked[:top_k] if hit.final_score >= score_threshold]
        self.last_diagnostics = {
            "fusion_strategy": fusion_strategy,
            "vector_hit_count": len(vector_hits),
            "bm25_hit_count": len(bm25_hits),
            "merged_hit_count": len(ranked),
            "final_hit_count": len(filtered),
        }
        return filtered

    def _weighted_score_fusion(
        self,
        vector_hits: list[RetrievalHit],
        bm25_hits: list[RetrievalHit],
        *,
        vector_weight: float,
    ) -> list[RetrievalHit]:
        by_id: dict[str, RetrievalHit] = {}
        vector_scores = {hit.chunk_id: float(hit.vector_score or 0.0) for hit in vector_hits}
        bm25_scores = {hit.chunk_id: float(hit.bm25_score or 0.0) for hit in bm25_hits}
        for hit in vector_hits + bm25_hits:
            if hit.chunk_id not in by_id:
                by_id[hit.chunk_id] = _clone_hit(hit)
        for chunk_id, hit in by_id.items():
            hit.vector_score = vector_scores.get(chunk_id)
            hit.bm25_score = bm25_scores.get(chunk_id)
            hit.final_score = (
                vector_weight * float(hit.vector_score or 0.0)
                + (1.0 - vector_weight) * _bm25_confidence(float(hit.bm25_score or 0.0))
            )
            hit.score = hit.final_score

        ranked = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
        for rank, hit in enumerate(ranked, start=1):
            hit.rank = rank
        return ranked

    def _rrf_fusion(
        self,
        vector_hits: list[RetrievalHit],
        bm25_hits: list[RetrievalHit],
        *,
        rrf_k: int,
    ) -> list[RetrievalHit]:
        by_id: dict[str, RetrievalHit] = {}
        vector_ranks = {hit.chunk_id: rank for rank, hit in enumerate(vector_hits, start=1)}
        bm25_ranks = {hit.chunk_id: rank for rank, hit in enumerate(bm25_hits, start=1)}
        vector_scores = {hit.chunk_id: float(hit.vector_score or 0.0) for hit in vector_hits}
        bm25_scores = {hit.chunk_id: float(hit.bm25_score or 0.0) for hit in bm25_hits}

        for hit in vector_hits + bm25_hits:
            if hit.chunk_id not in by_id:
                by_id[hit.chunk_id] = _clone_hit(hit)

        for chunk_id, hit in by_id.items():
            raw_score = 0.0
            if chunk_id in vector_ranks:
                raw_score += 1.0 / (rrf_k + vector_ranks[chunk_id])
            if chunk_id in bm25_ranks:
                raw_score += 1.0 / (rrf_k + bm25_ranks[chunk_id])
            hit.vector_score = vector_scores.get(chunk_id)
            hit.bm25_score = bm25_scores.get(chunk_id)
            max_possible = 2.0 / (rrf_k + 1)
            hit.final_score = raw_score / max_possible if max_possible else raw_score
            hit.score = hit.final_score

        ranked = sorted(by_id.values(), key=lambda item: item.final_score, reverse=True)
        for rank, hit in enumerate(ranked, start=1):
            hit.rank = rank
        return ranked
