import unittest

from course_rag_agent.agent_graph import _compress_hits, _next_after_relevance
from course_rag_agent.retriever import HybridRetriever
from course_rag_agent.schemas import RetrievalHit


def _hit(chunk_id: str, *, vector_score=0.0, bm25_score=0.0, text=None):
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id="doc",
        source="source.md",
        text=text or f"text for {chunk_id}",
        start_pos=0,
        end_pos=100,
        chunk_index=int(chunk_id[-1]) if chunk_id[-1].isdigit() else 0,
        vector_score=vector_score,
        bm25_score=bm25_score,
        score=vector_score or bm25_score,
        metadata={"page": 1},
    )


class FakeEmbeddings:
    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def search(self, query_embedding, top_k):
        return [
            _hit("chunk-1", vector_score=0.9),
            _hit("chunk-2", vector_score=0.7),
        ][:top_k]


class FakeBM25Store:
    def search(self, query, top_k):
        return [
            _hit("chunk-2", bm25_score=8.0),
            _hit("chunk-3", bm25_score=6.0),
        ][:top_k]


class RetrievalPhase5Tests(unittest.TestCase):
    def test_weighted_score_fusion_returns_standard_scores(self):
        retriever = HybridRetriever(FakeEmbeddings(), FakeVectorStore(), FakeBM25Store())
        hits = retriever.search(
            "query",
            top_k=3,
            candidate_k=3,
            vector_weight=0.65,
            fusion_strategy="weighted_score_fusion",
            score_threshold=0.0,
        )

        self.assertEqual(len(hits), 3)
        self.assertTrue(all(hit.final_score >= 0 for hit in hits))
        self.assertEqual([hit.rank for hit in hits], [1, 2, 3])
        self.assertEqual(retriever.last_diagnostics["fusion_strategy"], "weighted_score_fusion")

    def test_rrf_fusion_returns_ranked_hits(self):
        retriever = HybridRetriever(FakeEmbeddings(), FakeVectorStore(), FakeBM25Store())
        hits = retriever.search(
            "query",
            top_k=3,
            candidate_k=3,
            fusion_strategy="rrf_fusion",
            score_threshold=0.0,
        )

        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0].chunk_id, "chunk-2")
        self.assertGreater(hits[0].final_score, hits[-1].final_score)
        self.assertEqual(retriever.last_diagnostics["fusion_strategy"], "rrf_fusion")

    def test_compress_keeps_citation_map_and_budget(self):
        hits = [
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc",
                "source": "source.md",
                "text": "A" * 80,
                "final_score": 0.9,
                "start_pos": 0,
                "end_pos": 80,
                "chunk_index": 1,
                "metadata": {},
            },
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc",
                "source": "source.md",
                "text": "duplicate",
                "final_score": 0.8,
                "start_pos": 0,
                "end_pos": 80,
                "chunk_index": 1,
                "metadata": {},
            },
        ]

        context, citations, citation_map, kept_count, _ratio = _compress_hits(hits, char_budget=120)

        self.assertLessEqual(len(context), 120)
        self.assertEqual(kept_count, 1)
        self.assertEqual(len(citations), 1)
        self.assertIn("1", citation_map)
        self.assertIn("[1]", context)

    def test_low_relevance_still_routes_to_final_after_retry_limit(self):
        self.assertEqual(
            _next_after_relevance(
                relevance_passed=False,
                refused=True,
                retry_count=2,
                max_retries=2,
            ),
            "final",
        )


if __name__ == "__main__":
    unittest.main()
