import unittest

from course_rag_agent.agent_graph import _next_after_self_check, _rule_self_check


def _hit():
    return {
        "source": "data/sample_docs/rag_basics.md",
        "chunk_id": "doc-1:0",
        "text": "RAG uses retrieved evidence before generation.",
    }


def _citation():
    return {
        "source": "data/sample_docs/rag_basics.md",
        "chunk_id": "doc-1:0",
        "chunk_index": 0,
        "start_pos": 0,
        "end_pos": 64,
    }


class SelfCheckTests(unittest.TestCase):
    def test_self_check_passes_with_valid_citation(self):
        result = _rule_self_check(
            answer="RAG answers should be grounded in retrieved evidence [1].",
            refused=False,
            citations=[_citation()],
            retrieval_hits=[_hit()],
            top_score=0.7,
            threshold=0.25,
        )

        self.assertIs(result["passed"], True)
        self.assertEqual(result["citation_coverage"], 1.0)
        self.assertEqual(result["groundedness_score"], 1.0)

    def test_self_check_fails_without_inline_citation(self):
        result = _rule_self_check(
            answer="RAG answers should be grounded in retrieved evidence.",
            refused=False,
            citations=[_citation()],
            retrieval_hits=[_hit()],
            top_score=0.7,
            threshold=0.25,
        )

        self.assertIs(result["passed"], False)
        self.assertIn("answer_missing_inline_citation", result["unsupported_claims"])

    def test_self_check_fails_with_unknown_chunk_id(self):
        citation = _citation()
        citation["chunk_id"] = "missing"

        result = _rule_self_check(
            answer="RAG answers should be grounded in retrieved evidence [1].",
            refused=False,
            citations=[citation],
            retrieval_hits=[_hit()],
            top_score=0.7,
            threshold=0.25,
        )

        self.assertIs(result["passed"], False)
        self.assertIn("citation_not_from_retrieval_hits", result["unsupported_claims"])

    def test_self_check_routes_to_final_after_generation_retry_limit(self):
        self.assertEqual(
            _next_after_self_check(
                self_check_passed=False,
                refused=False,
                generation_retry_count=1,
                max_generation_retries=1,
            ),
            "final",
        )

    def test_self_check_routes_to_generate_before_generation_retry_limit(self):
        self.assertEqual(
            _next_after_self_check(
                self_check_passed=False,
                refused=False,
                generation_retry_count=0,
                max_generation_retries=1,
            ),
            "generate",
        )


if __name__ == "__main__":
    unittest.main()
