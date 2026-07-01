from __future__ import annotations

from .config import Settings, settings
from .knowledge_base import KnowledgeBase
from .llm import LLMClient
from .schemas import AskResponse, Citation


class QAService:
    def __init__(self, kb: KnowledgeBase | None = None, app_settings: Settings = settings):
        self.settings = app_settings
        self.kb = kb or KnowledgeBase(app_settings)
        self.llm = LLMClient(app_settings)

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        score_threshold: float | None = None,
        mode: str = "hybrid",
    ) -> AskResponse:
        hits = self.kb.search(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            score_threshold=score_threshold,
            mode=mode,
        )
        if not hits:
            return AskResponse(
                answer="我无法从当前本地资料库中找到足够相关的证据，因此不能可靠回答这个问题。",
                refused=True,
                reason="no_relevant_context",
            )

        answer = self.llm.answer(query, hits)
        citations = [
            Citation(
                source=hit.source,
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                chunk_index=hit.chunk_index,
                start_pos=hit.start_pos,
                end_pos=hit.end_pos,
            )
            for hit in hits
        ]
        return AskResponse(answer=answer, citations=citations, retrieval_hits=hits)

