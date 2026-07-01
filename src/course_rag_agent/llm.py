from __future__ import annotations

from .config import Settings
from .schemas import RetrievalHit


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def answer(self, query: str, hits: list[RetrievalHit]) -> str:
        if not self.settings.openai_api_key:
            return self._fallback_answer(query, hits)

        try:
            from openai import OpenAI
        except ImportError:
            return self._fallback_answer(query, hits)

        client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )
        context = self._format_context(hits)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是课程资料问答助手。只能根据给定上下文回答。"
                    "如果上下文没有证据，必须说无法从资料中确认。"
                    "回答末尾必须列出引用，格式为 [source#chunk_index:start-end]。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{query}\n\n上下文：\n{context}",
            },
        ]
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            temperature=self.settings.llm_temperature,
        )
        return response.choices[0].message.content or ""

    def _fallback_answer(self, query: str, hits: list[RetrievalHit]) -> str:
        lines = [
            "未配置 OPENAI_API_KEY，以下为基于检索片段的证据摘要：",
            f"问题：{query}",
            "",
        ]
        for i, hit in enumerate(hits, start=1):
            snippet = hit.text.replace("\n", " ").strip()
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            lines.append(
                f"{i}. {snippet} "
                f"[{hit.source}#{hit.chunk_index}:{hit.start_pos}-{hit.end_pos}]"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_context(hits: list[RetrievalHit]) -> str:
        blocks = []
        for hit in hits:
            blocks.append(
                f"[{hit.source}#{hit.chunk_index}:{hit.start_pos}-{hit.end_pos}]\n{hit.text}"
            )
        return "\n\n".join(blocks)

