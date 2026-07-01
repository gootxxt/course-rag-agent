from __future__ import annotations

import pickle
from pathlib import Path

from .config import Settings
from .schemas import RetrievalHit, TextChunk
from .utils import tokenize


class BM25Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.index_path = settings.bm25_index_path
        self.chunks: list[TextChunk] = []
        self.model = None
        self.load()

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        existing = {chunk.text_hash for chunk in self.chunks}
        self.chunks.extend(chunk for chunk in chunks if chunk.text_hash not in existing)
        self._rebuild()
        self.save()

    def replace_all(self, chunks: list[TextChunk]) -> None:
        self.chunks = list(chunks)
        self._rebuild()
        self.save()

    def delete_doc(self, doc_id: str) -> int:
        before = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.doc_id != doc_id]
        self._rebuild()
        self.save()
        return before - len(self.chunks)

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        if not self.chunks or self.model is None:
            return []
        scores = self.model.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:top_k]
        hits: list[RetrievalHit] = []
        for index, score in ranked:
            if float(score) <= 0:
                continue
            chunk = self.chunks[index]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    text=chunk.text,
                    start_pos=chunk.start_pos,
                    end_pos=chunk.end_pos,
                    chunk_index=chunk.chunk_index,
                    bm25_score=float(score),
                    score=float(score),
                    metadata=chunk.metadata,
                )
            )
        return hits

    def reset(self) -> None:
        self.chunks = []
        self.model = None
        if self.index_path.exists():
            self.index_path.unlink()

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as f:
            pickle.dump({"chunks": self.chunks}, f)

    def load(self) -> None:
        if not self.index_path.exists():
            return
        with self.index_path.open("rb") as f:
            payload = pickle.load(f)
        self.chunks = payload.get("chunks", [])
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.chunks:
            self.model = None
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("BM25 requires rank-bm25. Run: pip install -r requirements.txt") from exc
        corpus = [tokenize(chunk.text) for chunk in self.chunks]
        self.model = BM25Okapi(corpus)

