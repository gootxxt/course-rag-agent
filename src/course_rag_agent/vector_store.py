from __future__ import annotations

from collections import Counter

from .config import Settings
from .schemas import RetrievalHit, TextChunk


class ChromaVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.ensure_dirs()
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Chroma requires chromadb. Run: pip install -r requirements.txt") from exc
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(name=settings.collection_name)

    def reload_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(name=self.settings.collection_name)

    def add_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> tuple[int, int]:
        new_chunks: list[TextChunk] = []
        new_embeddings: list[list[float]] = []
        skipped = 0
        for chunk, embedding in zip(chunks, embeddings):
            if self.has_text_hash(chunk.text_hash):
                skipped += 1
                continue
            new_chunks.append(chunk)
            new_embeddings.append(embedding)

        if not new_chunks:
            return 0, skipped

        self.collection.add(
            ids=[chunk.chunk_id for chunk in new_chunks],
            documents=[chunk.text for chunk in new_chunks],
            embeddings=new_embeddings,
            metadatas=[self._metadata(chunk) for chunk in new_chunks],
        )
        return len(new_chunks), skipped

    def has_text_hash(self, digest: str) -> bool:
        try:
            result = self.collection.get(where={"text_hash": digest}, limit=1)
            return bool(result.get("ids"))
        except Exception:
            return False

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalHit]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievalHit] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            score = 1.0 / (1.0 + float(distance))
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    doc_id=str(meta["doc_id"]),
                    source=str(meta["source"]),
                    text=text,
                    start_pos=int(meta["start_pos"]),
                    end_pos=int(meta["end_pos"]),
                    chunk_index=int(meta["chunk_index"]),
                    vector_score=score,
                    score=score,
                    metadata=dict(meta),
                )
            )
        return hits

    def all_chunks(self) -> list[TextChunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        chunks: list[TextChunk] = []
        for chunk_id, text, meta in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", [])):
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=str(meta["doc_id"]),
                    source=str(meta["source"]),
                    text=text,
                    text_hash=str(meta["text_hash"]),
                    chunk_index=int(meta["chunk_index"]),
                    start_pos=int(meta["start_pos"]),
                    end_pos=int(meta["end_pos"]),
                    metadata=dict(meta),
                )
            )
        return chunks

    def delete_doc(self, doc_id: str) -> int:
        before = self.count()
        self.collection.delete(where={"doc_id": doc_id})
        return before - self.count()

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.settings.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.settings.collection_name)

    def count(self) -> int:
        return int(self.collection.count())

    def stats(self) -> dict[str, int]:
        result = self.collection.get(include=["metadatas"])
        doc_ids = [meta.get("doc_id") for meta in result.get("metadatas", [])]
        source_count = Counter(meta.get("source") for meta in result.get("metadatas", []))
        return {
            "documents": len(set(doc_ids)),
            "chunks": len(doc_ids),
            "sources": len(source_count),
        }

    @staticmethod
    def _metadata(chunk: TextChunk) -> dict[str, str | int | float | bool]:
        metadata = {
            "doc_id": chunk.doc_id,
            "source": chunk.source,
            "text_hash": chunk.text_hash,
            "chunk_index": chunk.chunk_index,
            "start_pos": chunk.start_pos,
            "end_pos": chunk.end_pos,
        }
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        return metadata
