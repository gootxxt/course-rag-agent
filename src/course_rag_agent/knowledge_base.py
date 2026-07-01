from __future__ import annotations

from pathlib import Path

from .bm25_store import BM25Store
from .config import Settings, settings
from .document_loader import load_document
from .embeddings import EmbeddingModel
from .retriever import HybridRetriever
from .schemas import AddResult, RetrievalHit
from .text_splitter import split_document
from .utils import doc_id_for_path, iter_supported_files
from .vector_store import ChromaVectorStore


class KnowledgeBase:
    def __init__(self, app_settings: Settings = settings):
        self.settings = app_settings
        self.settings.ensure_dirs()
        self.embeddings = EmbeddingModel(app_settings)
        self.vector_store = ChromaVectorStore(app_settings)
        self.bm25_store = BM25Store(app_settings)
        self.retriever = HybridRetriever(self.embeddings, self.vector_store, self.bm25_store)
        self._sync_bm25_from_chroma()

    def add_file(
        self,
        path: str | Path,
        strategy: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> AddResult:
        result = AddResult(files_seen=1)
        try:
            document = load_document(path)
            chunks = split_document(
                document,
                strategy=strategy or self.settings.chunk_strategy,
                chunk_size=chunk_size or self.settings.chunk_size,
                overlap=self.settings.chunk_overlap if overlap is None else overlap,
            )
            embeddings = self.embeddings.encode([chunk.text for chunk in chunks])
            added, skipped = self.vector_store.add_chunks(chunks, embeddings)
            self._sync_bm25_from_chroma()
            result.files_loaded = 1
            result.chunks_created = len(chunks)
            result.chunks_added = added
            result.chunks_skipped = skipped
        except Exception as exc:
            result.errors.append(f"{path}: {exc}")
        return result

    def add_dir(
        self,
        path: str | Path,
        strategy: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> AddResult:
        files = iter_supported_files(Path(path))
        total = AddResult(files_seen=len(files))
        for file_path in files:
            partial = self.add_file(file_path, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
            total.files_loaded += partial.files_loaded
            total.chunks_created += partial.chunks_created
            total.chunks_added += partial.chunks_added
            total.chunks_skipped += partial.chunks_skipped
            total.errors.extend(partial.errors)
        return total

    def update_file(self, path: str | Path, **kwargs) -> AddResult:
        doc_id = doc_id_for_path(Path(path))
        self.delete_doc(doc_id)
        return self.add_file(path, **kwargs)

    def delete_doc(self, doc_id: str) -> dict[str, int]:
        vector_deleted = self.vector_store.delete_doc(doc_id)
        bm25_deleted = self.bm25_store.delete_doc(doc_id)
        self._sync_bm25_from_chroma()
        return {"vector_chunks_deleted": vector_deleted, "bm25_chunks_deleted": bm25_deleted}

    def reset(self) -> None:
        self.vector_store.reset()
        self.bm25_store.reset()

    def refresh_indexes(self) -> None:
        self.vector_store.reload_collection()
        self._sync_bm25_from_chroma()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        score_threshold: float | None = None,
        mode: str = "hybrid",
    ) -> list[RetrievalHit]:
        self.refresh_indexes()
        return self.retriever.search(
            query=query,
            top_k=top_k or self.settings.top_k,
            candidate_k=candidate_k or self.settings.candidate_k,
            vector_weight=self.settings.vector_weight,
            score_threshold=self.settings.score_threshold if score_threshold is None else score_threshold,
            mode=mode,
        )

    def stats(self) -> dict[str, int | str | bool]:
        self.refresh_indexes()
        vector_stats = self.vector_store.stats()
        return {
            **vector_stats,
            "bm25_chunks": len(self.bm25_store.chunks),
            "collection": self.settings.collection_name,
            "embedding_model": self.settings.embedding_model,
            "embedding_fallback": self.embeddings.using_fallback,
        }

    def _sync_bm25_from_chroma(self) -> None:
        self.bm25_store.replace_all(self.vector_store.all_chunks())
