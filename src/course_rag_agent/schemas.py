from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    doc_id: str
    source: str
    file_type: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    text: str
    text_hash: str
    chunk_index: int
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddResult(BaseModel):
    files_seen: int = 0
    files_loaded: int = 0
    chunks_created: int = 0
    chunks_added: int = 0
    chunks_skipped: int = 0
    errors: list[str] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    text: str
    start_pos: int
    end_pos: int
    chunk_index: int
    vector_score: float = 0.0
    bm25_score: float = 0.0
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source: str
    doc_id: str
    chunk_id: str
    chunk_index: int
    start_pos: int
    end_pos: int


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    refused: bool = False
    reason: str | None = None

