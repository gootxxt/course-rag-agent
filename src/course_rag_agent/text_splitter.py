from __future__ import annotations

import re

from .schemas import LoadedDocument, TextChunk
from .utils import stable_hash, text_hash


def _clean_piece(piece: str) -> str:
    return re.sub(r"\s+", " ", piece).strip()


def _recursive_ranges(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    text_len = len(text)
    separators = ["\n\n", "\n", "。", ".", "；", ";", "，", ",", " "]

    while start < text_len:
        hard_end = min(start + chunk_size, text_len)
        end = hard_end
        window = text[start:hard_end]
        if hard_end < text_len:
            best = -1
            for sep in separators:
                pos = window.rfind(sep)
                if pos > best and pos >= max(80, chunk_size // 3):
                    best = pos + len(sep)
            if best > 0:
                end = start + best

        if end <= start:
            end = hard_end
        ranges.append((start, end))
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)
    return ranges


def _semantic_ranges(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    pattern = re.compile(r"(.+?(?:\n\s*\n|[。！？.!?]\s+|$))", re.S)
    pieces = [(m.start(), m.end()) for m in pattern.finditer(text) if _clean_piece(m.group(0))]
    if not pieces:
        return _recursive_ranges(text, chunk_size, overlap)

    ranges: list[tuple[int, int]] = []
    current_start = pieces[0][0]
    current_end = pieces[0][1]
    for piece_start, piece_end in pieces[1:]:
        if piece_end - current_start <= chunk_size:
            current_end = piece_end
        else:
            ranges.append((current_start, current_end))
            current_start = max(piece_start - overlap, 0) if overlap else piece_start
            current_end = piece_end
    ranges.append((current_start, current_end))
    return ranges


def split_document(
    document: LoadedDocument,
    strategy: str = "recursive",
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = document.text
    ranges = (
        _semantic_ranges(text, chunk_size, overlap)
        if strategy == "semantic"
        else _recursive_ranges(text, chunk_size, overlap)
    )

    chunks: list[TextChunk] = []
    for index, (start, end) in enumerate(ranges):
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        digest = text_hash(chunk_text)
        chunk_id = f"{document.doc_id}:{index}:{stable_hash(digest, 8)}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                source=document.source,
                text=chunk_text,
                text_hash=digest,
                chunk_index=index,
                start_pos=start,
                end_pos=end,
                metadata={
                    **document.metadata,
                    "chunk_strategy": strategy,
                },
            )
        )
    return chunks

