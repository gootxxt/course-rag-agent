from __future__ import annotations

import hashlib
import re
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def stable_hash(text: str, length: int = 40) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def doc_id_for_path(path: Path) -> str:
    return stable_hash(str(path.resolve()).lower())


def text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return stable_hash(normalized)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    english_words = re.findall(r"[a-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = ["".join(cjk_chars[i : i + 2]) for i in range(len(cjk_chars) - 1)]
    return english_words + cjk_bigrams


def iter_supported_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )
