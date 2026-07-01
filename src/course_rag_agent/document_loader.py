from __future__ import annotations

from pathlib import Path

from .schemas import LoadedDocument
from .utils import doc_id_for_path


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading PDF requires pypdf. Run: pip install -r requirements.txt") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n[Page {i + 1}]\n{text}")
    return "\n".join(pages)


def load_document(path: str | Path) -> LoadedDocument:
    file_path = Path(path).resolve()
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = _read_text_file(file_path)
    elif suffix == ".pdf":
        text = _read_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    return LoadedDocument(
        doc_id=doc_id_for_path(file_path),
        source=str(file_path),
        file_type=suffix.lstrip("."),
        text=text,
        metadata={
            "filename": file_path.name,
            "suffix": suffix,
        },
    )

