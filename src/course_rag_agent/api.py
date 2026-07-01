from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .agent_graph import run_agent
from .config import settings
from .knowledge_base import KnowledgeBase
from .qa import QAService

app = FastAPI(title="Course RAG Baseline", version="0.1.0")
kb = KnowledgeBase(settings)
qa_service = QAService(kb, settings)


class ImportDirRequest(BaseModel):
    path: str
    strategy: str | None = None
    chunk_size: int | None = None
    overlap: int | None = None


class AskRequest(BaseModel):
    query: str
    top_k: int | None = None
    candidate_k: int | None = None
    score_threshold: float | None = None
    mode: str = "hybrid"


class AgentRequest(BaseModel):
    query: str


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix
    if suffix.lower() not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only txt, md and pdf are supported")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = kb.add_file(tmp_path)
    return result.model_dump()


@app.post("/documents/import-dir")
def import_dir(request: ImportDirRequest):
    result = kb.add_dir(
        request.path,
        strategy=request.strategy,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )
    return result.model_dump()


@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: str):
    return kb.delete_doc(doc_id)


@app.get("/stats")
def stats():
    return kb.stats()


@app.post("/ask")
def ask(request: AskRequest):
    response = qa_service.ask(
        request.query,
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        score_threshold=request.score_threshold,
        mode=request.mode,
    )
    return response.model_dump()


@app.post("/agent")
def agent(request: AgentRequest):
    return run_agent(request.query, kb, settings)
