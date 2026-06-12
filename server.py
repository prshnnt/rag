"""FastAPI server for RAG.

Endpoints:
    GET  /        -> HTML frontend (upload + query UI)
    POST /ingest  -> multipart file upload, split, upsert into VectorStore
    POST /query   -> JSON {query, k} -> list of hits
    GET  /stats   -> document count in the collection
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from docLoader.loader import PageFormat, load_pages
from ingestion.pipeline import ingest
from vectordatabase.vectorstore import VectorStore


# ---------- paths ----------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


# ---------- app + singletons ----------

app = FastAPI(title="RAG Server", version="0.1.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# One VectorStore for the app lifetime. Swap embedding_fn here for prod.
store = VectorStore(persist_directory="./rag", collection_name="documents")

# Default chunker. Tune as needed.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# File types we know how to read.
PDF_TYPES = {"application/pdf", "application/x-pdf"}
TEXT_EXTS = {".txt", ".md", ".markdown"}


# ---------- helpers ----------

def _save_upload(upload: UploadFile) -> Path:
    """Persist the uploaded file under uploads/ and return the path."""
    if not upload.filename:
        raise HTTPException(status_code=400, detail="uploaded file has no filename")
    safe_name = Path(upload.filename).name  # strip any path components
    dest = UPLOAD_DIR / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _load_documents(path: Path) -> List[Document]:
    """Load a file into a list of LangChain Documents."""
    suffix = path.suffix.lower()
    if suffix == ".pdf" or (upload_mime := None):
        # Use existing loader. range_=None -> all pages.
        docs = load_pages(str(path), range_=None, fmt=PageFormat.DOCUMENT)
        return [d for d in docs if isinstance(d, Document)]
    if suffix in TEXT_EXTS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": path.name, "page": 1})]
    raise HTTPException(
        status_code=400,
        detail=f"unsupported file type: {suffix} (use .pdf, .txt, .md)",
    )


def _chunk_documents(docs: List[Document], source_tag: str) -> List[Document]:
    """Split documents and tag each chunk with its source."""
    chunks: List[Document] = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d.page_content)):
            meta = dict(d.metadata or {})
            meta.setdefault("source", source_tag)
            meta["chunk"] = i
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


# ---------- routes ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)) -> JSONResponse:
    path = _save_upload(file)
    try:
        docs = _load_documents(path)
        if not docs:
            raise HTTPException(status_code=400, detail="no extractable text in file")
        chunks = _chunk_documents(docs, source_tag=path.name)
        written = ingest(chunks, store, source_tag=path.name)
    finally:
        # keep file on disk for debugging; comment out the next line to delete
        # path.unlink(missing_ok=True)
        pass

    return JSONResponse(
        {
            "filename": path.name,
            "pages_or_docs": len(docs),
            "chunks": len(chunks),
            "stored": written,
            "collection_size": store.count(),
        }
    )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=50)


class QueryHitOut(BaseModel):
    id: str
    score: Optional[float] = None
    source: Optional[str] = None
    page: Optional[int] = None
    content: str


@app.post("/query", response_model=List[QueryHitOut])
async def query(req: QueryRequest) -> List[QueryHitOut]:
    hits = store.query(req.query, k=req.k)
    out: List[QueryHitOut] = []
    for h in hits:
        meta = h.document.metadata or {}
        out.append(
            QueryHitOut(
                id=h.id,
                score=h.score,
                source=str(meta.get("source", "")) or None,
                page=int(meta["page"]) if str(meta.get("page", "")).isdigit() else None,
                content=h.document.page_content,
            )
        )
    return out


@app.get("/stats")
async def stats() -> dict:
    return {"count": store.count(), "collection": store.collection_name}
