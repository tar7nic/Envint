"""Clause-level chunker: one chunk per numbered clause (e.g. '4.2.1 ...')."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.config import get_settings
from app.schemas import ChunkPayload

CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
TITLE_RE = re.compile(r"^#\s+([A-Z]+-\d+)\s*[:\-]?\s*(.*)$")


class MalformedDocumentError(ValueError):
    """Raised when a document has no valid ID header or no clauses."""


def normalize(text: str) -> str:
    return " ".join(text.split())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def chunk_text(raw: str, source_file: str = "inline") -> list[ChunkPayload]:
    if not raw or not raw.strip():
        raise MalformedDocumentError("empty document")
    s = get_settings()
    doc_id, section, version = None, None, "1"
    chunks: list[ChunkPayload] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if m := TITLE_RE.match(line):
            doc_id = m.group(1)
        elif line.lower().startswith("version:"):
            version = line.split(":", 1)[1].strip() or "1"
        elif line.startswith("## "):
            section = line[3:].strip()
        elif doc_id and (m := CLAUSE_RE.match(line)):
            clause_id, text = m.group(1), m.group(2)
            chunks.append(ChunkPayload(
                chunk_id=f"{doc_id}#{clause_id}", doc_id=doc_id, clause_id=clause_id,
                section=section, text=text, source_file=source_file, doc_version=version,
                content_hash=content_hash(text),
                embedding_model=s.embedding_model, embedding_version=s.embedding_version,
            ))

    if doc_id is None:
        raise MalformedDocumentError("missing '# DOC-ID' header")
    if not chunks:
        raise MalformedDocumentError(f"{doc_id}: no numbered clauses found")
    return chunks


def chunk_file(path: Path) -> list[ChunkPayload]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise MalformedDocumentError(f"{path}: not valid UTF-8") from e
    return chunk_text(raw, source_file=Path(path).name)


def load_corpus(corpus_dir: Path) -> tuple[list[ChunkPayload], list[tuple[str, str]]]:
    """Chunk every .md file. A malformed file is reported, never fatal to the others."""
    chunks, failed = [], []
    for f in sorted(Path(corpus_dir).glob("*.md")):
        try:
            chunks += chunk_file(f)
        except MalformedDocumentError as e:
            failed.append((f.name, str(e)))
    return chunks, failed