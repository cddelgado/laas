from __future__ import annotations

import json
import math
import mimetypes
import re
import sqlite3
import struct
import threading
import time
import uuid
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .embedding import EmbeddingManager, estimate_tokens
from .settings import Settings


class LocalFileNotFoundError(KeyError):
    pass


class VectorStoreNotFoundError(KeyError):
    pass


class VectorStoreFileNotFoundError(KeyError):
    pass


def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class LocalStorage:
    def __init__(self, settings: Settings, embedding_manager: EmbeddingManager) -> None:
        self.settings = settings
        self.embedding_manager = embedding_manager
        self._lock = threading.RLock()
        self._initialized = False

    @property
    def root(self) -> Path:
        return self.settings.file_storage_dir

    @property
    def files_dir(self) -> Path:
        return self.root / "files"

    @property
    def db_path(self) -> Path:
        return self.settings.file_storage_db_path

    def status(self) -> dict[str, Any]:
        self._ensure_initialized()
        return {
            "object": "local.file_storage",
            "root": str(self.root),
            "database": str(self.db_path),
            "files_dir": str(self.files_dir),
        }

    def create_file(
        self,
        *,
        filename: str,
        content: bytes,
        purpose: str,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        if not content:
            raise ValueError("uploaded file is empty")
        safe_name = Path(filename or "upload.bin").name or "upload.bin"
        file_id = _new_id("file")
        created_at = _now()
        storage_name = f"{file_id}{Path(safe_name).suffix}"
        with self._lock:
            self._ensure_initialized()
            path = self.files_dir / storage_name
            path.write_bytes(content)
            resolved_mime = mime_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
            with self._connect() as con:
                con.execute(
                    """
                    insert into files(id, purpose, filename, bytes, mime_type, path, created_at, status)
                    values(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (file_id, purpose, safe_name, len(content), resolved_mime, str(path), created_at, "processed"),
                )
            return self.get_file(file_id)

    def list_files(self, *, purpose: str | None = None) -> list[dict[str, Any]]:
        self._ensure_initialized()
        with self._connect() as con:
            if purpose:
                rows = con.execute(
                    "select * from files where purpose = ? order by created_at desc, id desc",
                    (purpose,),
                ).fetchall()
            else:
                rows = con.execute("select * from files order by created_at desc, id desc").fetchall()
        return [_file_object(row) for row in rows]

    def get_file(self, file_id: str) -> dict[str, Any]:
        row = self._file_row(file_id)
        return _file_object(row)

    def file_path(self, file_id: str) -> Path:
        row = self._file_row(file_id)
        path = Path(row["path"])
        if not path.exists() or not path.is_file():
            raise LocalFileNotFoundError(file_id)
        return path

    def delete_file(self, file_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._file_row(file_id)
            path = Path(row["path"])
            with self._connect() as con:
                con.execute("delete from vector_chunks where file_id = ?", (file_id,))
                con.execute("delete from vector_store_files where file_id = ?", (file_id,))
                con.execute("delete from files where id = ?", (file_id,))
            path.unlink(missing_ok=True)
        return {"id": file_id, "object": "file.deleted", "deleted": True}

    def create_vector_store(
        self,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        store_id = _new_id("vs")
        created_at = _now()
        with self._lock:
            self._ensure_initialized()
            with self._connect() as con:
                con.execute(
                    "insert into vector_stores(id, name, created_at, metadata_json) values(?, ?, ?, ?)",
                    (store_id, name, created_at, json.dumps(metadata or {}, sort_keys=True)),
                )
            return self.get_vector_store(store_id)

    def list_vector_stores(self) -> list[dict[str, Any]]:
        self._ensure_initialized()
        with self._connect() as con:
            rows = con.execute("select * from vector_stores order by created_at desc, id desc").fetchall()
        return [self._vector_store_object(row) for row in rows]

    def get_vector_store(self, vector_store_id: str) -> dict[str, Any]:
        row = self._vector_store_row(vector_store_id)
        return self._vector_store_object(row)

    def delete_vector_store(self, vector_store_id: str) -> dict[str, Any]:
        with self._lock:
            self._vector_store_row(vector_store_id)
            with self._connect() as con:
                con.execute("delete from vector_chunks where vector_store_id = ?", (vector_store_id,))
                con.execute("delete from vector_store_files where vector_store_id = ?", (vector_store_id,))
                con.execute("delete from vector_stores where id = ?", (vector_store_id,))
        return {"id": vector_store_id, "object": "vector_store.deleted", "deleted": True}

    def attach_file(self, *, vector_store_id: str, file_id: str) -> dict[str, Any]:
        self._prepare_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)
        self._index_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)
        return self.get_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)

    def attach_file_async(self, *, vector_store_id: str, file_id: str) -> dict[str, Any]:
        self._prepare_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)
        thread = threading.Thread(
            target=self._index_vector_store_file,
            kwargs={"vector_store_id": vector_store_id, "file_id": file_id, "raise_errors": False},
            daemon=True,
        )
        thread.start()
        return self.get_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)

    def vector_store_indexing_status(self, vector_store_id: str) -> dict[str, Any]:
        store = self.get_vector_store(vector_store_id)
        files = self.list_vector_store_files(vector_store_id)
        return {
            "object": "local.vector_store.indexing_status",
            "vector_store_id": vector_store_id,
            "status": store["status"],
            "file_counts": store["file_counts"],
            "files": files,
        }

    def _prepare_vector_store_file(self, *, vector_store_id: str, file_id: str) -> None:
        with self._lock:
            self._vector_store_row(vector_store_id)
            self._file_row(file_id)
            created_at = _now()
            with self._connect() as con:
                con.execute(
                    """
                    insert into vector_store_files(vector_store_id, file_id, status, created_at, last_error)
                    values(?, ?, ?, ?, null)
                    on conflict(vector_store_id, file_id) do update set status = excluded.status, last_error = null
                    """,
                    (vector_store_id, file_id, "in_progress", created_at),
                )

    def _index_vector_store_file(self, *, vector_store_id: str, file_id: str, raise_errors: bool = True) -> None:
        with self._lock:
            try:
                self._index_file(vector_store_id=vector_store_id, file_id=file_id)
            except Exception as exc:
                with self._connect() as con:
                    con.execute(
                        """
                        update vector_store_files set status = ?, last_error = ?
                        where vector_store_id = ? and file_id = ?
                        """,
                        ("failed", str(exc), vector_store_id, file_id),
                    )
                if raise_errors:
                    raise
                return
            with self._connect() as con:
                con.execute(
                    """
                    update vector_store_files set status = ?, last_error = null
                    where vector_store_id = ? and file_id = ?
                    """,
                    ("completed", vector_store_id, file_id),
                )

    def list_vector_store_files(self, vector_store_id: str) -> list[dict[str, Any]]:
        self._vector_store_row(vector_store_id)
        with self._connect() as con:
            rows = con.execute(
                """
                select * from vector_store_files
                where vector_store_id = ?
                order by created_at desc, file_id desc
                """,
                (vector_store_id,),
            ).fetchall()
        return [_vector_store_file_object(row) for row in rows]

    def get_vector_store_file(self, *, vector_store_id: str, file_id: str) -> dict[str, Any]:
        row = self._vector_store_file_row(vector_store_id=vector_store_id, file_id=file_id)
        return _vector_store_file_object(row)

    def delete_vector_store_file(self, *, vector_store_id: str, file_id: str) -> dict[str, Any]:
        with self._lock:
            self._vector_store_file_row(vector_store_id=vector_store_id, file_id=file_id)
            with self._connect() as con:
                con.execute(
                    "delete from vector_chunks where vector_store_id = ? and file_id = ?",
                    (vector_store_id, file_id),
                )
                con.execute(
                    "delete from vector_store_files where vector_store_id = ? and file_id = ?",
                    (vector_store_id, file_id),
                )
        return {
            "id": file_id,
            "object": "vector_store.file.deleted",
            "deleted": True,
            "vector_store_id": vector_store_id,
        }

    def search_vector_store(self, *, vector_store_id: str, query: str, limit: int = 8) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        self._vector_store_row(vector_store_id)
        query_vector = self.embedding_manager.embed([query], dimensions=self.settings.embedding_dimensions)[0]
        with self._connect() as con:
            rows = con.execute(
                """
                select vc.*, f.filename
                from vector_chunks vc
                join files f on f.id = vc.file_id
                where vc.vector_store_id = ?
                """,
                (vector_store_id,),
            ).fetchall()
        scored = []
        for row in rows:
            vector = _blob_to_vector(row["embedding"])
            scored.append((_cosine(query_vector, vector), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        data = []
        for score, row in scored[: max(1, limit)]:
            data.append(
                {
                    "object": "vector_store.search_result",
                    "id": row["id"],
                    "score": score,
                    "text": row["text"],
                    "file_id": row["file_id"],
                    "filename": row["filename"],
                    "chunk_index": row["chunk_index"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
            )
        return {"object": "list", "data": data, "has_more": False}

    def _index_file(self, *, vector_store_id: str, file_id: str) -> None:
        row = self._file_row(file_id)
        text = _extract_text(Path(row["path"]))
        chunks = _chunk_text(
            text,
            chunk_tokens=self.settings.vector_store_chunk_tokens,
            overlap_tokens=self.settings.vector_store_chunk_overlap_tokens,
        )
        if not chunks:
            raise ValueError("file did not contain extractable text")
        vectors = self.embedding_manager.embed(chunks, dimensions=self.settings.embedding_dimensions)
        with self._connect() as con:
            con.execute(
                "delete from vector_chunks where vector_store_id = ? and file_id = ?",
                (vector_store_id, file_id),
            )
            con.executemany(
                """
                insert into vector_chunks(
                    id, vector_store_id, file_id, chunk_index, text, token_count,
                    embedding_model, embedding, metadata_json
                )
                values(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        _new_id("chunk"),
                        vector_store_id,
                        file_id,
                        index,
                        chunk,
                        estimate_tokens(chunk),
                        self.settings.embedding_model_id,
                        _vector_to_blob(vector),
                        json.dumps({"filename": row["filename"]}, sort_keys=True),
                    )
                    for index, (chunk, vector) in enumerate(zip(chunks, vectors))
                ],
            )

    def _file_row(self, file_id: str) -> sqlite3.Row:
        self._ensure_initialized()
        with self._connect() as con:
            row = con.execute("select * from files where id = ?", (file_id,)).fetchone()
        if row is None:
            raise LocalFileNotFoundError(file_id)
        return row

    def _vector_store_row(self, vector_store_id: str) -> sqlite3.Row:
        self._ensure_initialized()
        with self._connect() as con:
            row = con.execute("select * from vector_stores where id = ?", (vector_store_id,)).fetchone()
        if row is None:
            raise VectorStoreNotFoundError(vector_store_id)
        return row

    def _vector_store_file_row(self, *, vector_store_id: str, file_id: str) -> sqlite3.Row:
        self._ensure_initialized()
        with self._connect() as con:
            row = con.execute(
                "select * from vector_store_files where vector_store_id = ? and file_id = ?",
                (vector_store_id, file_id),
            ).fetchone()
        if row is None:
            raise VectorStoreFileNotFoundError(file_id)
        return row

    def _vector_store_object(self, row: sqlite3.Row) -> dict[str, Any]:
        with self._connect() as con:
            counts = con.execute(
                """
                select
                    count(*) as total,
                    sum(case when status = 'completed' then 1 else 0 end) as completed,
                    sum(case when status = 'failed' then 1 else 0 end) as failed,
                    sum(case when status = 'in_progress' then 1 else 0 end) as in_progress
                from vector_store_files
                where vector_store_id = ?
                """,
                (row["id"],),
            ).fetchone()
        return {
            "id": row["id"],
            "object": "vector_store",
            "created_at": row["created_at"],
            "name": row["name"],
            "usage_bytes": 0,
            "file_counts": {
                "in_progress": int(counts["in_progress"] or 0),
                "completed": int(counts["completed"] or 0),
                "failed": int(counts["failed"] or 0),
                "cancelled": 0,
                "total": int(counts["total"] or 0),
            },
            "status": "completed" if not counts["in_progress"] else "in_progress",
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            self.files_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as con:
                con.executescript(
                    """
                    create table if not exists files(
                        id text primary key,
                        purpose text not null,
                        filename text not null,
                        bytes integer not null,
                        mime_type text not null,
                        path text not null,
                        created_at integer not null,
                        status text not null
                    );
                    create table if not exists vector_stores(
                        id text primary key,
                        name text,
                        created_at integer not null,
                        metadata_json text not null default '{}'
                    );
                    create table if not exists vector_store_files(
                        vector_store_id text not null,
                        file_id text not null,
                        status text not null,
                        created_at integer not null,
                        last_error text,
                        primary key(vector_store_id, file_id)
                    );
                    create table if not exists vector_chunks(
                        id text primary key,
                        vector_store_id text not null,
                        file_id text not null,
                        chunk_index integer not null,
                        text text not null,
                        token_count integer not null,
                        embedding_model text not null,
                        embedding blob not null,
                        metadata_json text not null default '{}'
                    );
                    create index if not exists idx_vector_chunks_store on vector_chunks(vector_store_id);
                    create index if not exists idx_vector_chunks_file on vector_chunks(file_id);
                    """
                )
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("pragma foreign_keys = on")
        return con


def _file_object(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "object": "file",
        "bytes": row["bytes"],
        "created_at": row["created_at"],
        "filename": row["filename"],
        "purpose": row["purpose"],
        "status": row["status"],
    }


def _vector_store_file_object(row: sqlite3.Row) -> dict[str, Any]:
    payload = {
        "id": row["file_id"],
        "object": "vector_store.file",
        "created_at": row["created_at"],
        "vector_store_id": row["vector_store_id"],
        "status": row["status"],
    }
    if row["last_error"]:
        payload["last_error"] = row["last_error"]
    return payload


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return _extract_html_text(path)
    if suffix in {".md", ".markdown"}:
        return _extract_markdown_text(path)
    if suffix == ".docx":
        return _extract_docx_text(path)
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    content = path.read_bytes()
    if not content:
        return ""
    return content.decode("utf-8", errors="ignore")


def _extract_markdown_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.MULTILINE)
    return text


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _extract_html_text(path: Path) -> str:
    parser = _TextHTMLParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return " ".join(parser.parts)


def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception:
        return path.read_bytes().decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", xml)
    return re.sub(r"\s+", " ", text)


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader
        except Exception:
            return path.read_bytes().decode("latin-1", errors="ignore")

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _chunk_text(text: str, *, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunk_size = max(1, chunk_tokens)
    overlap = min(max(0, overlap_tokens), chunk_size - 1)
    chunks = []
    index = 0
    while index < len(words):
        chunk = " ".join(words[index : index + chunk_size])
        chunks.append(chunk)
        if index + chunk_size >= len(words):
            break
        index += chunk_size - overlap
    return chunks


def _vector_to_blob(vector: list[float]) -> bytes:
    return b"".join(struct.pack("<f", float(value)) for value in vector)


def _blob_to_vector(blob: bytes) -> list[float]:
    return [value[0] for value in struct.iter_unpack("<f", blob)]


def _cosine(left: list[float], right: list[float]) -> float:
    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    dot = sum(left[index] * right[index] for index in range(length))
    left_norm = math.sqrt(sum(value * value for value in left[:length])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:length])) or 1.0
    return dot / (left_norm * right_norm)
