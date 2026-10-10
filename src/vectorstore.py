"""
Knowledge-base "training" (document ingestion + FAISS index) and semantic search.

How training works
------------------
A RAG chatbot is not trained like a neural network: "training" means reading your
documents, splitting them into chunks, turning every chunk into an embedding vector and
storing those vectors in a FAISS index. Queries are then answered by looking up the most
similar chunks instead of re-reading the documents - which is what makes answers fast.

* train() is INCREMENTAL: a manifest (manifest.json) remembers every trained file and its
  SHA-256 hash. Only new/changed files are embedded; removed files are dropped; unchanged
  files are reused straight from the existing index (their vectors are reconstructed from
  FAISS, no re-embedding).
* A full rebuild happens automatically when there is no manifest (first run / older index)
  or when the embedding model / chunk settings changed since the last training.
* Index files are written atomically (temp file + rename), so a crash or a refresh in the
  middle of training can never leave a half-written index behind.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from datetime import datetime, timezone

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src import documents

INDEX_FILE = "index.faiss"
META_FILE = "metadata.pkl"
MANIFEST_FILE = "manifest.json"
MANIFEST_VERSION = 1
EXCLUDED_NAMES = {"readme.md"}  # documentation files that live next to the documents

_embedder = None
_embedder_name = None
_embedder_lock = threading.Lock()
_train_lock = threading.Lock()


class TrainingBusyError(RuntimeError):
    """Raised when training is requested while another training run is active."""


# --------------------------------------------------------------------------- embedder
def get_embedder():
    """Load the sentence-transformer once. Tries the local cache first so the app starts
    instantly and works with no internet after the model has been downloaded once."""
    global _embedder, _embedder_name
    with _embedder_lock:
        if _embedder is not None and _embedder_name == config.EMBEDDING_MODEL:
            return _embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -r requirements.txt"
            ) from exc
        name = config.EMBEDDING_MODEL
        try:
            model = SentenceTransformer(name, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet: download it (needs internet once)
            try:
                model = SentenceTransformer(name)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"The embedding model '{name}' could not be loaded. The first run needs an "
                    "internet connection to download it once; after that it works fully offline. "
                    f"Details: {exc}"
                ) from exc
        _embedder, _embedder_name = model, name
        return _embedder


def _encode(texts: list, batch_size: int = None, progress=None) -> np.ndarray:
    """Embed texts in batches -> normalised float32 matrix (cosine similarity via inner product)."""
    batch_size = batch_size or config.EMBED_BATCH_SIZE
    embedder = get_embedder()
    parts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vecs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        parts.append(np.asarray(vecs, dtype="float32"))
        if progress:
            progress(min(start + len(batch), len(texts)), len(texts))
    if not parts:
        return np.zeros((0, 0), dtype="float32")
    matrix = np.ascontiguousarray(np.vstack(parts), dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


# --------------------------------------------------------------------------- files / manifest
def _paths(index_dir: str):
    return (os.path.join(index_dir, INDEX_FILE), os.path.join(index_dir, META_FILE),
            os.path.join(index_dir, MANIFEST_FILE))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _file_info(path: str) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def list_document_files(docs_dir: str = None) -> list:
    docs_dir = docs_dir or config.RAW_DOCS_DIR
    if not os.path.isdir(docs_dir):
        return []
    names = []
    for fname in sorted(os.listdir(docs_dir), key=str.lower):
        if fname.startswith(".") or fname.lower() in EXCLUDED_NAMES:
            continue
        if os.path.isfile(os.path.join(docs_dir, fname)) and documents.is_supported(fname):
            names.append(fname)
    return names


def load_manifest(index_dir: str = None):
    index_dir = index_dir or config.FAISS_INDEX_DIR
    path = _paths(index_dir)[2]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and isinstance(data.get("files"), dict) else None
    except (OSError, ValueError):
        return None


def _signature() -> dict:
    return {"embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP}


def index_token(index_dir: str = None):
    """Changes whenever the index is re-trained (used to reload caches). None = no index."""
    index_dir = index_dir or config.FAISS_INDEX_DIR
    index_path, meta_path, _ = _paths(index_dir)
    try:
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return None
        st = os.stat(index_path)
        return f"{st.st_mtime_ns}-{st.st_size}"
    except OSError:
        return None


def get_training_status(docs_dir: str = None, index_dir: str = None) -> dict:
    """Cheap (stat-based) comparison of the documents folder against the trained index."""
    docs_dir = docs_dir or config.RAW_DOCS_DIR
    index_dir = index_dir or config.FAISS_INDEX_DIR
    manifest = load_manifest(index_dir)
    index_exists = index_token(index_dir) is not None
    tracked = (manifest or {}).get("files", {})

    docs, on_disk = [], set()
    for name in list_document_files(docs_dir):
        on_disk.add(name)
        path = os.path.join(docs_dir, name)
        info = _file_info(path)
        entry = tracked.get(name)
        if entry is None:
            state = "new"
        elif entry.get("size") == info["size"] and entry.get("mtime_ns") == info["mtime_ns"]:
            state = "trained"
        else:  # touched: confirm by content hash so a simple re-save is not "modified"
            state = "trained" if entry.get("sha256") == _sha256(path) else "modified"
        docs.append({"name": name, "size": info["size"], "status": state,
                     "chunks": (entry or {}).get("chunks")})

    removed = sorted(n for n in tracked if n not in on_disk)
    needs_full = bool(manifest) and manifest.get("signature") != _signature()
    legacy = index_exists and manifest is None
    pending_docs = [d for d in docs if d["status"] != "trained"]
    return {
        "index_exists": index_exists,
        "legacy": legacy,
        "documents": docs,
        "removed": removed,
        "needs_full_retrain": needs_full,
        "pending": bool(pending_docs or removed or needs_full or (docs and not index_exists)),
        "n_pending": len(pending_docs) + len(removed),
        "n_chunks": (manifest or {}).get("n_chunks"),
        "trained_at": (manifest or {}).get("trained_at"),
        "embedding_model": ((manifest or {}).get("signature") or {}).get("embedding_model"),
    }


# --------------------------------------------------------------------------- chunking
def chunk_text(source: str, text: str, chunk_size: int = None, overlap: int = None) -> list:
    chunk_size = config.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for idx, piece in enumerate(p.strip() for p in splitter.split_text(text)):
        if piece:
            chunks.append({"source": source, "chunk_id": idx, "content": piece})
    return chunks


def chunk_documents(docs: list, chunk_size: int = None, overlap: int = None) -> list:
    out = []
    for doc in docs:
        out.extend(chunk_text(doc["source"], doc["text"], chunk_size, overlap))
    return out


def load_documents(docs_dir: str = None) -> list:
    """Load every supported file in docs_dir into {source, text} dicts."""
    docs_dir = docs_dir or config.RAW_DOCS_DIR
    docs = []
    for name in list_document_files(docs_dir):
        try:
            text = documents.extract_file(os.path.join(docs_dir, name))
        except ValueError:
            continue
        if text.strip():
            docs.append({"source": name, "text": text})
    return docs


# --------------------------------------------------------------------------- training
def _atomic_write(path: str, writer) -> None:
    tmp = path + ".tmp"
    writer(tmp)
    os.replace(tmp, path)


def _clear_index(index_dir: str) -> None:
    for p in _paths(index_dir):
        try:
            os.remove(p)
        except OSError:
            pass


def _load_existing(index_dir: str):
    """Existing (chunks, embeddings), or None if the index is missing/inconsistent."""
    index_path, meta_path, _ = _paths(index_dir)
    try:
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            chunks = pickle.load(f)
        if index.ntotal != len(chunks):
            return None
        emb = index.reconstruct_n(0, index.ntotal) if index.ntotal else np.zeros((0, index.d), "float32")
        return chunks, np.asarray(emb, dtype="float32")
    except Exception:  # noqa: BLE001 - any problem => rebuild from scratch
        return None


def train(docs_dir: str = None, index_dir: str = None, force_full: bool = False,
          progress_cb=None) -> dict:
    """Train (index) the knowledge base. progress_cb(fraction 0..1, message) is optional.

    Returns a summary dict: full, changed, added, removed, unchanged, skipped,
    n_docs, n_chunks, seconds.
    """
    docs_dir = docs_dir or config.RAW_DOCS_DIR
    index_dir = index_dir or config.FAISS_INDEX_DIR
    if not _train_lock.acquire(blocking=False):
        raise TrainingBusyError("Training is already running. Please wait for it to finish.")
    started = time.perf_counter()

    def report(frac, msg):
        if progress_cb:
            progress_cb(max(0.0, min(1.0, frac)), msg)

    try:
        os.makedirs(docs_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        index_path, meta_path, manifest_path = _paths(index_dir)

        report(0.0, "Scanning documents…")
        files = {}
        for name in list_document_files(docs_dir):
            path = os.path.join(docs_dir, name)
            files[name] = {**_file_info(path), "sha256": _sha256(path)}

        manifest = load_manifest(index_dir)
        signature = _signature()
        full = bool(force_full) or manifest is None or manifest.get("signature") != signature
        existing = None if full else _load_existing(index_dir)
        if existing is None:
            full = True

        tracked = {} if full else manifest["files"]
        removed = sorted(n for n in tracked if n not in files)
        to_process = [n for n, info in files.items()
                      if full or n not in tracked or tracked[n].get("sha256") != info["sha256"]]
        unchanged = [n for n in files if n not in to_process]

        if not full and not to_process and not removed:
            for n in unchanged:  # refresh timestamps so status stays "trained"
                tracked[n].update(size=files[n]["size"], mtime_ns=files[n]["mtime_ns"])
            _atomic_write(manifest_path, lambda p: _dump_json(p, manifest))
            report(1.0, "Everything is already trained.")
            return {"full": False, "changed": False, "added": [], "removed": [], "unchanged": len(unchanged),
                    "skipped": [], "n_docs": len(files), "n_chunks": manifest.get("n_chunks", 0),
                    "seconds": round(time.perf_counter() - started, 2)}

        # 1) read + chunk the new/changed files
        new_chunks, skipped, trained_files = [], [], []
        for i, name in enumerate(to_process):
            report(0.25 * i / max(len(to_process), 1), f"Reading {name}…")
            try:
                text = documents.extract_file(os.path.join(docs_dir, name))
            except ValueError as exc:
                skipped.append((name, str(exc)))
                continue
            chunks = chunk_text(name, text) if text.strip() else []
            if not chunks:
                skipped.append((name, "no extractable text (a scanned PDF needs OCR first)"))
                continue
            new_chunks.extend(chunks)
            trained_files.append((name, len(chunks)))

        # 2) keep vectors of unchanged files
        if full:
            kept_chunks, kept_emb = [], None
        else:
            old_chunks, old_emb = existing
            drop = set(removed) | set(to_process)
            keep_mask = np.array([c["source"] not in drop for c in old_chunks], dtype=bool)
            kept_chunks = [c for c, k in zip(old_chunks, keep_mask) if k]
            kept_emb = old_emb[keep_mask] if len(old_emb) else None

        # 3) embed the new chunks
        new_emb = None
        if new_chunks:
            report(0.25, f"Embedding {len(new_chunks)} chunks…")
            new_emb = _encode(
                [c["content"] for c in new_chunks],
                progress=lambda done, total: report(0.25 + 0.70 * done / total,
                                                    f"Embedding chunks… {done}/{total}"),
            )

        all_chunks = kept_chunks + new_chunks
        pieces = [m for m in (kept_emb, new_emb) if m is not None and len(m)]
        if not all_chunks or not pieces:
            _clear_index(index_dir)
            report(1.0, "No readable documents - knowledge base is empty.")
            return {"full": full, "changed": True, "added": [], "removed": removed, "unchanged": 0,
                    "skipped": skipped, "n_docs": 0, "n_chunks": 0,
                    "seconds": round(time.perf_counter() - started, 2)}

        matrix = np.ascontiguousarray(np.vstack(pieces), dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])  # inner product on unit vectors = cosine
        index.add(matrix)

        # 4) save atomically (manifest last: it marks the index as complete)
        report(0.96, "Saving index…")
        new_files = {}
        for name in unchanged:
            new_files[name] = {**files[name], "chunks": tracked[name].get("chunks")}
        for name, n in trained_files:
            new_files[name] = {**files[name], "chunks": n}
        _atomic_write(index_path, lambda p: faiss.write_index(index, p))
        _atomic_write(meta_path, lambda p: _dump_pickle(p, all_chunks))
        new_manifest = {
            "version": MANIFEST_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "signature": signature,
            "n_chunks": len(all_chunks),
            "files": new_files,
        }
        _atomic_write(manifest_path, lambda p: _dump_json(p, new_manifest))
        report(1.0, "Training complete.")
        return {
            "full": full, "changed": True,
            "added": [n for n, _ in trained_files], "removed": removed,
            "unchanged": len(unchanged), "skipped": skipped,
            "n_docs": len(new_files), "n_chunks": len(all_chunks),
            "seconds": round(time.perf_counter() - started, 2),
        }
    finally:
        _train_lock.release()


def _dump_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _dump_pickle(path: str, data) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def build_index(docs_dir: str = None, index_dir: str = None) -> int:
    """Backward-compatible helper: full rebuild, returns the number of chunks."""
    summary = train(docs_dir, index_dir, force_full=True)
    if summary["n_chunks"] == 0:
        raise ValueError(
            f"No readable documents found in {docs_dir or config.RAW_DOCS_DIR}. "
            "Add .pdf/.txt/.md/.docx files, then re-run."
        )
    return summary["n_chunks"]


# --------------------------------------------------------------------------- document management
def add_document_bytes(name: str, data: bytes, docs_dir: str = None):
    """Save an uploaded document into the knowledge-base folder.
    Returns (saved_name, outcome) where outcome is 'added', 'replaced' or 'unchanged'."""
    docs_dir = docs_dir or config.RAW_DOCS_DIR
    safe = documents.safe_filename(name)
    if not safe or not documents.is_supported(safe):
        raise ValueError(
            f"'{name}' is not a supported file. Supported: {', '.join(config.SUPPORTED_EXTENSIONS)}"
        )
    if safe.lower() in EXCLUDED_NAMES:
        raise ValueError(f"'{safe}' is a reserved file name - please rename the file.")
    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, safe)
    outcome = "added"
    if os.path.exists(path):
        with open(path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == hashlib.sha256(data).hexdigest():
                return safe, "unchanged"
        outcome = "replaced"
    with open(path, "wb") as f:
        f.write(data)
    return safe, outcome


def delete_document(name: str, docs_dir: str = None) -> bool:
    docs_dir = docs_dir or config.RAW_DOCS_DIR
    if os.path.basename(name) != name:
        raise ValueError("Invalid file name.")
    path = os.path.join(docs_dir, name)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


# --------------------------------------------------------------------------- search
class KnowledgeBase:
    """Loads the trained FAISS index + metadata and exposes semantic search."""

    def __init__(self, index_dir: str = None):
        index_dir = index_dir or config.FAISS_INDEX_DIR
        index_path, meta_path, _ = _paths(index_dir)
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                "The knowledge base has not been trained yet. Upload documents and click "
                "'Train' (or run `python ingest.py`)."
            )
        self.version = index_token(index_dir)
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError("The knowledge-base index is inconsistent. Click 'Full retrain'.")
        self.embedder = get_embedder()

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list:
        return sorted({c["source"] for c in self.chunks}, key=str.lower)

    def search(self, query: str, k: int = None) -> list:
        k = k or config.TOP_K
        if self.index.ntotal == 0 or not query.strip():
            return []
        q_emb = np.asarray(self.embedder.encode([query], convert_to_numpy=True), dtype="float32")
        if q_emb.shape[1] != self.index.d:
            raise RuntimeError(
                "The embedding model no longer matches the trained index "
                f"({q_emb.shape[1]} vs {self.index.d} dimensions). Click 'Full retrain'."
            )
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
