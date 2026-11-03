"""
sawvial.index.faiss_builder
===========================
Builds the retrieval index as a DERIVED artifact of dataset/knowledge_units.jsonl.

Index contents (format_version 2), all in ``index/``:

    lexical_vectorizer.joblib   fitted TF-IDF vectorizer (re-used for every query)
    lexical_matrix.npz          sparse TF-IDF matrix, one row per answerable record
    id_map.json                 row position -> knowledge_unit_id
    index_metadata.json         what was indexed, eligibility rules, dataset fingerprint
    faiss.index                 OPTIONAL dense index (only when a sentence-transformers
                                model was available at build time) -> hybrid retrieval

Only records that pass ``sawvial.eligibility.is_answerable`` are indexed, so
evidence_only / needs_review / pending_review / rejected records can never occupy
top-k positions and then be discarded at query time.

Safe update: the new index is built in ``<index>.building``, re-loaded and sanity-checked,
and only then swapped in. The previous working index is kept as ``<index>.prev``. If
anything fails, the existing index is left untouched.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import scipy.sparse as sp

from sawvial.eligibility import (ANSWERABLE_STATUSES, ELIGIBILITY_RULES_VERSION,
                                 is_answerable)
from sawvial.embeddings.embedder import make_tfidf_vectorizer

log = logging.getLogger("sawvial.index")
FORMAT_VERSION = 2


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_index(units: Iterable[Dict[str, Any]], index_dir: str | Path,
                dataset_file: Optional[str | Path] = None,
                dense_embedder=None) -> Dict[str, Any]:
    """Build and atomically install the index. Returns the metadata dict."""
    index_dir = Path(index_dir)
    eligible = [ku for ku in units if is_answerable(ku)]
    if not eligible:
        raise ValueError("No answerable knowledge units: refusing to replace the existing index "
                         f"(answerable statuses: {sorted(ANSWERABLE_STATUSES)})")

    tmp = index_dir.with_name(index_dir.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        texts = [ku["retrieval"]["semantic_text"] for ku in eligible]
        ids = [ku["id"] for ku in eligible]

        vectorizer = make_tfidf_vectorizer(max_features=None)
        matrix = vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        joblib.dump(vectorizer, tmp / "lexical_vectorizer.joblib")
        sp.save_npz(tmp / "lexical_matrix.npz", matrix)
        (tmp / "id_map.json").write_text(json.dumps(ids), encoding="utf-8")

        dense_meta = None
        if dense_embedder is not None:
            import faiss  # optional dependency
            vectors = dense_embedder.encode(texts).astype(np.float32)
            dense_index = faiss.IndexFlatIP(vectors.shape[1])
            dense_index.add(vectors)
            faiss.write_index(dense_index, str(tmp / "faiss.index"))
            dense_meta = {"model_name": getattr(dense_embedder, "model_name", "unknown"),
                          "dimension": int(vectors.shape[1])}

        status_counts: Dict[str, int] = {}
        for ku in eligible:
            s = ku["quality"]["validation_status"]
            status_counts[s] = status_counts.get(s, 0) + 1
        meta = {
            "format_version": FORMAT_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "n_units": len(ids),
            "answerable_statuses": sorted(ANSWERABLE_STATUSES),
            "eligibility_rules_version": ELIGIBILITY_RULES_VERSION,
            "indexed_status_counts": status_counts,
            "dataset_file": str(dataset_file) if dataset_file else None,
            "dataset_sha256": file_sha256(Path(dataset_file)) if dataset_file else None,
            "sklearn_version": __import__("sklearn").__version__,      # the saved vectorizer is only valid for this version
            "lexical": {"type": "tfidf(1,2)-sparse-cosine", "n_features": int(matrix.shape[1])},
            "dense": dense_meta,
        }
        (tmp / "index_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        _sanity_check(tmp, len(ids), dense_meta is not None)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    _swap_in(tmp, index_dir)
    log.info("Index installed at %s (%d answerable records, dense=%s)",
             index_dir, len(ids), bool(dense_meta))
    return meta


def _sanity_check(path: Path, n: int, dense: bool) -> None:
    ids = json.loads((path / "id_map.json").read_text(encoding="utf-8"))
    matrix = sp.load_npz(path / "lexical_matrix.npz")
    vec = joblib.load(path / "lexical_vectorizer.joblib")
    if len(ids) != n or matrix.shape[0] != n:
        raise RuntimeError("index sanity check failed: row count mismatch")
    if matrix.shape[1] != len(vec.vocabulary_):
        raise RuntimeError("index sanity check failed: vocabulary mismatch")
    if dense:
        import faiss
        if faiss.read_index(str(path / "faiss.index")).ntotal != n:
            raise RuntimeError("index sanity check failed: dense vector count mismatch")


def _swap_in(tmp: Path, final: Path) -> None:
    prev = final.with_name(final.name + ".prev")
    had_old = final.exists()
    if had_old:
        if prev.exists():
            shutil.rmtree(prev)
        final.rename(prev)
    try:
        tmp.rename(final)
    except Exception:
        if had_old and prev.exists() and not final.exists():
            prev.rename(final)           # roll back to the previous working index
        raise
