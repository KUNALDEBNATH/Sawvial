"""
sawvial.index.faiss_builder
==============================
Builds the FAISS index as a DERIVED artifact from the already-valid
canonical JSONL dataset (never the reverse — canonical dataset first,
per master-spec section 18).

Each vector maps deterministically to a `knowledge_unit_id` via an
explicit id-map file persisted alongside the index, rather than
relying on array position alone (sec. 20) — this makes the index safe
to rebuild partially or merge later.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def persist_index(index_dir: str, index: faiss.IndexFlatIP, vectors: np.ndarray,
                   unit_ids: List[str], embedding_meta: Dict[str, Any],
                   dataset_stats: Dict[str, Any]) -> None:
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "faiss.index"))
    np.save(str(out / "vectors.npy"), vectors)

    # Explicit position -> knowledge_unit_id map (never rely on array
    # position alone when the index is later merged/rebuilt).
    id_map = {str(pos): uid for pos, uid in enumerate(unit_ids)}
    with open(out / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)

    metadata = {
        "embedding": embedding_meta,
        "faiss_index_type": "IndexFlatIP (exact cosine, vectors L2-normalized)",
        "n_vectors": int(index.ntotal),
        "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "dataset_stats_summary": {
            "total_knowledge_units": dataset_stats.get("total_knowledge_units"),
            "validated": dataset_stats.get("validation_status_distribution", {}).get("validated"),
        },
    }
    with open(out / "index_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
