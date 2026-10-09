"""
sawvial.index.retriever
==========================
Top-K retrieval over the FAISS index. Every returned result carries
the full traceable payload required by master-spec section 20:
knowledge_unit_id, score, evidence, finding, recommendation, domain,
subdomain, context, source, provenance — not just a bare score.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np

from sawvial.embeddings.embedder import TFIDFEmbedder, make_tfidf_vectorizer


def load_index_bundle(index_dir: str, dataset_dir: str) -> Dict[str, Any]:
    idx_path = Path(index_dir)
    ds_path = Path(dataset_dir)

    index = faiss.read_index(str(idx_path / "faiss.index"))
    with open(idx_path / "id_map.json", encoding="utf-8") as f:
        id_map = json.load(f)  # position(str) -> knowledge_unit_id
    with open(idx_path / "index_metadata.json", encoding="utf-8") as f:
        index_meta = json.load(f)

    units_by_id: Dict[str, Any] = {}
    with open(ds_path / "knowledge_units.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ku = json.loads(line)
                units_by_id[ku["id"]] = ku

    return {"index": index, "id_map": id_map, "index_meta": index_meta, "units_by_id": units_by_id}


def query(bundle: Dict[str, Any], embedder, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search the full TF-IDF vocabulary offline, or use the stored dense index."""
    if isinstance(embedder, TFIDFEmbedder):
        lexical = bundle.get("_lexical_search")
        if lexical is None:
            indexed_ids = [bundle["id_map"][str(pos)] for pos in range(bundle["index"].ntotal)]
            texts = [bundle["units_by_id"][unit_id]["retrieval"]["semantic_text"]
                     for unit_id in indexed_ids]
            vectorizer = make_tfidf_vectorizer(max_features=None)
            matrix = vectorizer.fit_transform(texts)
            lexical = bundle["_lexical_search"] = (vectorizer, matrix, indexed_ids)
        vectorizer, matrix, indexed_ids = lexical
        query_vector = vectorizer.transform([query_text])
        all_scores = (matrix @ query_vector.T).toarray().ravel()
        candidate_positions = np.flatnonzero(all_scores > 0)
        candidate_positions = candidate_positions[
            np.argsort(all_scores[candidate_positions])[::-1][:top_k]
        ]
        scores = all_scores[candidate_positions]
        indices = candidate_positions
    else:
        q_vec = embedder.encode([query_text]).astype(np.float32)
        scores, indices = bundle["index"].search(q_vec, top_k)

    results = []
    if isinstance(embedder, TFIDFEmbedder):
        ranked_results = zip(scores, indices)
    else:
        ranked_results = zip(scores[0], indices[0])
    for rank, (score, pos) in enumerate(ranked_results, start=1):
        if pos < 0:
            continue
        unit_id = indexed_ids[pos] if isinstance(embedder, TFIDFEmbedder) else bundle["id_map"].get(str(pos))
        ku = bundle["units_by_id"].get(unit_id)
        if not ku:
            continue
        results.append({
            "rank": rank,
            "score": float(score),
            "knowledge_unit_id": unit_id,
            "evidence": ku["knowledge"]["evidence"],
            "finding": ku["knowledge"]["finding"],
            "recommendation": ku["knowledge"]["recommendation"],
            "structured_facts": ku.get("structured_facts", []),
            "validation_status": ku.get("quality", {}).get("validation_status"),
            "primary_domain": ku["classification"]["primary_domain"],
            "subdomains": ku["classification"]["subdomains"],
            "context": ku["context"],
            "source": ku["source"],
            "provenance": ku["provenance"],
        })
    return results
