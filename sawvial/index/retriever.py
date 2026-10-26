"""
sawvial.index.retriever
=======================
Query-time retrieval over the index written by ``faiss_builder``.

* The fitted TF-IDF vectorizer and sparse matrix are loaded once and re-used.
* Crop filtering happens BEFORE ranking, so a crop-specific question never has its
  top slots filled with generic records.
* When the index also contains a dense (sentence-transformers) FAISS index, lexical and
  dense rankings are merged with reciprocal-rank fusion (hybrid retrieval). Otherwise
  retrieval is lexical only. Both paths return the same payload.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import scipy.sparse as sp

from sawvial.index.faiss_builder import FORMAT_VERSION, file_sha256

log = logging.getLogger("sawvial.retriever")

REBUILD_HINT = "python build_dataset.py --skip-build --build-index"

# ---- agricultural vocabulary -------------------------------------------------
# canonical crop name (as used in dataset context.crop) -> surface forms
CROP_SYNONYMS: Dict[str, List[str]] = {
    "rice": ["rice", "paddy"], "wheat": ["wheat"], "maize": ["maize", "corn"],
    "cotton": ["cotton"], "sugarcane": ["sugarcane", "sugar cane"],
    "soybean": ["soybean", "soyabean", "soya bean"],
    "groundnut": ["groundnut", "peanut"], "sunflower": ["sunflower"],
    "mustard": ["mustard", "rapeseed"], "barley": ["barley"],
    "jowar": ["jowar", "sorghum"], "bajra": ["bajra", "pearl millet"],
    "ragi": ["ragi", "finger millet"], "lentil": ["lentil"],
    "chickpea": ["chickpea", "bengal gram"],
    "pigeon pea": ["pigeon pea", "pigeonpea", "arhar", "red gram"],
    "potato": ["potato"], "tomato": ["tomato"], "onion": ["onion"],
    "chilli": ["chilli", "chili"],
    # crops that may be asked about even if the knowledge base has no evidence for them
    "banana": ["banana"], "mango": ["mango"], "brinjal": ["brinjal", "eggplant", "aubergine"],
    "cabbage": ["cabbage"], "cauliflower": ["cauliflower"], "tea": ["tea plant", "tea garden"],
    "coffee": ["coffee"], "jute": ["jute"], "okra": ["okra"], "citrus": ["citrus", "orange", "lemon"],
    "apple": ["apple"], "grape": ["grape"], "papaya": ["papaya"], "coconut": ["coconut"],
    "turmeric": ["turmeric"], "ginger": ["ginger"], "millet": ["millet"],
}
GENERIC_CROP_WORDS = {"vegetables", "fruits", "pulses", "crop", "crops"}
MULTI_CROP_QUESTION = re.compile(
    r"\b(which|what|other|another|different|various|all|any|best)\s+(kind\s+of\s+|types?\s+of\s+)?crops?\b", re.I)

QUERY_SYNONYMS: Dict[str, List[str]] = {
    "wue": ["water use efficiency"], "iwue": ["irrigation water use efficiency"],
    "fym": ["farmyard manure"], "soc": ["soil organic carbon"], "npk": ["nitrogen phosphorus potassium"],
    "drip": ["drip irrigation", "micro irrigation"], "sprinkler": ["sprinkler irrigation"],
    "irrigate": ["irrigation"], "irrigating": ["irrigation"], "watering": ["irrigation"],
    "salty": ["salinity", "saline"], "salt": ["salinity"], "alkaline": ["sodicity", "alkalinity"],
    "fertile": ["fertility"], "fertiliser": ["fertilizer"], "fertilisers": ["fertilizer"],
    "mulch": ["mulching"], "waterlogged": ["waterlogging", "drainage"],
    "erode": ["erosion"], "eroded": ["erosion"], "runoff": ["run-off", "surface runoff"],
    "compost": ["organic manure", "organic amendment"], "manure": ["organic manure"],
    "drought": ["water stress", "moisture stress"], "harvesting": ["rainwater harvesting"],
    "tillage": ["conservation tillage", "zero tillage"], "yield": ["productivity"],
    "acidic": ["acidity", "lime"], "lime": ["liming"],
}
_ABBREV = {"WUE", "IWUE", "FYM", "SOC", "NPK"}


def expand_query(text: str) -> str:
    """Append agricultural synonyms for words/abbreviations found in the query."""
    words = re.findall(r"[A-Za-z]+", text)
    extra: List[str] = []
    for w in words:
        key = w.lower()
        if key in QUERY_SYNONYMS and (key not in {a.lower() for a in _ABBREV} or w.upper() in _ABBREV):
            extra.extend(QUERY_SYNONYMS[key])
    return f"{text} {' '.join(extra)}" if extra else text


def _crop_pattern(forms: List[str]) -> re.Pattern:
    alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})(?:s|es)?\b", re.I)


def detect_crops(question: str, extra_crops: Optional[set] = None) -> List[str]:
    """Canonical crop names explicitly asked about (empty for multi-crop/generic questions)."""
    if MULTI_CROP_QUESTION.search(question):
        return []
    table = dict(CROP_SYNONYMS)
    for c in (extra_crops or set()):
        if c not in table and c not in GENERIC_CROP_WORDS:
            table[c] = [c]
    return [crop for crop, forms in table.items() if _crop_pattern(forms).search(question)]


# ---- loading -------------------------------------------------------------------
def load_index_bundle(index_dir: str | Path, dataset_dir: str | Path,
                      load_dense_model: bool = True) -> Dict[str, Any]:
    idx, ds = Path(index_dir), Path(dataset_dir)
    meta_file = idx / "index_metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"No index at {idx}. Build it with: {REBUILD_HINT}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(f"Index at {idx} uses an old format (no format_version={FORMAT_VERSION}). "
                           f"Rebuild it with: {REBUILD_HINT}")

    ids = json.loads((idx / "id_map.json").read_text(encoding="utf-8"))
    vectorizer = joblib.load(idx / "lexical_vectorizer.joblib")
    matrix = sp.load_npz(idx / "lexical_matrix.npz").tocsr()
    if matrix.shape[0] != len(ids):
        raise RuntimeError(f"Corrupt index at {idx} (rows != ids). Rebuild with: {REBUILD_HINT}")

    ds_file = ds / "knowledge_units.jsonl"
    wanted = set(ids)
    units: Dict[str, Any] = {}
    crop_names: set = set()
    with open(ds_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ku = json.loads(line)
            if ku["id"] in wanted:
                units[ku["id"]] = ku
                crop_names.update(str(c).strip().lower() for c in ku.get("context", {}).get("crop", [])
                                  if isinstance(c, str) and c.strip())
    missing = wanted - set(units)
    if missing:
        raise RuntimeError(f"{len(missing)} indexed ids are missing from {ds_file}; "
                           f"the dataset changed since the index was built. Rebuild with: {REBUILD_HINT}")

    stale = bool(meta.get("dataset_sha256")) and meta["dataset_sha256"] != file_sha256(ds_file)
    if stale:
        log.warning("knowledge_units.jsonl changed since the index was built - new/changed records "
                    "are NOT searchable yet. Run: %s", REBUILD_HINT)

    bundle: Dict[str, Any] = {
        "ids": ids, "vectorizer": vectorizer, "matrix": matrix, "units_by_id": units,
        "index_meta": meta, "stale": stale, "crop_names": crop_names,
        "dense_index": None, "dense_embedder": None, "_crop_masks": {},
    }
    if meta.get("dense") and (idx / "faiss.index").exists():
        try:
            import faiss
            bundle["dense_index"] = faiss.read_index(str(idx / "faiss.index"))
            if load_dense_model:
                from sawvial.embeddings.embedder import SentenceTransformerEmbedder
                bundle["dense_embedder"] = SentenceTransformerEmbedder(meta["dense"]["model_name"])
        except Exception as exc:
            log.warning("Dense index present but unusable (%s); using lexical retrieval only", exc)
            bundle["dense_index"] = bundle["dense_embedder"] = None
    return bundle


_ALL_CROP_PATTERNS = {c: _crop_pattern(forms) for c, forms in CROP_SYNONYMS.items()}


def _crops_in_text(bundle: Dict[str, Any]) -> List[set]:
    """Per index row: the set of known crops mentioned in the evidence text (cached)."""
    if "_text_crops" not in bundle:
        rows = []
        for uid in bundle["ids"]:
            text = bundle["units_by_id"][uid]["knowledge"]["evidence"]
            rows.append({c for c, pat in _ALL_CROP_PATTERNS.items() if pat.search(text)})
        bundle["_text_crops"] = rows
    return bundle["_text_crops"]


def warm_up(bundle: Dict[str, Any]) -> None:
    """Pre-compute the crop-mention cache so the first user request is not slow."""
    _crops_in_text(bundle)


def crop_mask(bundle: Dict[str, Any], crop: str) -> np.ndarray:
    """Boolean mask over index rows: the record is genuinely about `crop`.

    A record qualifies if `crop` is one of its context.crop labels, or if the evidence text
    mentions the crop AND at most one other crop. A table/list that merely names the crop
    among many others is NOT crop-specific evidence.
    """
    cache = bundle["_crop_masks"]
    if crop not in cache:
        text_crops = _crops_in_text(bundle)
        pat = _ALL_CROP_PATTERNS.get(crop) or _crop_pattern([crop])
        mask = np.zeros(len(bundle["ids"]), dtype=bool)
        for pos, uid in enumerate(bundle["ids"]):
            ku = bundle["units_by_id"][uid]
            labels = {str(c).strip().lower() for c in ku.get("context", {}).get("crop", [])}
            if crop in labels:
                mask[pos] = True
            elif pat.search(ku["knowledge"]["evidence"]):
                mask[pos] = len(text_crops[pos] - {crop}) <= 1
        cache[crop] = mask
    return cache[crop]


def query(bundle: Dict[str, Any], embedder, query_text: str, top_k: int = 5,
          crops: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Ranked, eligible, optionally crop-filtered hits with the full traceable payload."""
    n = len(bundle["ids"])
    allowed = np.ones(n, dtype=bool)
    if crops:
        allowed = np.zeros(n, dtype=bool)
        for crop in crops:
            allowed |= crop_mask(bundle, crop)
        if not allowed.any():
            return []

    qvec = bundle["vectorizer"].transform([expand_query(query_text)])
    lex = (bundle["matrix"] @ qvec.T).toarray().ravel()
    lex = np.where(allowed, lex, 0.0)

    dense = None
    dense_embedder = embedder if hasattr(embedder, "encode") else bundle.get("dense_embedder")
    if bundle.get("dense_index") is not None and dense_embedder is not None:
        qd = dense_embedder.encode([query_text]).astype(np.float32)
        d_scores, d_idx = bundle["dense_index"].search(qd, n)
        dense = np.full(n, -1.0, dtype=np.float32)
        dense[d_idx[0][d_idx[0] >= 0]] = d_scores[0][d_idx[0] >= 0]
        dense = np.where(allowed, dense, -1.0)

    if dense is None:
        cand = np.flatnonzero(lex > 0)
        order = cand[np.argsort(lex[cand])[::-1][:top_k]]
        fused = {int(p): float(lex[p]) for p in order}
    else:                                   # reciprocal-rank fusion, k=60
        lex_rank = {int(p): r for r, p in enumerate(np.argsort(lex)[::-1][:200]) if lex[p] > 0}
        den_rank = {int(p): r for r, p in enumerate(np.argsort(dense)[::-1][:200]) if dense[p] > 0.3}
        scores = {}
        for p in set(lex_rank) | set(den_rank):
            scores[p] = sum(1.0 / (60 + r[p]) for r in (lex_rank, den_rank) if p in r)
        order = sorted(scores, key=scores.get, reverse=True)[:top_k]
        fused = {int(p): scores[p] for p in order}

    results = []
    for rank, pos in enumerate(fused, start=1):
        uid = bundle["ids"][pos]
        ku = bundle["units_by_id"][uid]
        primary = float(lex[pos]) if lex[pos] > 0 else float(dense[pos]) if dense is not None else 0.0
        results.append({
            "rank": rank, "score": primary, "fused_score": fused[pos],
            "dense_score": float(dense[pos]) if dense is not None else None,
            "knowledge_unit_id": uid,
            "evidence": ku["knowledge"]["evidence"],
            "finding": ku["knowledge"].get("finding"),
            "recommendation": ku["knowledge"].get("recommendation"),
            "structured_facts": ku.get("structured_facts", []),
            "validation_status": ku.get("quality", {}).get("validation_status"),
            "primary_domain": ku["classification"]["primary_domain"],
            "subdomains": ku["classification"]["subdomains"],
            "context": ku["context"], "source": ku["source"], "provenance": ku["provenance"],
        })
    return results
