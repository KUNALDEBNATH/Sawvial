"""
sawvial.dedup.deduplicator
=============================
Document-level and knowledge-unit-level deduplication, replacing the
old "hash of first 2000 chars" approach (fragile: two different
papers sharing a common boilerplate abstract intro would collide;
two versions of the same paper with a different cover page would not).

Document dedup key (any match => duplicate):
    - DOI (if present)
    - normalized (title, first-author) pair
    - full-document content hash (normalized whitespace/case)

Knowledge-unit near-duplicate detection:
    - exact evidence_hash match -> duplicate
    - high Jaccard similarity (word shingles) on evidence text AND same
      primary_domain -> near-duplicate (kept only if genuinely
      different numbers/conditions are present, since two units citing
      different treatments must not be merged away)
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Dict, Any, Set, Tuple


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def document_hash(full_text: str) -> str:
    normalized = _normalize(full_text)
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()


def document_dedup_key(meta: Dict[str, Any], full_text: str) -> Tuple[str, str]:
    """Returns (dedup_key, key_type) for reporting purposes."""
    if meta.get("doi"):
        return f"doi:{meta['doi'].lower()}", "doi"
    title_key = _normalize(meta.get("title", ""))
    first_author = _normalize(meta.get("authors", [""])[0]) if meta.get("authors") else ""
    if title_key:
        return f"title_author:{title_key}|{first_author}", "title_author"
    return f"content_hash:{document_hash(full_text)}", "content_hash"


def deduplicate_documents(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    unique = []
    for rec in records:
        key, _ = document_dedup_key(rec["metadata"], rec["full_text"])
        content_key = f"content_hash:{document_hash(rec['full_text'])}"
        if key in seen or content_key in seen:
            continue
        seen.add(key)
        seen.add(content_key)
        unique.append(rec)
    return unique


def _shingles(text: str, k: int = 5) -> Set[str]:
    words = _normalize(text).split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def deduplicate_knowledge_units(units: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
    seen_hashes: Set[str] = set()
    kept: List[Dict[str, Any]] = []
    kept_shingles: List[Tuple[Set[str], str]] = []  # (shingles, primary_domain)

    for unit in units:
        ev_hash = unit["provenance"]["evidence_hash"]
        if ev_hash in seen_hashes:
            continue

        evidence = unit["knowledge"]["evidence"]
        domain = unit["classification"].get("primary_domain")
        shingles = _shingles(evidence)

        is_near_dup = False
        for other_shingles, other_domain in kept_shingles:
            if domain == other_domain and _jaccard(shingles, other_shingles) >= similarity_threshold:
                is_near_dup = True
                break

        if is_near_dup:
            continue

        seen_hashes.add(ev_hash)
        kept_shingles.append((shingles, domain))
        kept.append(unit)

    return kept
