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
    """Returns (dedup_key, key_type) for reporting purposes.

    Title-only matching is unsafe: many agricultural papers share short, generic titles
    ("Effect of irrigation on crop yield") and would wrongly collide if author information is
    missing or empty. The title+author key is only used when a real first-author name is
    available; otherwise we fall back to the full-document content hash, which cannot produce a
    false match between two different papers.
    """
    if meta.get("doi"):
        return f"doi:{meta['doi'].lower()}", "doi"
    title_key = _normalize(meta.get("title", ""))
    first_author = _normalize(meta.get("authors", [""])[0]) if meta.get("authors") else ""
    if title_key and first_author:
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


_NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")


def _numeric_fingerprint(evidence: str) -> frozenset:
    """Numbers mentioned in the evidence (commas stripped), used to tell apart two passages
    that read almost the same but report different values, ranges or conditions (e.g. two
    fertilizer-dose trials written with near-identical wording but different kg/ha figures)."""
    return frozenset(n.replace(",", "") for n in _NUMBER.findall(evidence or ""))


def _crop_fingerprint(unit: Dict[str, Any]) -> frozenset:
    crops = (unit.get("context") or {}).get("crop") or []
    return frozenset(str(c).strip().lower() for c in crops if isinstance(c, str) and c.strip())


def deduplicate_knowledge_units(units: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Near-duplicate wording alone is not enough to merge two findings away: two passages can be
    worded almost identically (same template sentence, same domain) while reporting a different
    crop, treatment, condition or numerical result - each of those is a distinct scientific
    finding and must be preserved. A unit is only dropped as a near-duplicate when the wording is
    similar AND it reports the same crop(s) AND the same numbers (dose, yield, pH, etc.); any
    difference there means the two units are kept side by side, not merged.
    """
    seen_hashes: Set[str] = set()
    kept: List[Dict[str, Any]] = []
    kept_fingerprints: List[Tuple[Set[str], str, frozenset, frozenset]] = []  # (shingles, domain, numbers, crops)

    for unit in units:
        ev_hash = unit["provenance"]["evidence_hash"]
        if ev_hash in seen_hashes:
            continue

        evidence = unit["knowledge"]["evidence"]
        domain = unit["classification"].get("primary_domain")
        shingles = _shingles(evidence)
        numbers = _numeric_fingerprint(evidence)
        crops = _crop_fingerprint(unit)

        is_near_dup = False
        for other_shingles, other_domain, other_numbers, other_crops in kept_fingerprints:
            if (domain == other_domain
                    and _jaccard(shingles, other_shingles) >= similarity_threshold
                    and numbers == other_numbers        # same reported values/conditions
                    and crops == other_crops):           # same crop(s) discussed
                is_near_dup = True
                break

        if is_near_dup:
            continue

        seen_hashes.add(ev_hash)
        kept_fingerprints.append((shingles, domain, numbers, crops))
        kept.append(unit)

    return kept
