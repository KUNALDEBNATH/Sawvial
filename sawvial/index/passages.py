"""
sawvial.index.passages
======================
Turns raw retrieval hits into the *passages* the LLM sees and the API cites.
Torch-free on purpose, so it can be imported and tested without the model stack.

Crop rule: when the question names a crop, only records that actually concern that crop
(labelled in context.crop or mentioned in the evidence text) may be returned. If there
are none, ``crop_missing`` is reported and the caller must say the knowledge base has
no evidence for that crop - generic records are never substituted.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sawvial.index.retriever import (CROP_SYNONYMS, _crop_pattern, detect_crops,
                                     expand_query, query as retrieve)

QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "between", "by", "can", "could",
    "did", "do", "does", "during", "effect", "find", "found", "for", "from", "give",
    "how", "i", "in", "into", "is", "it", "me", "my", "of", "on", "or", "over", "report",
    "reported", "result", "results", "say", "should", "show", "source", "study", "tell",
    "the", "their", "these", "this", "those", "to", "was", "were", "what", "when", "where",
    "which", "while", "who", "why", "with", "would", "you", "your",
    "agriculture", "agricultural", "crop", "crops", "farming", "farmer", "research",
    "system", "systems", "year", "years", "about", "much", "many", "need", "use", "best",
}


def normalize_term(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("oes") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 4:
        return word[:-1]
    return word


def _plural(crop: str) -> str:
    if crop.endswith("y"):
        return crop[:-1] + "ies"
    if crop.endswith(("o", "ch", "sh", "x", "z", "s")):
        return crop + "es"
    return crop + "s"


def focus_crop_labeled_evidence(evidence: str, crops: List[str], requested: set) -> tuple[str, bool]:
    """For records like 'wheat: ...; rice: ...' keep only the requested crop's segment."""
    markers = []
    for crop in crops:
        name = str(crop).strip().lower()
        if not name:
            continue
        pat = re.compile(rf"(?<!\w)(?:{re.escape(name)}|{re.escape(_plural(name))})\s*:", re.I)
        markers.extend((m.start(), m.end(), name) for m in pat.finditer(evidence))
    markers.sort()
    if not markers:
        return evidence, False
    targets = requested.intersection(crops)
    target_markers = [m for m in markers if m[2] in targets]
    if not targets or {m[2] for m in target_markers} != targets:
        return evidence, False
    segments = []
    for start, end, crop in target_markers:
        nxt = next((s for s, _, _ in markers if s > start), len(evidence))
        detail = evidence[end:nxt].strip(" \t\r\n;,:")
        if detail:
            segments.append(f"{crop}: {detail}")
    return ("; ".join(segments), True) if segments else (evidence, False)


_TEMPLATED_FINDING = re.compile(r"^\s*(?:a (?:range|value) of\b|findings? relate to\b)", re.I)


def clean_finding(finding: Optional[str]) -> Optional[str]:
    """Auto-generated 'a value of 2.0 was reported' findings carry no meaning; hide them."""
    if not finding or _TEMPLATED_FINDING.match(str(finding)):
        return None
    return finding


def retrieve_passages(bundle: Dict[str, Any], question: str, top_k: int,
                      embedder=None) -> Dict[str, Any]:
    """Return {"passages": [...], "crops": [...], "crop_missing": bool}."""
    crops = detect_crops(question, bundle.get("crop_names"))
    hits = retrieve(bundle, embedder, question, top_k=max(top_k * 4, 20), crops=crops or None)
    crop_missing = bool(crops) and not hits

    expanded = expand_query(question)
    crop_words = {w for c in crops for f in CROP_SYNONYMS.get(c, [c]) for w in f.lower().split()}
    crop_words |= {_plural(w) for w in crop_words}
    query_terms = {normalize_term(w) for w in re.findall(r"[a-z]{2,}", expanded.lower())
                   if w not in QUERY_STOPWORDS and w not in crop_words}
    requested = set(crops)

    passages: List[Dict[str, Any]] = []
    for r in hits:
        text = r["evidence"]
        ev_terms = {normalize_term(w) for w in re.findall(r"[a-z]{2,}", text.lower())}
        if query_terms and not query_terms & ev_terms:
            continue                                 # shares only the crop name, not the topic
        labels = [str(c).strip().lower() for c in r["context"].get("crop", [])
                  if isinstance(c, str) and c.strip()]
        focused, crop_focused = focus_crop_labeled_evidence(text, labels, requested)
        context = {"crop": sorted(set(labels) & requested)} if crop_focused else r["context"]
        crop_labeled = bool(set(labels) & requested)
        crop_mentions = sum(len(_crop_pattern(CROP_SYNONYMS.get(c, [c])).findall(text)) for c in crops)
        if crops and not (crop_labeled or crop_mentions >= 2):
            continue         # a single passing mention is not crop-specific evidence
        src, prov = r["source"], r["provenance"]
        passages.append({
            "id": r["knowledge_unit_id"], "document_id": prov.get("document_id"),
            "title": src.get("title"), "authors": src.get("authors") or [],
            "year": src.get("year"), "journal": src.get("journal"),
            "doi": src.get("doi"), "url": src.get("url"),
            "page": prov.get("page"), "section": prov.get("section"),
            "text": focused,
            "finding": None if crop_focused else clean_finding(r["finding"]),
            "recommendation": None if crop_focused else r["recommendation"],
            "context": context,
            "structured_facts": [] if crop_focused else r["structured_facts"],
            "crop_focused": crop_focused, "crop_labeled": crop_labeled,
            "crop_mentions": crop_mentions, "status": r["validation_status"],
            "score": r["score"], "rank": len(passages) + 1,
        })
        if len(passages) == top_k:
            break
    if crops and not passages:
        crop_missing = True
    return {"passages": passages, "crops": crops, "crop_missing": crop_missing}


def get_passages(bundle, embedder, question, top_k):
    """Backward-compatible wrapper returning just the passage list."""
    return retrieve_passages(bundle, question, top_k, embedder)["passages"]
