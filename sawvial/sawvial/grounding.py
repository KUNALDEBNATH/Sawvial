"""
sawvial.grounding
=================
Meaningful, testable checks that an answer is supported by the evidence it cites.

A passage being *retrieved* does not make an answer grounded. A model sentence counts as
supported only if ALL of these hold:

  1. it carries at least one citation ``[ID]`` that exists in the retrieved passages
     (invented IDs are rejected and reported);
  2. enough of the sentence's content words occur in the cited passage text; and
  3. every number in the sentence occurs in the cited passage text.

Sentences that fail are dropped from the visible answer (never presented as research
findings). This is a lexical support check: it catches fabricated sources, uncited claims
and invented numbers, but it cannot prove a paraphrase is scientifically correct. Domain
review by an agronomist is still required before the answers are relied upon.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from sawvial.embeddings.embedder import _tokenize

CITE = re.compile(r"\s*\[\s*([A-Z]+_\d+(?:\s*[,;]\s*[A-Z]+_\d+)*)\s*\]")
CITE_ID = re.compile(r"[A-Z]+_\d+")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")
_REFUSAL = re.compile(
    r"^(?:the (?:provided )?(?:sources|records|dataset)\b.{0,60}\b(?:do(?:es)? not|don't|cannot)\b|"
    r"(?:there is|i could(?: not|n't) find) (?:no|not enough)\b)", re.I)

# Overclaiming words: a sentence may only use them if the cited evidence uses them too
# ("doubles", "always", "guaranteed", "miracle" ... are not supported just because the topic words match).
_ABSOLUTE = re.compile(r"\b(doubl\w*|tripl\w*|halv\w*|eliminat\w*|guarante\w*|always|never|every\w*|all|completely|"
                       r"entirely|only|miracle\w*|cure\w*|unlimited|best|worst|perfect\w*|no)\b", re.I)

# fraction of a sentence's content terms that must occur in the cited text
MIN_TERM_SUPPORT = 0.40
MIN_SHARED_TERMS = 3

_STOP = {
    "the", "and", "for", "with", "that", "this", "these", "those", "from", "was", "were", "are",
    "has", "have", "had", "been", "being", "its", "their", "which", "also", "than", "then",
    "can", "may", "not", "but", "into", "such", "more", "most", "over", "under", "between",
    "study", "studies", "record", "records", "source", "sources", "reported", "found", "show",
    "shows", "showed", "according", "dataset", "research", "evidence", "result", "results",
}


def find_citations(text: str) -> set:
    return {i for group in CITE.findall(text or "") for i in CITE_ID.findall(group)}


def strip_citations(text: str) -> str:
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", CITE.sub("", text or "")).strip()


def _terms(text: str) -> set:
    return {t for t in _tokenize(strip_citations(text)) if len(t) >= 3 and t not in _STOP and not t.isdigit()}


def _numbers(text: str) -> set:
    return {n.replace(",", "") for n in _NUMBER.findall(strip_citations(text))}


def split_sentences(text: str) -> List[str]:
    """Split on sentence ends and bullet/line breaks; keep citation markers attached."""
    out: List[str] = []
    for line in re.split(r"\n+", text or ""):
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).replace("**", "").strip()
        if not line:
            continue
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", line):
            piece = piece.strip()
            if not piece:
                continue
            if out and not strip_citations(piece).strip(" .;,"):      # a lone "[ID]" after a period
                out[-1] = f"{out[-1]} {piece}"
            else:
                out.append(piece)
    return out


def _supports(sentence: str, evidence: str) -> Tuple[bool, str]:
    s_terms, e_terms = _terms(sentence), _terms(evidence)
    if not s_terms:
        return False, "no_content_terms"
    shared = len(s_terms & e_terms)
    need = min(len(s_terms), max(MIN_SHARED_TERMS, math.ceil(MIN_TERM_SUPPORT * len(s_terms))))
    if shared < need:
        return False, f"low_term_overlap:{shared}/{len(s_terms)}"
    missing = _numbers(sentence) - _numbers(evidence)
    if missing:
        return False, "number_not_in_evidence:" + ",".join(sorted(missing))
    ev_words = set(re.findall(r"[a-z]+", evidence.lower()))
    for m in _ABSOLUTE.finditer(strip_citations(sentence)):
        stem = m.group(1).lower()[:5]
        if not any(w.startswith(stem) for w in ev_words):
            return False, f"overclaim_not_in_evidence:{m.group(1).lower()}"
    return True, "ok"


def assess(raw_answer: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sentence-by-sentence support report for a raw (citation-bearing) model answer."""
    by_id = {p["id"]: p for p in passages}
    cited_anywhere = find_citations(raw_answer)
    invalid = sorted(cited_anywhere - set(by_id))
    sentences = []
    for sent in split_sentences(raw_answer):
        ids = [i for g in CITE.findall(sent) for i in CITE_ID.findall(g)]
        valid_ids = [i for i in dict.fromkeys(ids) if i in by_id]
        clean = strip_citations(sent)
        if _REFUSAL.match(clean):
            sentences.append({"text": clean, "ids": ids, "valid_ids": valid_ids,
                              "supported": False, "refusal": True, "reason": "refusal"})
            continue
        if not valid_ids:
            reason = "only_invalid_citations" if ids else "no_citation"
            sentences.append({"text": clean, "ids": ids, "valid_ids": [], "supported": False,
                              "refusal": False, "reason": reason})
            continue
        evidence = " ".join((by_id[i].get("text") or "") + " " + (by_id[i].get("recommendation") or "")
                            for i in valid_ids)
        ok, why = _supports(clean, evidence)
        sentences.append({"text": clean, "ids": ids, "valid_ids": valid_ids,
                          "supported": ok, "refusal": False, "reason": why})
    claims = [s for s in sentences if not s["refusal"]]
    n_supported = sum(s["supported"] for s in claims)
    return {
        "sentences": sentences,
        "invalid_citations": invalid,
        "n_claims": len(claims),
        "n_supported": n_supported,
        "citation_coverage": (n_supported / len(claims)) if claims else 0.0,
        "is_refusal": bool(sentences) and not claims,
    }


def compose_supported_answer(assessment: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Visible answer made only of supported sentences, plus the IDs that back it."""
    keep = [s for s in assessment["sentences"] if s["supported"]]
    text = " ".join(s["text"] for s in keep).strip()
    ids = list(dict.fromkeys(i for s in keep for i in s["valid_ids"]))
    return text, ids


def extractive_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Fallback that quotes the best-matching evidence sentences verbatim.

    Supported by construction (it is the evidence itself). Returns None if nothing in the
    passages actually matches the question.
    """
    q_terms = _terms(question)
    best = None
    for rank, p in enumerate(passages):
        text = re.sub(r"\s+", " ", p.get("text") or "").strip()
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.strip().split()) >= 6]
        for i, sent in enumerate(sents):
            if sent[0].islower():
                continue                                   # mid-sentence fragment
            score = len(q_terms & _terms(sent)) - rank * 0.05
            if best is None or score > best[0]:
                best = (score, p, sents[i:i + 2])
    if not best or best[0] < 2:
        return None
    _, passage, sents = best
    out = " ".join(sents)
    if len(out) > 480:
        out = sents[0][:480].rsplit(" ", 1)[0]
    if out and out[-1] not in ".!?":
        out += "..."
    return f"According to the retrieved research records: {out}", passage


def excerpt_answer(passages: List[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Last resort when the model's text could not be verified and no sentence matches the question closely:
    return the best-ranked passage's first complete sentences VERBATIM, clearly labelled. Never invented text;
    None if no passage contains complete sentences."""
    for p in sorted(passages, key=lambda x: -float(x.get("score") or 0)):
        text = re.sub(r"\s+", " ", p.get("text") or "").strip()
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.split()) >= 6 and x[0].isupper()]
        if sents:
            out = " ".join(sents[:2])
            if len(out) > 480:
                out = sents[0][:480].rsplit(" ", 1)[0] + "..."
            return f"The most relevant passage found says: {out}", p
    return None


def evidence_level(cited_ids: List[str], passages: List[Dict[str, Any]]) -> str:
    """'multiple_sources' if the cited records come from >1 document, else 'single_source'."""
    docs = {p.get("document_id") or p["id"] for p in passages if p["id"] in set(cited_ids)}
    return "multiple_sources" if len(docs) > 1 else "single_source"
