"""
sawvial.validation.qa_checks
============================
Detects QA pairs that must not be used as training answers (or shown as findings).

The QA generator is template-based and EXTRACTIVE: "answers" are the evidence text, not
independently verified facts. Every pair is therefore labelled ``origin=template_from_evidence``
and ``verified=False``; these checks only remove pairs that fail basic sanity tests.

HARD issues  -> the pair is removed from the dataset (audit_dataset.py --fix) and from SFT.
SOFT issues  -> the pair stays in the dataset but is excluded from SFT training.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from sawvial.eligibility import is_usable_evidence, looks_like_reference
from sawvial.embeddings.embedder import _tokenize

HARD = {"question_too_short", "empty_answer", "answer_too_short", "answer_is_title_or_metadata",
        "answer_is_reference_list", "answer_unusable_evidence", "answer_off_topic"}
SOFT = {"answer_starts_mid_sentence", "answer_truncated", "question_not_specific"}
_GENERIC_Q = re.compile(r"^what does the source report about [\w\s-]+\?$", re.I)
_STOP = {"what", "does", "the", "source", "report", "about", "was", "were", "found", "regarding",
         "recommended", "effect", "observed", "related", "during", "under", "conditions", "condition"}


def qa_issues(ku: Dict[str, Any], qa: Dict[str, Any]) -> List[str]:
    q, a = (qa.get("question") or "").strip(), (qa.get("answer") or "").strip()
    out: List[str] = []
    if len(q) < 12:
        out.append("question_too_short")
    if not a:
        return out + ["empty_answer"]
    if len(a) < 40:
        out.append("answer_too_short")
    title = ((ku.get("source") or {}).get("title") or "").strip()
    if title and (a == title or (a.startswith(title) and len(a) < len(title) + 30)):
        out.append("answer_is_title_or_metadata")
    if looks_like_reference(a):
        out.append("answer_is_reference_list")
    elif qa.get("question_type") != "practice_recommendation" and not is_usable_evidence(a, title):
        out.append("answer_unusable_evidence")
    q_terms = {t for t in _tokenize(q) if len(t) > 3 and t not in _STOP}
    if q_terms and not q_terms & set(_tokenize(a)):
        out.append("answer_off_topic")
    if a[0].islower() or a[0] in ",.;:)]":
        out.append("answer_starts_mid_sentence")
    if a[-1] not in ".!?)\"'%":
        out.append("answer_truncated")
    if _GENERIC_Q.match(q):
        out.append("question_not_specific")
    return out


def repair_answer(answer: str) -> str:
    """Trim an extractive answer to complete sentences (a SUBSTRING of the evidence; nothing is
    added or rewritten): drop a leading mid-sentence fragment and a trailing cut-off sentence."""
    a = (answer or "").strip()
    if a and (a[0].islower() or a[0] in ",.;:)]"):
        m = re.search(r"[.!?]\s+(?=[A-Z0-9])", a)
        a = a[m.end():] if m else ""
    if a and a[-1] not in ".!?)\"'%":
        cut = max(a.rfind(". "), a.rfind("! "), a.rfind("? "))
        a = a[:cut + 1] if cut > 0 else ""
    return a.strip()


def sft_usable(ku: Dict[str, Any], qa: Dict[str, Any]) -> tuple:
    """-> (repaired_answer_or_None, issues). Usable = no hard issue and complete sentences after repair."""
    fixed = dict(qa, answer=repair_answer(qa.get("answer", "")))
    issues = qa_issues(ku, fixed)
    blocking = [i for i in issues if i in HARD or i in {"answer_starts_mid_sentence", "answer_truncated"}]
    return (None if blocking else fixed["answer"]), issues


def conflicting_questions(qa_rows: Iterable[Dict[str, Any]]) -> set:
    """Question texts that appear with more than one different answer (contradictory labels)."""
    answers: Dict[str, set] = defaultdict(set)
    for r in qa_rows:
        answers[(r.get("question") or "").strip().lower()].add((r.get("answer") or "").strip())
    return {q for q, a in answers.items() if len(a) > 1}


def is_hard(issues: List[str]) -> bool:
    return any(i in HARD for i in issues)
