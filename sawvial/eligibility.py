"""Single source of truth for *which knowledge records may support an answer*.

Status meanings (used everywhere - build, index, API, research, evaluation):

  validated      Passed the automated checks in sawvial/validation/validator.py.
                 NOT reviewed by a person. Answerable by default.
  verified       Approved by a named human reviewer (online-research workflow).
                 Answerable.
  needs_review   Borderline automatic score or incomplete extraction.
                 Never answerable by default (opt-in via SAWVIAL_ANSWERABLE_STATUSES).
  evidence_only  Solid text but no domain could be assigned. Kept for triage only.
  pending_review Freshly collected online record awaiting a human decision. Not answerable.
  rejected       Failed a hard check or was rejected by a reviewer. Never answerable.

The retrieval index is built ONLY from answerable records (so unsuitable records can
never occupy top-k slots), and the runtime re-checks the same rule defensively.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from sawvial.settings import env_list

STATUS_DEFINITIONS: Dict[str, str] = {
    "validated": "passed automated checks; not human-reviewed; answerable",
    "verified": "approved by a named human reviewer; answerable",
    "needs_review": "borderline or incomplete extraction; not answerable by default",
    "evidence_only": "no domain assigned; triage only; not answerable",
    "pending_review": "new online record awaiting review; not answerable",
    "rejected": "failed checks or rejected by reviewer; not answerable",
}
ALL_STATUSES = frozenset(STATUS_DEFINITIONS)
ANSWERABLE_STATUSES = frozenset(env_list("SAWVIAL_ANSWERABLE_STATUSES", "validated,verified"))
ELIGIBILITY_RULES_VERSION = 2

BAD_SECTIONS = {"references", "reference", "bibliography", "literature cited",
                "acknowledgements", "acknowledgments", "acknowledgement", "appendix"}

# ---- reference-list / citation detection ------------------------------------
_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_INITIALS = re.compile(r"\b[A-Z]\.(?:\s?[A-Z]\.)*")
_PAGES_END = re.compile(r"\d{1,4}\s*[,:]\s*\d{1,5}\s*[-–]\s*\d{1,5}\.?\s*$")
_VOL_PAGES = re.compile(r"\b\d{1,3}\s*\(\d{1,3}\)\s*[,:]\s*\d{1,5}\s*[-–]\s*\d{1,5}")
_DOI = re.compile(r"doi\.org|doi:\s*10\.", re.I)

# Instruction-like text inside a source document. Documents are untrusted input.
_INJECTION = re.compile(
    r"(ignore (?:all |any )?(?:the )?(?:previous|prior|above) (?:instructions|prompts?)|"
    r"disregard (?:all |any )?(?:the )?(?:previous|prior|above)|"
    r"you are now\b|new instructions?:|system prompt|"
    r"(?:reveal|print|show) (?:your|the) (?:system )?(?:prompt|instructions)|"
    r"act as (?:an?|the) |do not (?:cite|follow) the (?:sources|records))", re.I)


def looks_like_reference(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    has_year = bool(_YEAR.search(t))
    initials = len(_INITIALS.findall(t))
    pages = bool(_PAGES_END.search(t) or _VOL_PAGES.search(t))
    if has_year and pages:
        return True
    if has_year and initials >= 4:
        return True
    return bool(_DOI.search(t) and initials >= 2)


def injection_flags(text: str) -> list[str]:
    """Return a list of matched instruction-like phrases (empty list = clean)."""
    return sorted({m.group(0).lower().strip() for m in _INJECTION.finditer(text or "")})


def is_usable_evidence(evidence: str, title: Optional[str] = None) -> bool:
    ev = (evidence or "").strip()
    if len(ev) < 80 or len(ev.split()) < 12:
        return False
    if re.match(r"^(?:table|figure)\s+\d+\s*[.:]", ev, re.I) and len(ev.split()) < 24:
        return False
    if re.match(r"^(?:keywords?|index terms?)\b", ev, re.I) and len(ev.split()) < 35:
        return False
    if looks_like_reference(ev):
        return False
    if title:
        t = title.strip()
        if ev == t or (t in ev and len(ev) < len(t) + 20):
            return False
    return True


def is_answerable(ku: Dict[str, Any], statuses: frozenset = ANSWERABLE_STATUSES) -> bool:
    """True if this record may support a user-facing answer."""
    q = ku.get("quality", {})
    if q.get("validation_status") not in statuses:
        return False
    if q.get("flags"):                      # e.g. possible_prompt_injection
        return False
    src = ku.get("source", {})
    if (src.get("section") or "").lower() in BAD_SECTIONS:
        return False
    return is_usable_evidence(ku.get("knowledge", {}).get("evidence"), src.get("title"))
