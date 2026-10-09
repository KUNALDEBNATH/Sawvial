"""
sawvial.preprocessing.cleaner
==============================
Cleans raw PDF line text while explicitly PRESERVING scientific
notation that the old pipeline's `[^\\x09\\x0A\\x20-\\x7E]` regex
destroyed (±, ≤, ≥, °, µ, superscripts/subscripts, etc.).

Also removes repeated headers/footers/page-number artifacts detected
across the whole document rather than a blind "drop lines with < 3
words" rule, which used to eat short but scientifically important
lines like table captions ("Table 2. Soil pH by treatment.") and
short headings.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List

# Characters that MUST survive cleaning — scientific/measurement notation.
_PRESERVE_CHARS = "±≤≥°µΔσα β γ×÷–—’‘“”…" \
                  "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉"

_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}

# Controls and true non-printables only; scientific unicode symbols are kept.
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*(page\s+)?\d{1,4}\s*(of\s*\d{1,4})?\s*$", re.IGNORECASE)


def _fix_ligatures(text: str) -> str:
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    return text


def _dehyphenate(text: str) -> str:
    """Rejoin PDF line-break hyphenation like 'irriga-\ntion' -> 'irrigation',
    but do NOT touch legitimate compound scientific terms that keep their
    hyphen mid-line (e.g. 'water-use', 'soil-water')."""
    # Only join when hyphen is immediately followed by a line break in the
    # original block stream (represented here as '-\n' before block join,
    # so this must be applied on raw block text before blocks are joined
    # with spaces/newlines — see document_structure.py).
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def detect_repeated_lines(all_page_texts: List[str], min_repeat_ratio: float = 0.4) -> set:
    """Identify lines (e.g. running headers/footers) that repeat across a
    large fraction of pages — these are noise, unlike short captions which
    only appear once."""
    counter = Counter()
    n_pages = max(len(all_page_texts), 1)
    for page_text in all_page_texts:
        seen_this_page = set()
        for line in page_text.split("\n"):
            norm = line.strip().lower()
            if norm and norm not in seen_this_page:
                counter[norm] += 1
                seen_this_page.add(norm)
    threshold = max(2, int(n_pages * min_repeat_ratio))
    return {line for line, count in counter.items() if count >= threshold and len(line) < 120}


def clean_page_text(text: str, repeated_lines: set) -> str:
    """Clean a single page's text, preserving scientific notation and
    dropping only genuine noise (repeated headers/footers, bare page
    numbers, control characters)."""
    text = _fix_ligatures(text)
    text = _dehyphenate(text)
    text = _CONTROL_CHARS_RE.sub(" ", text)

    kept_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _PAGE_NUMBER_LINE_RE.match(stripped):
            continue
        if stripped.lower() in repeated_lines:
            continue
        kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
