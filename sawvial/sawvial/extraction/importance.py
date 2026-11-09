"""
sawvial.extraction.importance
================================
Scores how scientifically important a paragraph/candidate knowledge
unit is, combining multiple signals rather than one fixed keyword
list, per master-spec section 5.

Signals combined (each normalized to roughly 0..1 before weighting):

  - section_weight        : which document section it came from
  - concept_density       : taxonomy concept hits per 100 words
  - numeric_signal        : presence of numbers/units/percentages/stats
  - boost_phrase_signal   : presence of importance language patterns
  - length_penalty        : very short/very long paragraphs score lower

Final score is a weighted sum clipped to [0, 1]. This score feeds
`quality.scientific_importance_score` and the validator's accept /
needs_review / reject decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from sawvial.config import Taxonomy
from sawvial.extraction.numeric import extract_structured_facts


@dataclass
class ImportanceBreakdown:
    section_weight: float
    concept_density: float
    numeric_signal: float
    boost_phrase_signal: float
    length_penalty: float
    total: float


class ImportanceScorer:
    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy
        boost_terms = taxonomy.importance_signals.get("boost_terms", [])
        pattern = "|".join(re.escape(t) for t in sorted(boost_terms, key=len, reverse=True))
        self._boost_re = re.compile(pattern, re.IGNORECASE) if pattern else None
        self._section_weights = taxonomy.importance_signals.get("section_weights", {})

    def score(self, text: str, section: str, concept_hits: int) -> ImportanceBreakdown:
        words = text.split()
        n_words = max(len(words), 1)

        section_weight = self._section_weights.get(section, self._section_weights.get("unknown", 1.0))
        # normalize section weight (roughly 0.1 - 1.5 range in taxonomy) to 0..1
        section_norm = min(section_weight / 1.5, 1.0)

        concept_density_raw = (concept_hits / n_words) * 100.0
        concept_density = min(concept_density_raw / 8.0, 1.0)  # 8 hits/100w ~ saturating

        facts = extract_structured_facts(text)
        numeric_signal = min(len(facts) / 3.0, 1.0)

        boost_hits = len(self._boost_re.findall(text)) if self._boost_re else 0
        boost_phrase_signal = min(boost_hits / 4.0, 1.0)

        if n_words < 8:
            length_penalty = 0.3
        elif n_words > 400:
            length_penalty = 0.6
        else:
            length_penalty = 1.0

        total = (
            0.20 * section_norm +
            0.30 * concept_density +
            0.25 * numeric_signal +
            0.15 * boost_phrase_signal +
            0.10 * length_penalty
        )
        total = max(0.0, min(total, 1.0))

        return ImportanceBreakdown(
            section_weight=round(section_norm, 3),
            concept_density=round(concept_density, 3),
            numeric_signal=round(numeric_signal, 3),
            boost_phrase_signal=round(boost_phrase_signal, 3),
            length_penalty=round(length_penalty, 3),
            total=round(total, 3),
        )
