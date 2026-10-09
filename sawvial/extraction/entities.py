"""
sawvial.extraction.entities
=============================
Detects scientific soil/water/conservation concepts in a paragraph and
normalizes surface-form variants (e.g. "SOC", "soil C", "organic
carbon") to a single canonical concept id using taxonomy.json's
`concept_aliases` table, instead of the old brittle `if term in text`
keyword check with no normalization.

This is deterministic pattern-based NLP (regex + alias table), not a
black-box classifier — every detection is explainable and traceable
back to the exact surface text that triggered it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict

from sawvial.config import Taxonomy

_WORD_BOUNDARY = r"(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"


@dataclass
class ConceptMatch:
    canonical: str
    surface_form: str
    domain: str | None
    subdomain: str | None
    start: int
    end: int


def _compile_alias_patterns(taxonomy: Taxonomy) -> List[tuple]:
    """Build (compiled_regex, canonical_concept) pairs, longest aliases
    first so multi-word aliases win over short substrings (e.g. 'soil
    organic carbon' matches before bare 'organic carbon')."""
    pairs = []
    for canonical, aliases in taxonomy.concept_aliases.items():
        all_forms = sorted(set(aliases + [canonical.replace("_", " ")]), key=len, reverse=True)
        for form in all_forms:
            escaped = re.escape(form)
            pattern = re.compile(_WORD_BOUNDARY.format(escaped), re.IGNORECASE)
            pairs.append((pattern, canonical))
    return pairs


def _compile_topic_patterns(taxonomy: Taxonomy) -> List[tuple]:
    """Every taxonomy topic/subdomain name (as its natural-language
    surface form) is also a detectable concept, mapped to its
    (domain, subdomain)."""
    pairs = []
    for topic, (domain, subdomain) in taxonomy.topic_to_domain_subdomain.items():
        form = topic.replace("_", " ")
        pattern = re.compile(_WORD_BOUNDARY.format(re.escape(form)), re.IGNORECASE)
        pairs.append((pattern, topic, domain, subdomain))
    return pairs


class ScientificEntityExtractor:
    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy
        self._alias_patterns = _compile_alias_patterns(taxonomy)
        self._topic_patterns = _compile_topic_patterns(taxonomy)

    def extract(self, text: str) -> List[ConceptMatch]:
        matches: List[ConceptMatch] = []
        seen_spans = set()

        for pattern, topic, domain, subdomain in self._topic_patterns:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                matches.append(ConceptMatch(
                    canonical=topic, surface_form=m.group(0),
                    domain=domain, subdomain=subdomain,
                    start=m.start(), end=m.end(),
                ))

        for pattern, canonical in self._alias_patterns:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                domain_subdomain = self.taxonomy.topic_to_domain_subdomain.get(canonical)
                domain, subdomain = domain_subdomain if domain_subdomain else (None, None)
                matches.append(ConceptMatch(
                    canonical=canonical, surface_form=m.group(0),
                    domain=domain, subdomain=subdomain,
                    start=m.start(), end=m.end(),
                ))

        matches.sort(key=lambda m: m.start)
        return matches

    def unique_concepts(self, text: str) -> List[str]:
        return sorted({m.canonical for m in self.extract(text)})
