"""
sawvial.extraction.context_extractor
======================================
Extracts agricultural CONTEXT — crop, variety, season, location, soil
type, irrigation method/source, agro-climatic zone — from a paragraph.

Per the master spec, crop / climate / technology / location are never
top-level domains; they are always attached as `context` on a
knowledge unit whose `primary_domain` is one of the three allowed
values.
"""
from __future__ import annotations

import re
from typing import Dict, List

from sawvial.config import Taxonomy

_WORD_BOUNDARY = r"(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"


class ContextExtractor:
    def __init__(self, taxonomy: Taxonomy):
        self.vocab = taxonomy.context_vocab
        self._compiled: Dict[str, List[tuple]] = {}
        for field_name, terms in self.vocab.items():
            self._compiled[field_name] = [
                (re.compile(_WORD_BOUNDARY.format(re.escape(t)), re.IGNORECASE), t)
                for t in sorted(terms, key=len, reverse=True)
            ]

    def extract(self, text: str) -> Dict[str, List[str]]:
        found: Dict[str, List[str]] = {}
        for field_name, patterns in self._compiled.items():
            hits = []
            for pattern, canonical in patterns:
                if pattern.search(text):
                    hits.append(canonical)
            found[field_name] = sorted(set(hits))
        return found

    def as_knowledge_context(self, text: str) -> Dict[str, object]:
        raw = self.extract(text)
        return {
            "crop": raw.get("crops", []),
            "variety": [],
            "growth_stage": [],
            "location": raw.get("locations", []),
            "agro_climatic_zone": raw.get("agro_climatic_zones", []),
            "season": raw.get("seasons", []),
            "soil_type": raw.get("soil_types", []),
            "soil_properties": {},
            "irrigation_source": [],
            "irrigation_method": raw.get("irrigation_methods", []),
            "climate": raw.get("agro_climatic_zones", []),
            "experimental_conditions": [],
        }
