"""
sawvial.classification.domain_classifier
==========================================
Classifies a single candidate knowledge unit (NOT a whole document)
into exactly one `primary_domain` plus optional `secondary_domains`,
each drawn strictly from taxonomy.allowed_domains.

Design goals mandated by the spec:

  * Classification happens per knowledge-unit, never per-document.
  * It must be more than "keyword appears N times" — a paragraph that
    mentions "crop" 100 times must NOT become crop_management (which
    does not exist as a domain here). Crop/climate/technology terms
    are explicitly down-weighted / excluded from domain scoring and
    routed to context instead (see context_extractor.py).
  * Scoring is based on which taxonomy subdomain/topic surface forms
    are actually matched in the text (via ScientificEntityExtractor),
    not a flat keyword list, so scores are traceable to concrete
    topic hits.
  * A hard assertion guarantees no invalid domain is ever returned —
    if scoring produces nothing usable, the unit is routed to
    "needs_review" by the validator rather than silently mis-tagged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

from sawvial.config import Taxonomy
from sawvial.extraction.entities import ScientificEntityExtractor, ConceptMatch


@dataclass
class ClassificationResult:
    primary_domain: Optional[str]
    secondary_domains: List[str]
    subdomains: List[str]
    topics: List[str]
    domain_scores: Dict[str, float]
    confidence: float


class DomainClassifier:
    def __init__(self, taxonomy: Taxonomy, entity_extractor: ScientificEntityExtractor):
        self.taxonomy = taxonomy
        self.entity_extractor = entity_extractor

    def classify(self, text: str, section: str = "unknown") -> ClassificationResult:
        matches: List[ConceptMatch] = self.entity_extractor.extract(text)
        domain_scores: Dict[str, float] = {d: 0.0 for d in self.taxonomy.allowed_domains}
        subdomain_hits: Dict[str, int] = {}
        topic_hits: set = set()

        section_weight = self.taxonomy.importance_signals.get("section_weights", {}).get(section, 1.0)

        for match in matches:
            if not match.domain:
                continue  # concept without a taxonomy-mapped domain (rare) -> ignored for scoring
            weight = 1.0 * section_weight
            # Multi-word, more specific matches count more than a single
            # generic word so specificity (not raw frequency) drives the score.
            weight *= 1.0 + 0.15 * (len(match.surface_form.split()) - 1)
            domain_scores[match.domain] += weight
            if match.subdomain:
                subdomain_hits[match.subdomain] = subdomain_hits.get(match.subdomain, 0) + 1
            topic_hits.add(match.canonical)

        total = sum(domain_scores.values())
        if total <= 0:
            return ClassificationResult(
                primary_domain=None, secondary_domains=[], subdomains=[],
                topics=[], domain_scores=domain_scores, confidence=0.0,
            )

        ranked = sorted(domain_scores.items(), key=lambda kv: kv[1], reverse=True)
        primary_domain, primary_score = ranked[0]
        confidence = primary_score / total

        secondary_domains = [
            d for d, s in ranked[1:]
            if s > 0 and (s / total) >= 0.2  # meaningful secondary presence only
        ]

        subdomains = sorted(subdomain_hits.keys(), key=lambda k: subdomain_hits[k], reverse=True)

        # Absolute final guard — never allow an invalid domain to leave this
        # function, matching master-spec section 32 (Final Domain Validation).
        assert self.taxonomy.validate_domain(primary_domain), (
            f"Classifier produced invalid primary_domain '{primary_domain}'"
        )
        for d in secondary_domains:
            assert self.taxonomy.validate_domain(d), f"Invalid secondary domain '{d}'"

        return ClassificationResult(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            subdomains=subdomains,
            topics=sorted(topic_hits),
            domain_scores=domain_scores,
            confidence=round(confidence, 3),
        )
