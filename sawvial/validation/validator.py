"""
sawvial.validation.validator
===============================
Quality-control layer (master-spec section 17). Every knowledge unit
is checked and assigned a `validation_status`:

    validated      — passes all checks
    needs_review   — passes core checks but is borderline (low
                     importance score, low classification confidence,
                     or missing optional context)
    rejected       — fails a hard requirement (invalid domain, no
                     evidence, no provenance, evidence too short)
    evidence_only  — evidence is solid but no domain could be
                     confidently assigned; still kept (with evidence)
                     for manual triage rather than silently dropped

Nothing is silently discarded: rejected/needs_review records are
still written to the dataset with their status so the failure mode is
auditable, exactly as sec. 17 requires ("do not silently include bad
records" implies transparent status, not deletion).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

from sawvial.config import Taxonomy, ExtractionConfig


@dataclass
class ValidationResult:
    status: str
    reasons: List[str]


def validate_knowledge_unit(ku: Dict[str, Any], taxonomy: Taxonomy, cfg: ExtractionConfig) -> ValidationResult:
    reasons: List[str] = []

    evidence = ku["knowledge"]["evidence"]
    if not evidence or len(evidence.strip()) < 20:
        reasons.append("evidence_missing_or_too_short")
        return ValidationResult(status="rejected", reasons=reasons)

    if not ku["provenance"].get("document_id") or ku["provenance"].get("page") is None:
        reasons.append("provenance_incomplete")
        return ValidationResult(status="rejected", reasons=reasons)

    primary_domain = ku["classification"].get("primary_domain")
    if primary_domain is None:
        reasons.append("no_domain_assigned")
        return ValidationResult(status="evidence_only", reasons=reasons)

    if not taxonomy.validate_domain(primary_domain):
        reasons.append(f"invalid_primary_domain:{primary_domain}")
        return ValidationResult(status="rejected", reasons=reasons)

    for sd in ku["classification"].get("secondary_domains", []):
        if not taxonomy.validate_domain(sd):
            reasons.append(f"invalid_secondary_domain:{sd}")
            return ValidationResult(status="rejected", reasons=reasons)

    importance = ku["quality"].get("scientific_importance_score", 0.0)
    confidence = ku["quality"].get("metadata_confidence", 0.0)

    if importance < cfg.min_importance_score:
        reasons.append(f"low_importance_score:{importance}")
        return ValidationResult(status="needs_review", reasons=reasons)

    if importance < cfg.review_importance_score or confidence < 0.4:
        reasons.append("borderline_confidence_or_importance")
        return ValidationResult(status="needs_review", reasons=reasons)

    for qa in ku.get("qa", []):
        if qa["answer"].strip() not in evidence and qa["answer"].strip() not in (ku["knowledge"].get("recommendation") or "") and \
           qa["question_type"] not in ("quantitative", "condition_specific"):
            # Non-verbatim QA types (e.g. summarized answers) are allowed,
            # but flag if wildly disconnected in length (crude grounding check).
            if len(qa["answer"]) > len(evidence) * 1.5:
                reasons.append("qa_possibly_ungrounded")
                return ValidationResult(status="needs_review", reasons=reasons)

    return ValidationResult(status="validated", reasons=reasons or ["ok"])
