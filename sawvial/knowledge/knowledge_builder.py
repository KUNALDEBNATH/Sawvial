"""
sawvial.knowledge.knowledge_builder
======================================
Builds the canonical knowledge-unit record (master-spec section 11)
from a classified, scored CandidateChunk, separating:

    evidence        — the original source-grounded passage, verbatim
    finding         — a concise semantic interpretation (built from
                       detected structured facts, NEVER invented text
                       beyond what those facts assert)
    recommendation  — only populated when the source text itself
                       contains explicit recommendation language
                       ("recommended", "should be", "is advised"),
                       and always labelled with recommendation_type
                       so a downstream consumer knows whether the
                       source said it verbatim or Claude derived it
                       from the evidence.

No numbers, entities or conclusions are fabricated: `finding` is
assembled purely by restating the StructuredFacts and matched
concepts already extracted from `evidence`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

_RECOMMENDATION_CUE_RE = re.compile(
    r"\b(is recommended|are recommended|recommended to|is advised|should be|"
    r"it is suggested|farmers should|it is advisable|ought to)\b",
    re.IGNORECASE,
)


def make_id(counter: int) -> str:
    return f"SWM_{counter:06d}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_finding(evidence: str, facts: List[Dict[str, Any]], concepts: List[str]) -> str:
    """Concise, source-grounded restatement built ONLY from what was
    actually extracted — never a free paraphrase that could drift from
    the evidence."""
    if not facts and not concepts:
        # Fall back to a trimmed version of the evidence itself rather
        # than inventing an interpretation.
        return evidence if len(evidence) <= 220 else evidence[:217].rsplit(" ", 1)[0] + "..."

    pieces = []
    for fact in facts:
        if fact.get("fact_type") == "effect":
            param = fact.get("parameter", "the measured parameter")
            pieces.append(f"{param} showed a {fact.get('effect')} of {fact.get('magnitude')}{fact.get('unit', '')}")
        elif fact.get("fact_type") == "measurement":
            pieces.append(f"a value of {fact.get('magnitude')} {fact.get('unit', '')} was reported")
        elif fact.get("fact_type") == "range":
            pieces.append(f"a range of {fact.get('low')}-{fact.get('high')} {fact.get('unit', '')} was reported")

    concept_str = ", ".join(c.replace("_", " ") for c in concepts[:4])
    finding = "; ".join(pieces)
    if concept_str:
        finding = f"{finding} (related to {concept_str})" if finding else f"Findings relate to {concept_str}."
    return finding.strip() or evidence[:200]


def _extract_recommendation(evidence: str) -> tuple[Optional[str], str]:
    """Returns (recommendation_text_or_None, recommendation_type)."""
    match = _RECOMMENDATION_CUE_RE.search(evidence)
    if not match:
        return None, "none"
    # Grab the sentence containing the cue as the explicit recommendation —
    # never invent wording beyond the source sentence itself.
    start = evidence.rfind(".", 0, match.start())
    end = evidence.find(".", match.end())
    start = 0 if start == -1 else start + 1
    end = len(evidence) if end == -1 else end + 1
    sentence = evidence[start:end].strip()
    return (sentence or None), "source_explicit"


@dataclass
class KnowledgeUnitInputs:
    evidence: str
    section: str
    page: Optional[int]
    document_meta: Dict[str, Any]
    classification: Dict[str, Any]
    context: Dict[str, Any]
    structured_facts: List[Dict[str, Any]]
    concepts: List[str]
    importance_score: float
    chunk_type: str = "text"


def build_knowledge_unit(inputs: KnowledgeUnitInputs, unit_id: str) -> Dict[str, Any]:
    finding = _build_finding(inputs.evidence, inputs.structured_facts, inputs.concepts)
    recommendation, recommendation_type = _extract_recommendation(inputs.evidence)

    keywords = sorted(set(inputs.concepts + inputs.classification.get("topics", [])))
    domain_part = inputs.classification.get("primary_domain", "")
    retrieval_text = " | ".join(filter(None, [
        domain_part,
        ", ".join(inputs.classification.get("subdomains", [])),
        ", ".join(inputs.classification.get("topics", [])),
        ", ".join(inputs.context.get("crop", [])),
        ", ".join(inputs.context.get("soil_type", [])),
        ", ".join(inputs.context.get("irrigation_method", [])),
        inputs.evidence,
        finding,
    ]))

    return {
        "id": unit_id,
        "source": {
            "document_id": inputs.document_meta.get("document_id"),
            "title": inputs.document_meta.get("title"),
            "authors": inputs.document_meta.get("authors", []),
            "year": inputs.document_meta.get("year"),
            "journal": inputs.document_meta.get("journal"),
            "publisher": inputs.document_meta.get("publisher"),
            "doi": inputs.document_meta.get("doi"),
            "url": inputs.document_meta.get("url"),
            "document_type": inputs.document_meta.get("document_type"),
            "page": inputs.page,
            "section": inputs.section,
        },
        "classification": {
            "primary_domain": inputs.classification.get("primary_domain"),
            "secondary_domains": inputs.classification.get("secondary_domains", []),
            "subdomains": inputs.classification.get("subdomains", []),
            "topics": inputs.classification.get("topics", []),
        },
        "context": inputs.context,
        "knowledge": {
            "evidence": inputs.evidence,
            "finding": finding,
            "recommendation": recommendation,
            "recommendation_type": recommendation_type,
        },
        "structured_facts": inputs.structured_facts,
        "qa": [],  # filled in by qa_generator
        "retrieval": {
            "semantic_text": retrieval_text,
            "keywords": keywords,
            "entities": inputs.concepts,
        },
        "provenance": {
            "document_id": inputs.document_meta.get("document_id"),
            "page": inputs.page,
            "section": inputs.section,
            "source_hash": inputs.document_meta.get("source_hash"),
            "evidence_hash": _sha256(inputs.evidence),
        },
        "quality": {
            "scientific_importance_score": inputs.importance_score,
            "metadata_confidence": inputs.classification.get("confidence", 0.0),
            "extraction_quality": "chunk_type:" + inputs.chunk_type,
            "validation_status": "pending",
        },
    }
