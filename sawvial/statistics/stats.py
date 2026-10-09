"""
sawvial.statistics.stats
===========================
Generates dataset statistics per master-spec section 24, plus a
simple class-imbalance flag.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Dict, Any


def _distribution(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: kv[1], reverse=True))


def compute_statistics(documents: List[Dict[str, Any]], knowledge_units: List[Dict[str, Any]],
                        validation_statuses: List[str]) -> Dict[str, Any]:
    domain_counter = Counter()
    subdomain_counter = Counter()
    topic_counter = Counter()
    crop_counter = Counter()
    soil_type_counter = Counter()
    irrigation_counter = Counter()
    season_counter = Counter()
    source_type_counter = Counter()
    evidence_lengths = []

    for ku in knowledge_units:
        domain_counter[ku["classification"].get("primary_domain") or "unassigned"] += 1
        for sd in ku["classification"].get("subdomains", []):
            subdomain_counter[sd] += 1
        for t in ku["classification"].get("topics", []):
            topic_counter[t] += 1
        for c in ku["context"].get("crop", []):
            crop_counter[c] += 1
        for s in ku["context"].get("soil_type", []):
            soil_type_counter[s] += 1
        for im in ku["context"].get("irrigation_method", []):
            irrigation_counter[im] += 1
        for se in ku["context"].get("season", []):
            season_counter[se] += 1
        source_type_counter[ku["source"].get("document_type") or "unknown"] += 1
        evidence_lengths.append(len(ku["knowledge"]["evidence"]))

    validation_counter = Counter(validation_statuses)
    total_units = len(knowledge_units)
    total_qa = sum(len(ku.get("qa", [])) for ku in knowledge_units)

    avg_evidence_len = round(sum(evidence_lengths) / len(evidence_lengths), 1) if evidence_lengths else 0.0

    # Simple imbalance flag: any domain with < 10% of the units when there
    # are 3+ domains represented at all.
    imbalance_flags = []
    if total_units > 0:
        for domain, count in domain_counter.items():
            share = count / total_units
            if share < 0.10:
                imbalance_flags.append({"domain": domain, "share": round(share, 3)})

    return {
        "total_documents": len(documents),
        "total_knowledge_units": total_units,
        "total_qa_pairs": total_qa,
        "units_per_domain": _distribution(domain_counter),
        "units_per_subdomain": _distribution(subdomain_counter),
        "units_per_topic": _distribution(topic_counter),
        "units_per_source_type": _distribution(source_type_counter),
        "units_per_crop": _distribution(crop_counter),
        "units_per_soil_type": _distribution(soil_type_counter),
        "units_per_irrigation_method": _distribution(irrigation_counter),
        "units_per_season": _distribution(season_counter),
        "average_evidence_length_chars": avg_evidence_len,
        "validation_status_distribution": _distribution(validation_counter),
        "class_imbalance_flags": imbalance_flags,
    }
