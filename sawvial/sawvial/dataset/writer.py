"""
sawvial.dataset.writer
=========================
Writes the canonical dataset (JSONL, per master-spec section 22),
because each knowledge unit is then independently readable/processable
without loading a single giant JSON array into memory:

    dataset/
        documents.jsonl
        knowledge_units.jsonl
        qa.jsonl
        dataset_statistics.json
        taxonomy.json           (copy, for reproducibility)
        validation_report.json
        build_metadata.json
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write via a temp file + os.replace so a crash never leaves a half-written dataset file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def merge_into_dataset(output_dir: str, documents: List[Dict[str, Any]], units: List[Dict[str, Any]],
                       taxonomy, extraction_cfg, id_prefix: str = "SWM",
                       preserve_ids: bool = False) -> Dict[str, Any]:
    """Append new documents/units to an existing dataset: de-duplicates against what is already
    there (DOI / title / content hash / evidence hash), assigns continuing IDs, validates, then
    rewrites the dataset files atomically and recomputes statistics from the real data."""
    from sawvial.dedup.deduplicator import document_dedup_key, document_hash
    from sawvial.statistics.stats import compute_statistics
    from sawvial.validation.validator import validate_knowledge_unit

    out = Path(output_dir)
    old_docs, old_units = _read_jsonl(out / "documents.jsonl"), _read_jsonl(out / "knowledge_units.jsonl")
    seen_docs = set()
    for d in old_docs:
        seen_docs.add(document_dedup_key(d, "")[0])
        if d.get("source_hash"):
            seen_docs.add(d["source_hash"])
    seen_evidence = {u["provenance"]["evidence_hash"] for u in old_units}

    added_docs, skipped_docs = [], []
    for d in documents:
        key = document_dedup_key(d["metadata"], d.get("full_text", ""))[0]
        content = d["metadata"].get("source_hash")
        if key in seen_docs or (content and content in seen_docs):
            skipped_docs.append(d["document_id"])
            continue
        seen_docs.update({key, content} - {None})
        added_docs.append({"document_id": d["document_id"], **d["metadata"],
                           "n_candidate_chunks": d["n_candidate_chunks"]})
    keep_ids = {d["document_id"] for d in added_docs}

    start = 1 + max((int(u["id"].split("_")[1]) for u in old_units if u["id"].startswith(id_prefix + "_")),
                    default=0)
    existing_ids = {u["id"] for u in old_units}
    added_units = []
    for u in units:
        if u["source"]["document_id"] not in keep_ids or u["provenance"]["evidence_hash"] in seen_evidence:
            continue
        if preserve_ids and u["id"] in existing_ids:
            continue
        seen_evidence.add(u["provenance"]["evidence_hash"])
        if not preserve_ids:
            u["id"] = f"{id_prefix}_{start + len(added_units):06d}"
        result = validate_knowledge_unit(u, taxonomy, extraction_cfg)
        forced = u["quality"].pop("forced_status", None)
        u["quality"]["validation_status"] = forced or result.status
        added_units.append(u)

    all_docs, all_units = old_docs + added_docs, old_units + added_units
    statuses = [u["quality"]["validation_status"] for u in all_units]
    stats = compute_statistics(all_docs, all_units, statuses)
    _write_jsonl(out / "documents.jsonl", all_docs)
    _write_jsonl(out / "knowledge_units.jsonl", all_units)
    _write_jsonl(out / "qa.jsonl", [
        {"knowledge_unit_id": u["id"], "question": q["question"], "answer": q["answer"],
         "question_type": q["question_type"], "primary_domain": u["classification"]["primary_domain"],
         "origin": q.get("origin"), "verified": q.get("verified", False)}
        for u in all_units for q in u.get("qa", [])])
    tmp = out / "dataset_statistics.json.tmp"
    tmp.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out / "dataset_statistics.json")
    return {"documents_added": len(added_docs), "documents_skipped_duplicate": len(skipped_docs),
            "units_added": len(added_units), "new_unit_ids": [u["id"] for u in added_units]}


def write_dataset(
    output_dir: str,
    documents: List[Dict[str, Any]],
    knowledge_units: List[Dict[str, Any]],
    validation_report: Dict[str, Any],
    statistics: Dict[str, Any],
    build_metadata: Dict[str, Any],
    taxonomy_path: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_jsonl(out / "documents.jsonl", documents)
    _write_jsonl(out / "knowledge_units.jsonl", knowledge_units)

    qa_records = []
    for ku in knowledge_units:
        for qa in ku.get("qa", []):
            qa_records.append({
                "knowledge_unit_id": ku["id"],
                "question": qa["question"],
                "answer": qa["answer"],
                "question_type": qa["question_type"],
                "primary_domain": ku["classification"]["primary_domain"],
                "origin": qa.get("origin"), "verified": qa.get("verified", False),
            })
    _write_jsonl(out / "qa.jsonl", qa_records)

    with open(out / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)

    with open(out / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)

    with open(out / "build_metadata.json", "w", encoding="utf-8") as f:
        json.dump(build_metadata, f, indent=2, ensure_ascii=False)

    shutil.copy(taxonomy_path, out / "taxonomy.json")
