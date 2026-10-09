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
import shutil
from pathlib import Path
from typing import List, Dict, Any


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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
            })
    _write_jsonl(out / "qa.jsonl", qa_records)

    with open(out / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)

    with open(out / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)

    with open(out / "build_metadata.json", "w", encoding="utf-8") as f:
        json.dump(build_metadata, f, indent=2, ensure_ascii=False)

    shutil.copy(taxonomy_path, out / "taxonomy.json")
