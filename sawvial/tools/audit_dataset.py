#!/usr/bin/env python3
"""Audit (and optionally repair) dataset quality WITHOUT rebuilding it from the PDFs.

    python tools/audit_dataset.py          # report only  -> dataset/qa_audit.json
    python tools/audit_dataset.py --fix    # also rewrite the dataset files (see below)

Reads : dataset/knowledge_units.jsonl, dataset/documents.jsonl
Report: dataset/qa_audit.json
--fix writes (atomically): dataset/knowledge_units.jsonl (hard-failing QA pairs removed, every
        remaining pair labelled origin=template_from_evidence / verified=false), dataset/qa.jsonl,
        dataset/qa_rejected.jsonl (what was removed and why), dataset/dataset_statistics.json
        (recomputed from the actual files). knowledge text, statuses and IDs are never changed.
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sawvial import settings
from sawvial.eligibility import ANSWERABLE_STATUSES, is_answerable
from sawvial.statistics.stats import compute_statistics
from sawvial.validation.qa_checks import conflicting_questions, is_hard, qa_issues, sft_usable


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=str(settings.DATASET_DIR))
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    d = Path(a.dataset_dir)
    units, docs = read(d / "knowledge_units.jsonl"), read(d / "documents.jsonl")

    issue_counts, rejected, rows = Counter(), [], []
    for ku in units:
        for qa in ku.get("qa", []):
            iss = qa_issues(ku, qa)
            issue_counts.update(iss)
            rows.append((ku, qa, iss))
    conflicts = conflicting_questions(qa for _, qa, _ in rows)
    n_conflict_pairs = sum(1 for _, qa, _ in rows if qa["question"].strip().lower() in conflicts)
    hard = [r for r in rows if is_hard(r[2])]
    sft_ok = [r for r in rows if is_answerable(r[0]) and sft_usable(r[0], r[1])[0]]
    status = Counter(u["quality"]["validation_status"] for u in units)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qa_pairs_total": len(rows), "issue_counts": dict(issue_counts.most_common()),
        "hard_failures_removed_by_fix": len(hard),
        "generic_questions_reused_with_different_answers": len(conflicts),
        "qa_pairs_using_a_reused_generic_question": n_conflict_pairs,
        "reused_question_note": "Not removed: SFT prompts include the retrieved context, so the same "
                                "generic question with a different passage is not contradictory.",
        "qa_pairs_usable_for_sft": len(sft_ok),
        "units_by_status": dict(status),
        "answerable_units": sum(is_answerable(u) for u in units),
        "answerable_statuses": sorted(ANSWERABLE_STATUSES),
        "note": "QA answers are extractive templates (evidence text), not independently verified facts.",
    }
    (d / "qa_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not a.fix:
        print("\n(report only; add --fix to rewrite the dataset files)")
        return
    out_qa = []
    for ku in units:
        kept = []
        for qa in ku.get("qa", []):
            iss = qa_issues(ku, qa)
            if is_hard(iss):
                rejected.append({"knowledge_unit_id": ku["id"], "question": qa["question"],
                                 "answer": qa["answer"][:300], "issues": iss})
                continue
            qa["origin"], qa["verified"] = "template_from_evidence", False
            kept.append(qa)
            out_qa.append({"knowledge_unit_id": ku["id"], "question": qa["question"], "answer": qa["answer"],
                           "question_type": qa["question_type"], "origin": qa["origin"], "verified": False,
                           "primary_domain": ku["classification"]["primary_domain"]})
        ku["qa"] = kept
    write(d / "knowledge_units.jsonl", units)
    write(d / "qa.jsonl", out_qa)
    write(d / "qa_rejected.jsonl", rejected)
    stats = compute_statistics(docs, units, [u["quality"]["validation_status"] for u in units])
    stats["answerable_units"] = report["answerable_units"]
    stats["answerable_statuses"] = report["answerable_statuses"]
    stats["qa_pairs_unverified"] = len(out_qa)
    stats["statistics_recomputed_at"] = report["generated_at"]
    (d / "dataset_statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFixed: removed {len(rejected)} QA pairs, kept {len(out_qa)}. Rebuild the index: "
          "python build_dataset.py --skip-build --build-index")


if __name__ == "__main__":
    main()
