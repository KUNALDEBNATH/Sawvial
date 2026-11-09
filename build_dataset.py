#!/usr/bin/env python3
"""
SAWVIAL Scientific Knowledge Dataset Builder — CLI
====================================================
Usage
-----
    # Build the canonical dataset (JSONL) from a folder of PDFs
    python build_dataset.py --input_dir ./papers --output_dir ./dataset

    # Build the canonical dataset AND the FAISS index in one go
    python build_dataset.py --input_dir ./papers --output_dir ./dataset --build-index

    # (Re)build the retrieval index from an already-built dataset (lexical, fully local)
    python build_dataset.py --skip-build --build-index

    # Same, plus an optional dense index for hybrid retrieval (needs sentence-transformers)
    python build_dataset.py --skip-build --build-index --dense

    # Add one or more PDFs to the existing dataset without a full rebuild, then re-index
    python build_dataset.py --add_pdf new_paper.pdf --build-index
    python build_dataset.py --add_pdf scanned.pdf --ocr --build-index   # optional OCR (needs Tesseract)

    # Query the index
    python build_dataset.py --output_dir ./dataset --index_dir ./index \\
        --query "effect of drip irrigation on water use efficiency in cotton" --top_k 5

    # Print dataset statistics
    python build_dataset.py --output_dir ./dataset --stats

    # Re-run validation only (recomputes and rewrites validation_report.json)
    python build_dataset.py --output_dir ./dataset --validate
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sawvial.config import Taxonomy, BuildConfig, ChunkingConfig, ExtractionConfig, EmbeddingConfig
from sawvial.pipeline import run_add_pdfs, run_build, run_embed_and_index
from sawvial.index.retriever import load_index_bundle, query as retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sawvial.cli")

DEFAULT_TAXONOMY_PATH = str(Path(__file__).parent / "taxonomy.json")


def make_config(args) -> BuildConfig:
    taxonomy = Taxonomy(args.taxonomy)
    return BuildConfig(
        taxonomy=taxonomy,
        chunking=ChunkingConfig(),
        extraction=ExtractionConfig(
            min_importance_score=args.min_importance,
            review_importance_score=args.review_importance,
        ),
        embedding=EmbeddingConfig(model_name=args.model, offline=not args.dense),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        index_dir=args.index_dir,
        ocr=args.ocr, ocr_language=args.ocr_language,
    )


def cmd_stats(args):
    stats_path = Path(args.output_dir) / "dataset_statistics.json"
    if not stats_path.exists():
        log.error(f"No dataset_statistics.json found at {stats_path}. Build the dataset first.")
        return
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    print(json.dumps(stats, indent=2))


def cmd_validate(args):
    from sawvial.validation.validator import validate_knowledge_unit
    from sawvial.statistics.stats import compute_statistics

    taxonomy = Taxonomy(args.taxonomy)
    cfg_extraction = ExtractionConfig(min_importance_score=args.min_importance,
                                       review_importance_score=args.review_importance)

    units = []
    with open(Path(args.output_dir) / "knowledge_units.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                units.append(json.loads(line))

    docs = []
    with open(Path(args.output_dir) / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    # Statuses a human reviewer (or the publish transaction acting on their decision) assigned are
    # never produced by validate_knowledge_unit() itself (it only returns "validated", "needs_review",
    # "evidence_only" or "rejected" - see sawvial/validation/validator.py). A unit already carrying
    # "verified" (approved by a named reviewer via the online-research workflow and merged in by
    # sawvial.dataset.writer.merge_into_dataset) or "pending_review"/"published" must keep that
    # status: re-running automatic validation must never silently downgrade or re-grade a human
    # decision back to an automatic one.
    HUMAN_DECIDED_STATUSES = {"verified", "pending_review", "published"}
    validation_log = []
    for ku in units:
        current = ku["quality"].get("validation_status")
        if current in HUMAN_DECIDED_STATUSES or ku["quality"].get("published_as"):
            validation_log.append({"id": ku["id"], "status": current, "reasons": ["preserved human review decision"]})
            continue
        result = validate_knowledge_unit(ku, taxonomy, cfg_extraction)
        ku["quality"]["validation_status"] = result.status
        validation_log.append({"id": ku["id"], "status": result.status, "reasons": result.reasons})

    statuses = [v["status"] for v in validation_log]
    stats = compute_statistics(docs, units, statuses)
    report = {"total_checked": len(validation_log),
              "status_counts": stats["validation_status_distribution"],
              "details": validation_log}

    with open(Path(args.output_dir) / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(Path(args.output_dir) / "knowledge_units.jsonl", "w", encoding="utf-8") as f:
        for ku in units:
            f.write(json.dumps(ku, ensure_ascii=False) + "\n")
    with open(Path(args.output_dir) / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(report["status_counts"], indent=2))


def cmd_query(args):
    bundle = load_index_bundle(args.index_dir, args.output_dir)
    results = retrieve(bundle, None, args.query, top_k=args.top_k)

    print("\n" + "─" * 70)
    print(f"Top-{args.top_k} results for: \"{args.query}\"")
    print("─" * 70)
    for r in results:
        print(f"\nRank {r['rank']}  |  Score: {r['score']:.4f}  |  ID: {r['knowledge_unit_id']}")
        print(f"Domain    : {r['primary_domain']}  |  Subdomains: {', '.join(r['subdomains'])}")
        print(f"Source    : {r['source'].get('title')}  (p.{r['provenance'].get('page')})")
        print(f"Finding   : {r['finding']}")
        print(f"Evidence  : {r['evidence'][:280]}{'...' if len(r['evidence']) > 280 else ''}")
        if r["recommendation"]:
            print(f"Recommendation: {r['recommendation']}")
    print("─" * 70)


def main():
    parser = argparse.ArgumentParser(description="SAWVIAL Scientific Knowledge Dataset Builder")
    parser.add_argument("--input_dir", type=str, default="./papers")
    parser.add_argument("--output_dir", type=str, default="./dataset")
    parser.add_argument("--index_dir", type=str, default="./index")
    parser.add_argument("--taxonomy", type=str, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--offline", action="store_true", help="(default; lexical index only - kept for compatibility)")
    parser.add_argument("--dense", action="store_true", help="Also build a dense sentence-transformers index (hybrid retrieval).")
    parser.add_argument("--add_pdf", nargs="+", metavar="PDF", help="Add PDF(s) to the existing dataset incrementally.")
    parser.add_argument("--ocr", action="store_true", help="OCR scanned pages (optional; needs Tesseract installed).")
    parser.add_argument("--ocr_language", default="eng", help="Tesseract language code(s), e.g. eng or eng+hin")
    parser.add_argument("--min_importance", type=float, default=0.28)
    parser.add_argument("--review_importance", type=float, default=0.4)

    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of canonical dataset.")
    parser.add_argument("--skip-build", action="store_true", help="Skip canonical dataset build (index/query only).")
    parser.add_argument("--build-index", action="store_true", help="Build/rebuild the FAISS index after dataset build.")
    parser.add_argument("--validate", action="store_true", help="Re-run validation only on an existing dataset.")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics and exit.")
    parser.add_argument("--query", type=str, default=None, help="Run a Top-K retrieval query against the index.")
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    if args.stats:
        cmd_stats(args)
        return

    if args.validate:
        cmd_validate(args)
        return

    if args.add_pdf:
        cfg = make_config(args)
        result = run_add_pdfs(cfg, [Path(p) for p in args.add_pdf])
        print(json.dumps({k: v for k, v in result.items() if k != "new_unit_ids"}, indent=2))
        if args.build_index and result["units_added"]:
            run_embed_and_index(cfg)
        elif args.build_index:
            log.info("No new units were added; index left unchanged.")
        return

    dataset_exists = (Path(args.output_dir) / "knowledge_units.jsonl").exists()

    result = None
    if not args.skip_build and (not dataset_exists or args.rebuild):
        cfg = make_config(args)
        result = run_build(cfg)
    elif dataset_exists:
        log.info(f"Existing dataset found at {args.output_dir}. Use --rebuild to force a fresh build.")

    if args.build_index:
        cfg = make_config(args)
        run_embed_and_index(cfg, knowledge_units=result["knowledge_units"] if result else None)

    if args.query:
        cmd_query(args)


if __name__ == "__main__":
    main()
