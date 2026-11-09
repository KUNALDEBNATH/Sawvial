"""
sawvial.pipeline
==================
Orchestrates the full offline knowledge-corpus construction pipeline:

    PDFs -> ingestion -> preprocessing -> document_understanding ->
    chunking -> extraction -> classification -> knowledge building ->
    QA generation -> validation -> dedup -> canonical dataset (JSONL)
    -> embeddings -> FAISS

This module contains no PDF-parsing, regex, or ML logic itself — it
wires the single-responsibility modules together and keeps the whole
thing auditable end-to-end.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from sawvial.config import BuildConfig
from sawvial.eligibility import injection_flags
from sawvial.ingestion.pdf_loader import PDFLoadError, load_pdf
from sawvial.preprocessing.metadata_extractor import extract_metadata
from sawvial.document_understanding.structure import build_structured_document
from sawvial.chunking.semantic_chunker import build_candidate_chunks, CandidateChunk
from sawvial.extraction.entities import ScientificEntityExtractor
from sawvial.extraction.context_extractor import ContextExtractor
from sawvial.extraction.numeric import extract_structured_facts
from sawvial.extraction.importance import ImportanceScorer
from sawvial.classification.domain_classifier import DomainClassifier
from sawvial.knowledge.knowledge_builder import build_knowledge_unit, make_id, KnowledgeUnitInputs
from sawvial.qa.qa_generator import generate_qa
from sawvial.validation.validator import validate_knowledge_unit
from sawvial.dedup.deduplicator import deduplicate_documents, deduplicate_knowledge_units, document_hash
from sawvial.dataset.writer import write_dataset
from sawvial.statistics.stats import compute_statistics
from sawvial.embeddings.embedder import load_embedder, embed_texts
from sawvial.index.faiss_builder import build_index

log = logging.getLogger("sawvial.pipeline")


def collect_pdf_paths(input_dir: str) -> List[Path]:
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    paths = sorted(base.rglob("*.pdf"))
    log.info(f"Found {len(paths)} PDF file(s) in '{input_dir}'")
    return paths


def _document_id(source_path: str) -> str:
    return hashlib.md5(source_path.encode()).hexdigest()[:12]


def process_document(pdf_path: Path, cfg: BuildConfig,
                      entity_extractor: ScientificEntityExtractor,
                      context_extractor: ContextExtractor,
                      importance_scorer: ImportanceScorer,
                      classifier: DomainClassifier,
                      raw_doc=None, meta_override: Dict[str, Any] | None = None,
                      initial_status: str | None = None) -> tuple:
    """Runs the full per-document pipeline (structure -> chunk ->
    extract -> classify -> knowledge unit -> QA -> validate) and
    returns (document_record, list_of_knowledge_units).

    raw_doc / meta_override let the online-research workflow feed already-fetched documents
    through the very same pipeline; initial_status (e.g. "pending_review") overrides the
    automatic validation status of every resulting unit.
    """
    raw_doc = raw_doc or load_pdf(pdf_path, ocr=cfg.ocr, ocr_language=cfg.ocr_language)
    meta = extract_metadata(raw_doc)
    meta.update({k: v for k, v in (meta_override or {}).items() if v not in (None, "", [])})
    doc_id = meta.get("document_id") or _document_id(str(pdf_path))
    meta["document_id"] = doc_id
    meta["extraction"] = raw_doc.extraction.to_dict()
    if raw_doc.extraction.status == "no_text":      # nothing to learn from; reported, not indexed
        return {"document_id": doc_id, "metadata": meta, "full_text": "", "n_candidate_chunks": 0}, []

    structured = build_structured_document(raw_doc)
    full_text = raw_doc.full_plain_text
    meta["source_hash"] = document_hash(full_text)

    candidates: List[CandidateChunk] = build_candidate_chunks(structured, cfg.chunking)

    knowledge_units: List[Dict[str, Any]] = []
    for chunk in candidates:
        text = chunk.text
        concepts = entity_extractor.unique_concepts(text)
        classification = classifier.classify(text, section=chunk.section)
        importance = importance_scorer.score(text, chunk.section, concept_hits=len(concepts))
        context = context_extractor.as_knowledge_context(text)
        facts = [f.to_dict() for f in extract_structured_facts(text)]

        ku_inputs = KnowledgeUnitInputs(
            evidence=text,
            section=chunk.section,
            page=chunk.page,
            document_meta=meta,
            classification={
                "primary_domain": classification.primary_domain,
                "secondary_domains": classification.secondary_domains,
                "subdomains": classification.subdomains,
                "topics": classification.topics,
                "confidence": classification.confidence,
            },
            context=context,
            structured_facts=facts,
            concepts=concepts,
            importance_score=importance.total,
            chunk_type=chunk.chunk_type,
        )
        ku = build_knowledge_unit(ku_inputs, unit_id="PENDING")
        ku["quality"]["document_incomplete"] = raw_doc.extraction.incomplete
        flags = injection_flags(text)
        if flags:                                  # instruction-like text inside a document
            ku["quality"]["flags"] = ["possible_prompt_injection"]
            ku["quality"]["flag_detail"] = flags
        if initial_status:
            ku["quality"]["forced_status"] = initial_status

        if (importance.total >= cfg.extraction.review_importance_score
                and ku["classification"]["primary_domain"] and not initial_status):
            ku["qa"] = generate_qa(ku)

        knowledge_units.append(ku)

    document_record = {
        "document_id": doc_id,
        "metadata": meta,
        "full_text": full_text,
        "n_candidate_chunks": len(candidates),
    }
    return document_record, knowledge_units


def run_build(cfg: BuildConfig) -> Dict[str, Any]:
    entity_extractor = ScientificEntityExtractor(cfg.taxonomy)
    context_extractor = ContextExtractor(cfg.taxonomy)
    importance_scorer = ImportanceScorer(cfg.taxonomy)
    classifier = DomainClassifier(cfg.taxonomy, entity_extractor)

    pdf_paths = collect_pdf_paths(cfg.input_dir)
    if not pdf_paths:
        log.error("No PDFs found. Nothing to build.")
        return {}

    document_records: List[Dict[str, Any]] = []
    all_units: List[Dict[str, Any]] = []

    ingestion: List[Dict[str, Any]] = []
    for pdf_path in tqdm(pdf_paths, desc="Processing documents"):
        try:
            doc_record, units = process_document(
                pdf_path, cfg, entity_extractor, context_extractor, importance_scorer, classifier,
            )
            document_records.append(doc_record)
            all_units.extend(units)
            ex = doc_record["metadata"]["extraction"]
            ingestion.append({"file": str(pdf_path), "status": ex["status"], "units": len(units),
                              "warnings": ex["warnings"]})
        except (PDFLoadError, Exception) as e:       # recorded in ingestion_report.json
            log.warning(f"Skipped {pdf_path.name}: {e}")
            ingestion.append({"file": str(pdf_path), "status": "failed", "units": 0, "warnings": [str(e)]})

    document_records = deduplicate_documents(document_records)
    kept_doc_ids = {d["document_id"] for d in document_records}
    all_units = [u for u in all_units if u["source"]["document_id"] in kept_doc_ids]

    all_units = deduplicate_knowledge_units(all_units)

    # Assign final sequential deterministic IDs after dedup so IDs are
    # stable for a given (deduplicated) build.
    for i, ku in enumerate(all_units, start=1):
        ku["id"] = make_id(i)

    # Validate every unit; never silently drop — status is recorded.
    validation_log = []
    for ku in all_units:
        result = validate_knowledge_unit(ku, cfg.taxonomy, cfg.extraction)
        forced = ku["quality"].pop("forced_status", None)
        ku["quality"]["validation_status"] = forced or result.status
        validation_log.append({"id": ku["id"], "status": ku["quality"]["validation_status"],
                               "reasons": result.reasons})

    statuses = [v["status"] for v in validation_log]
    statistics = compute_statistics(document_records, all_units, statuses)
    validation_report = {
        "total_checked": len(validation_log),
        "status_counts": statistics["validation_status_distribution"],
        "details": validation_log,
    }

    # Documents don't need full_text persisted in the canonical dataset —
    # keep documents.jsonl lightweight (metadata + stats only).
    documents_out = [
        {"document_id": d["document_id"], **d["metadata"], "n_candidate_chunks": d["n_candidate_chunks"]}
        for d in document_records
    ]

    build_metadata = cfg.build_metadata()
    write_dataset(
        output_dir=cfg.output_dir,
        documents=documents_out,
        knowledge_units=all_units,
        validation_report=validation_report,
        statistics=statistics,
        build_metadata=build_metadata,
        taxonomy_path=str(cfg.taxonomy.path),
    )

    import json as _json
    problems = [i for i in ingestion if i["status"] != "ok"]
    with open(Path(cfg.output_dir) / "ingestion_report.json", "w", encoding="utf-8") as f:
        _json.dump({"n_files": len(ingestion), "n_problem_files": len(problems), "files": ingestion},
                   f, indent=2)
    for i in problems:
        log.warning("INGESTION %s: %s -> %s", i["status"].upper(), i["file"], "; ".join(i["warnings"]) or "-")
    log.info(f"Canonical dataset written to {cfg.output_dir}")
    log.info(f"Documents: {len(documents_out)} | Knowledge units: {len(all_units)} | "
             f"QA pairs: {statistics['total_qa_pairs']}")

    return {
        "documents": documents_out,
        "knowledge_units": all_units,
        "statistics": statistics,
        "validation_report": validation_report,
    }


def run_embed_and_index(cfg: BuildConfig, knowledge_units: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """(Re)builds the retrieval index from the canonical dataset. Always a DERIVED step.

    Only answerable records are indexed (see sawvial/eligibility.py). The new index replaces
    the old one atomically; on any failure the previous working index stays in place.
    """
    import json

    dataset_file = Path(cfg.output_dir) / "knowledge_units.jsonl"
    if knowledge_units is None:
        with open(dataset_file, encoding="utf-8") as f:
            knowledge_units = [json.loads(line) for line in f if line.strip()]
    dense = None
    if not cfg.embedding.offline:
        embedder = load_embedder(cfg.embedding.model_name, False)
        dense = embedder if hasattr(embedder, "model_name") else None   # TF-IDF fallback -> lexical only
        if dense is None:
            log.warning("Dense model unavailable; building a lexical-only index")
    return build_index(knowledge_units, cfg.index_dir, dataset_file=dataset_file, dense_embedder=dense)


def run_add_pdfs(cfg: BuildConfig, pdf_paths: List[Path]) -> Dict[str, Any]:
    """Incrementally add PDFs to an existing dataset (no full rebuild, existing IDs unchanged)."""
    from sawvial.dataset.writer import merge_into_dataset

    entity_extractor = ScientificEntityExtractor(cfg.taxonomy)
    context_extractor = ContextExtractor(cfg.taxonomy)
    importance_scorer = ImportanceScorer(cfg.taxonomy)
    classifier = DomainClassifier(cfg.taxonomy, entity_extractor)
    docs, units, report = [], [], []
    for path in pdf_paths:
        try:
            doc, new_units = process_document(Path(path), cfg, entity_extractor, context_extractor,
                                              importance_scorer, classifier)
        except Exception as exc:
            report.append({"file": str(path), "status": "failed", "reason": str(exc)})
            log.warning("Could not ingest %s: %s", path, exc)
            continue
        ex = doc["metadata"]["extraction"]
        if ex["status"] == "no_text":
            report.append({"file": str(path), "status": "no_text", "reason": "; ".join(ex["warnings"])})
            continue
        docs.append(doc)
        units.extend(new_units)
        report.append({"file": str(path), "status": ex["status"], "units": len(new_units),
                       "warnings": ex["warnings"]})
    result = merge_into_dataset(cfg.output_dir, docs, units, cfg.taxonomy, cfg.extraction)
    result["files"] = report
    return result
