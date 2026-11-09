#!/usr/bin/env python3
"""
tools.ingest_kcc_csv
=====================
Ingests a Kisan Call Centre (KCC) style CSV export (columns such as
QueryText, KccAns, Crop, Category, StateName, ...) into SAWVIAL's
canonical dataset/index, using the SAME extraction, classification,
context, QA-generation and validation code the PDF pipeline uses --
nothing about this is a special-cased shortcut, so a KCC-derived
record is governed by exactly the same eligibility/grounding rules
(sawvial/eligibility.py) as a PDF-derived one, and can never bypass
the "is_answerable" gate the API/grounding checks enforce at query
time.

Each usable CSV row becomes its own tiny "document" (one government
call-record), and its Query+Answer text becomes one knowledge-unit
"chunk" -- the same unit of evidence a PDF paragraph would produce.

Usage
-----
    # Dry run: see how many rows would be kept, no writes
    python tools/ingest_kcc_csv.py --csv kcc_export.csv --dry-run

    # Ingest into ./dataset, then rebuild ./index (lexical, no GPU needed)
    python tools/ingest_kcc_csv.py --csv kcc_export.csv --build-index

Design notes
------------
* Only rows judged "usable evidence" are kept: non-empty QueryText AND
  KccAns, an answer long enough to carry real information
  (sawvial.eligibility.is_usable_evidence applies the same floor used
  everywhere else: >=80 chars / >=12 words), and -- by default --
  text written mostly in Latin script, because the taxonomy's
  vocabulary (sawvial/taxonomy.json) is English, so classification on
  Hindi/other-script text is unreliable and defaults to
  `needs_review`/`evidence_only` (never answerable) rather than being
  silently mis-tagged. Non-Latin rows are not discarded -- see
  --keep-untranslated below.
* Evidence text is the KCC answer ONLY -- never the farmer's raw
  question -- so a verbatim/extractive fallback quote can never just
  echo the question back as if it were the answer. The question is
  still preserved as a verbatim Q/A pair (ku["qa"]) and in the
  record's title, just not inside the quotable evidence string.
* Answers that are a verbless, comma-joined list of short fragments
  (typo'd/garbled CSV rows, e.g. "organci soils ,fertile soils ,ondru
  nelalu etc.,") are filtered out at ingestion -- see
  `_looks_like_garbled_fragment_list` -- rather than indexed and later
  surfacing as a low-quality answer or an unsupportable LLM claim.
* document_id is a stable hash of the KCC call ID so re-running the
  script on the same export (or an overlapping export) does not create
  duplicate knowledge units -- sawvial.dataset.writer.merge_into_dataset
  already de-duplicates by document + evidence hash, this just makes
  the document hash deterministic per call record instead of random.
* Nothing here invents facts: evidence is the verbatim KccAns text,
  'finding' is built the same auto-restatement knowledge_builder.py
  already uses for PDF chunks, and validation_status is decided by the
  same validator as everything else (not force-set to "validated").
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sawvial.config import Taxonomy, BuildConfig, ExtractionConfig, EmbeddingConfig
from sawvial.eligibility import is_usable_evidence
from sawvial.extraction.entities import ScientificEntityExtractor
from sawvial.extraction.context_extractor import ContextExtractor
from sawvial.extraction.numeric import extract_structured_facts
from sawvial.extraction.importance import ImportanceScorer
from sawvial.classification.domain_classifier import DomainClassifier
from sawvial.knowledge.knowledge_builder import build_knowledge_unit, KnowledgeUnitInputs
from sawvial.qa.qa_generator import generate_qa
from sawvial.dataset.writer import merge_into_dataset
from sawvial.pipeline import run_embed_and_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sawvial.ingest_kcc")

DEFAULT_TAXONOMY_PATH = str(Path(__file__).parent.parent / "taxonomy.json")

REQUIRED_COLS = {"QueryText", "KccAns"}

# A well-formed advisory answer uses ordinary sentence grammar somewhere (a verb/linking
# word), not just a comma-joined list of fragments. This is deliberately cheap/heuristic --
# it does not understand English -- but it reliably separates garbled/typo'd/mixed-language
# CSV rows (e.g. "organci soils ,fertile soils ,ondru nelalu etc.,") from real sentences
# (e.g. "...they grow best in well-drained, light loam soil with a pH of 5 to 7.").
_VERB_CUES = re.compile(
    r"\b(is|are|was|were|be|been|should|recommended|advised|needs?|requires?|requires?|"
    r"grows?|growing|grown|apply|applied|applying|use|used|using|add|adding|added|mix|mixed|"
    r"spray|sprayed|spraying|avoid|prefer|preferred|prefers|maintain|ensure|keep|treat|"
    r"treated|control|controlled|contains?|suitable|suited|provide|provided|gives?|give|"
    r"has|have|had|can|will|do|does|needed)\b", re.IGNORECASE)

# Rows that are mostly short comma/space-joined fragments with no verb at all read as a
# keyword dump, not an answer -- e.g. "organci soils ,fertile soils ,ondru nelalu etc.,"
_FRAGMENT_SPLIT = re.compile(r"[,.;]")


def _looks_like_garbled_fragment_list(answer: str) -> bool:
    if _VERB_CUES.search(answer):
        return False
    fragments = [f.strip() for f in _FRAGMENT_SPLIT.split(answer) if f.strip()]
    if len(fragments) < 2:
        return False
    avg_len = sum(len(f.split()) for f in fragments) / len(fragments)
    return avg_len <= 4  # short, verbless, comma-joined -> treat as unusable evidence


def _latin_fraction(text: str) -> float:
    """Fraction of alphabetic characters in `text` that are Latin script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters)


def _clean(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def _row_document_id(kcc_call_id: Any, query: str, answer: str) -> str:
    key = f"kcc:{kcc_call_id}" if _clean(kcc_call_id) else f"kcc:{query}|{answer}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def load_rows(csv_path: str, min_latin_fraction: float, keep_untranslated: bool) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    seen_pairs = set()
    for _, r in df.iterrows():
        query = _clean(r.get("QueryText"))
        answer = _clean(r.get("KccAns"))
        if not query or not answer:
            continue

        # Evidence is the ANSWER ONLY -- never the farmer's question text -- so a quoted/
        # extractive-fallback excerpt (sawvial.grounding.extractive_answer) can never just
        # echo the question back as if it were the answer. The question still reaches the
        # index via `qa` (the verbatim Q/A pair added below) and via the record's title.
        if not is_usable_evidence(answer):
            continue
        if _looks_like_garbled_fragment_list(answer):
            continue

        combined = f"{query}. {answer}"
        latin_frac = _latin_fraction(combined)
        if latin_frac < min_latin_fraction and not keep_untranslated:
            continue

        pair_key = (query.lower(), answer.lower())
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        year = None
        try:
            y = r.get("year")
            year = int(float(y)) if y not in (None, "", "nan") else None
        except (ValueError, TypeError):
            year = None

        rows.append({
            "kcc_call_id": r.get("KCCCallID"),
            "query": query,
            "answer": answer,
            "crop": _clean(r.get("Crop")) or _clean(r.get("expected_crop")),
            "category": _clean(r.get("Category")),
            "sector": _clean(r.get("Sector")),
            "state": _clean(r.get("StateName")),
            "district": _clean(r.get("DistrictName")),
            "year": year,
            "non_latin": latin_frac < min_latin_fraction,
        })
    return rows


def build_units_from_rows(rows: List[Dict[str, Any]], cfg: BuildConfig):
    taxonomy = cfg.taxonomy
    entity_extractor = ScientificEntityExtractor(taxonomy)
    context_extractor = ContextExtractor(taxonomy)
    importance_scorer = ImportanceScorer(taxonomy)
    classifier = DomainClassifier(taxonomy, entity_extractor)

    documents, units = [], []
    skipped_low_importance = 0
    skipped_no_domain = 0

    for row in rows:
        evidence = row["answer"]  # answer only -- see note at the filtering step above
        doc_id = _row_document_id(row["kcc_call_id"], row["query"], row["answer"])

        title_bits = [b for b in [row["crop"], row["query"][:120]] if b]
        title = " — ".join(title_bits) if title_bits else row["query"][:120]

        doc_meta = {
            "document_id": doc_id,
            "title": title,
            "authors": [],
            "year": row["year"],
            "journal": None,
            "publisher": "Kisan Call Centre (KCC), Government of India",
            "doi": None,
            "url": None,
            "document_type": "farmer_advisory_record",
            "source_hash": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
            "extraction": {"status": "ok", "warnings": [], "incomplete": False},
        }

        concepts = entity_extractor.unique_concepts(evidence)
        classification = classifier.classify(evidence, section="kcc_qa")
        importance = importance_scorer.score(evidence, "kcc_qa", concept_hits=len(concepts))
        context = context_extractor.as_knowledge_context(evidence)
        if row["crop"] and not context.get("crop"):
            # KCC's own structured Crop column is a reliable signal even when
            # the free-text vocabulary match misses a local/variety name.
            context["crop"] = [row["crop"].lower()]
        facts = [f.to_dict() for f in extract_structured_facts(evidence)]

        if not classification.primary_domain:
            skipped_no_domain += 1
            # still build the unit -- the validator (same as everywhere else)
            # will route it to evidence_only; we don't silently drop it so the
            # ingestion report stays honest about what happened to every row.

        if importance.total < cfg.extraction.min_importance_score:
            skipped_low_importance += 1

        ku_inputs = KnowledgeUnitInputs(
            evidence=evidence,
            section="kcc_qa",
            # KCC call records have no page concept, but the validator's provenance
            # check (sawvial/validation/validator.py) hard-rejects page=None, so a
            # sentinel 0 is used -- never None -- to mean "not a paginated document".
            page=0,
            document_meta=doc_meta,
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
            chunk_type="kcc_qa",
        )
        ku = build_knowledge_unit(ku_inputs, unit_id="PENDING")
        ku["quality"]["document_incomplete"] = False
        ku["source"]["state"] = row["state"] or None
        ku["source"]["district"] = row["district"] or None
        if row["non_latin"]:
            ku["quality"]["flags"] = ku["quality"].get("flags", []) + ["non_latin_script_low_confidence"]

        if (importance.total >= cfg.extraction.review_importance_score
                and ku["classification"]["primary_domain"]):
            ku["qa"] = generate_qa(ku)
            # The farmer's own question is itself a grounded, verbatim Q/A pair --
            # add it alongside the template-generated ones.
            ku["qa"].append({
                "question": row["query"],
                "answer": row["answer"],
                "question_type": "kcc_verbatim",
                "origin": "kcc_source",
            })

        documents.append({
            "document_id": doc_id,
            "metadata": doc_meta,
            "full_text": evidence,
            "n_candidate_chunks": 1,
        })
        units.append(ku)

    return documents, units, {
        "rows_in": len(rows),
        "skipped_low_importance": skipped_low_importance,
        "skipped_no_domain_routed_evidence_only": skipped_no_domain,
    }


def main():
    ap = argparse.ArgumentParser(description="Ingest a KCC-style CSV into the SAWVIAL dataset/index.")
    ap.add_argument("--csv", required=True, help="Path to the KCC CSV export.")
    ap.add_argument("--output_dir", default="./dataset")
    ap.add_argument("--index_dir", default="./index")
    ap.add_argument("--taxonomy", default=DEFAULT_TAXONOMY_PATH)
    ap.add_argument("--min_importance", type=float, default=0.28)
    ap.add_argument("--review_importance", type=float, default=0.4)
    ap.add_argument("--min_latin_fraction", type=float, default=0.8,
                     help="Rows whose evidence is less than this fraction Latin-script are "
                          "skipped unless --keep-untranslated is given (taxonomy vocab is English).")
    ap.add_argument("--keep-untranslated", action="store_true",
                     help="Keep non-Latin-script rows too (they are flagged and will mostly land "
                          "in evidence_only/needs_review, i.e. not answerable, until translated).")
    ap.add_argument("--build-index", action="store_true", help="Rebuild the lexical index after ingesting.")
    ap.add_argument("--dry-run", action="store_true", help="Only report what would be ingested; write nothing.")
    args = ap.parse_args()

    taxonomy = Taxonomy(args.taxonomy)
    cfg = BuildConfig(
        taxonomy=taxonomy,
        extraction=ExtractionConfig(min_importance_score=args.min_importance,
                                     review_importance_score=args.review_importance),
        embedding=EmbeddingConfig(offline=True),
        output_dir=args.output_dir,
        index_dir=args.index_dir,
    )

    log.info("Reading %s ...", args.csv)
    rows = load_rows(args.csv, args.min_latin_fraction, args.keep_untranslated)
    log.info("%d usable row(s) after filtering.", len(rows))
    if not rows:
        log.error("Nothing usable found -- check column names / --min_latin_fraction.")
        return

    log.info("Building knowledge units (classification, context, importance, QA) ...")
    documents, units, report = build_units_from_rows(rows, cfg)
    log.info("Built %d knowledge unit(s) from %d document(s). %s", len(units), len(documents), report)

    if args.dry_run:
        from collections import Counter
        domains = Counter(u["classification"]["primary_domain"] for u in units)
        crops = Counter(c for u in units for c in u["context"]["crop"])
        log.info("[dry-run] primary_domain counts: %s", dict(domains))
        log.info("[dry-run] top crops: %s", crops.most_common(15))
        return

    result = merge_into_dataset(args.output_dir, documents, units, taxonomy, cfg.extraction)
    log.info("Merged into %s: %s", args.output_dir, {k: v for k, v in result.items() if k != "new_unit_ids"})

    if args.build_index:
        log.info("Rebuilding index at %s ...", args.index_dir)
        meta = run_embed_and_index(cfg)
        log.info("Index rebuilt: %d answerable units indexed.", meta["n_units"])
    else:
        log.info("Skipped index rebuild (pass --build-index to also refresh %s).", args.index_dir)


if __name__ == "__main__":
    main()
