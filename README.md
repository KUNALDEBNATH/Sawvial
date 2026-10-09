# SAWVIAL Scientific Knowledge Dataset Builder v2.0

A redesigned pipeline that turns agricultural research PDFs (soil
management, water management, soil & water conservation) into a
structured, traceable, RAG-ready knowledge corpus — not a bag of
fixed-size text chunks.

## 1. What changed vs. the original `sawvial_corpus_builder.py`

The original script (kept unmodified in your upload for reference) was
a reasonable prototype but had architectural weaknesses that this
rewrite specifically replaces:

| Weakness in original | Fix in v2.0 |
|---|---|
| Fixed 400-word sliding-window chunks, cut mid-reasoning | Section → paragraph → sentence-aware **semantic chunker** (`chunking/semantic_chunker.py`) that only splits at paragraph/sentence boundaries and never mid-finding |
| Whole-document topic tagging (`topics: [...]` on every chunk) | **Per-knowledge-unit classification** (`classification/domain_classifier.py`) — every chunk is classified independently |
| 5 arbitrary topic buckets incl. `crop_agronomy`, `climate_weather` as top-level "domains" | **Exactly 3 top-level domains** enforced by a hard assertion; crop/climate/technology become `context`, never a domain |
| `if term in text` keyword lists, no normalization | Alias-aware concept extraction (`extraction/entities.py`) — "SOC", "soil C", "organic carbon" all resolve to one canonical concept |
| `recommendation: ""` — always empty | Evidence/finding/recommendation separated (`knowledge/knowledge_builder.py`); recommendation is only populated when the **source text itself** contains recommendation language, and is labelled `recommendation_type: source_explicit` |
| No numeric-value preservation | `extraction/numeric.py` extracts effects/ranges/measurements/p-values/CIs as structured facts, so "decreased by 24%" is never flattened to "decreased" |
| No Q&A | `qa/qa_generator.py` builds grounded Q&A directly from evidence + structured facts (never freely generated) |
| Dedup on first-2000-char hash only | `dedup/deduplicator.py` uses DOI / title+author / full-content hash for documents, and shingle-based near-duplicate detection for knowledge units |
| Tables silently dropped (`page.get_text("text")` flattens everything) | `ingestion/pdf_loader.py` extracts tables via PyMuPDF's table finder and keeps them as their own provenance-tagged knowledge units |
| No validation layer | `validation/validator.py` assigns `validated / needs_review / rejected / evidence_only` to every unit — nothing is silently dropped |
| FAISS built directly from chunks, no canonical dataset | Canonical **JSONL first**, FAISS is always a derived, rebuildable artifact (`dataset/writer.py` → `embeddings/` → `index/faiss_builder.py`) |
| Embedding model hard-coded to sentence-transformers with a try/except fallback baked into one function | Fully modular, configurable embedding backend (`embeddings/embedder.py`), TF-IDF fallback works fully offline, no commercial API ever required |

## 2. Domain architecture (non-negotiable)

Every knowledge unit's `primary_domain` (and any `secondary_domains`)
MUST be one of exactly three values, enforced by a hard `assert` in
`classification/domain_classifier.py` and re-checked in
`validation/validator.py`:

```
soil_management
water_management
soil_water_conservation_management
```

Crop, season, location, climate, irrigation method, soil type, and
similar attributes are always attached under `context`, never used as
a domain. The full subdomain/topic hierarchy under each domain lives
in `taxonomy.json` (data-driven — edit that file to extend the
taxonomy without touching Python).

Classification happens **per knowledge unit**, not per document — a
single PDF routinely produces knowledge units in all three domains
(see the example output in section 6).

## 3. Pipeline stages

```
PDF
 → ingestion/pdf_loader.py            (page/block/table-aware extraction)
 → preprocessing/metadata_extractor.py (title/authors/year/DOI)
 → preprocessing/cleaner.py            (scientific-notation-safe cleaning)
 → document_understanding/structure.py (sections → paragraphs, heading detection)
 → chunking/semantic_chunker.py        (adaptive, sentence-boundary chunking)
 → extraction/entities.py              (alias-normalized scientific concepts)
 → extraction/numeric.py               (numbers/units/effects/p-values)
 → extraction/context_extractor.py     (crop/season/location/soil-type context)
 → extraction/importance.py            (multi-signal importance scoring)
 → classification/domain_classifier.py (3-domain semantic classification)
 → knowledge/knowledge_builder.py      (evidence/finding/recommendation + provenance)
 → qa/qa_generator.py                  (grounded Q&A)
 → dedup/deduplicator.py               (document + near-duplicate knowledge-unit dedup)
 → validation/validator.py             (validated/needs_review/rejected/evidence_only)
 → dataset/writer.py                   (canonical JSONL — source of truth)
 → embeddings/embedder.py              (configurable embedding backend, derived step)
 → index/faiss_builder.py              (FAISS index, id-mapped, derived step)
 → index/retriever.py                  (Top-K retrieval with full traceable payload)
```

## 4. Knowledge-unit schema

```jsonc
{
  "id": "SWM_000004",
  "source": {
    "document_id": "fb33553ebe40", "title": "...", "authors": [], "year": 2022,
    "journal": null, "publisher": null, "doi": null, "url": null,
    "document_type": "research_paper", "page": 1, "section": "results"
  },
  "classification": {
    "primary_domain": "soil_management",
    "secondary_domains": ["water_management"],
    "subdomains": ["soil_chemistry", "soil_amendments"],
    "topics": ["organic_carbon", "ph", "gypsum"]
  },
  "context": {
    "crop": ["cotton"], "variety": [], "growth_stage": [], "location": [],
    "agro_climatic_zone": ["semi-arid"], "season": [], "soil_type": [],
    "soil_properties": {}, "irrigation_source": [], "irrigation_method": ["drip"],
    "climate": ["semi-arid"], "experimental_conditions": []
  },
  "knowledge": {
    "evidence": "Application of organic manure increased soil organic carbon by 0.3 g/kg ...",
    "finding": "soil organic carbon showed a increase of 0.3g/kg (related to organic_carbon, manure)",
    "recommendation": null,
    "recommendation_type": "none"
  },
  "structured_facts": [
    {"fact_type": "measurement", "raw_text": "0.3 g/kg", "magnitude": 0.3, "unit": "g/kg"}
  ],
  "qa": [
    {"question": "What effect was observed on soil organic carbon related to organic carbon?",
     "answer": "Soil organic carbon showed a increase of 0.3g/kg, based on: ...",
     "question_type": "quantitative"}
  ],
  "retrieval": {
    "semantic_text": "soil_management | soil_chemistry, soil_amendments | ... | <evidence> | <finding>",
    "keywords": ["organic_carbon", "manure", "ph"],
    "entities": ["organic_carbon", "manure"]
  },
  "provenance": {
    "document_id": "fb33553ebe40", "page": 1, "section": "results",
    "source_hash": "0f94...", "evidence_hash": "87b3..."
  },
  "quality": {
    "scientific_importance_score": 0.71, "metadata_confidence": 0.62,
    "extraction_quality": "chunk_type:text", "validation_status": "validated"
  }
}
```

See `example_knowledge_units.jsonl` and `example_qa.jsonl` for real
output generated from a sample PDF by this exact pipeline.

## 5. Explanations (per deliverable checklist)

- **Semantic chunking**: `document_understanding/structure.py` detects
  paragraph boundaries from vertical spacing between text lines (not
  just headings), so an entire section is not treated as one blob.
  `chunking/semantic_chunker.py` then only splits a paragraph if it
  exceeds `max_chunk_chars`, and only at sentence boundaries, packing
  sentences up to `target_chunk_chars`. Tiny trailing fragments are
  merged back rather than emitted as context-free one-liners.
- **Scientific importance detection**: `extraction/importance.py`
  combines section weight, taxonomy-concept density, numeric-fact
  count, and importance-phrase hits (from `taxonomy.json`'s
  `importance_signals`) into one 0–1 score — never a single keyword
  check.
- **Scientific entity extraction**: `extraction/entities.py` matches
  both taxonomy topic surface forms and an alias table
  (`concept_aliases` in `taxonomy.json`), so abbreviations and
  synonyms normalize to one canonical concept id.
- **Three-domain classification**: `classification/domain_classifier.py`
  scores each of the 3 allowed domains by which taxonomy concepts
  were actually matched in that specific knowledge unit's text, picks
  the highest-scoring domain as `primary_domain`, and only adds a
  `secondary_domain` when its share of total score is meaningful
  (≥20%). A hard `assert` guarantees no other domain string can ever
  be returned.
- **Context extraction**: `extraction/context_extractor.py` runs
  entirely separately from domain classification, against
  `context_vocabularies` in `taxonomy.json` (crops, seasons, soil
  types, irrigation methods, locations, agro-climatic zones).
- **Evidence/finding/recommendation separation**: `knowledge/
  knowledge_builder.py` — `evidence` is the exact source passage;
  `finding` is assembled only from already-extracted structured facts
  and concepts (never freely generated); `recommendation` is `null`
  unless the source text itself contains recommendation language, in
  which case the exact source sentence is used and tagged
  `recommendation_type: source_explicit`.
- **Grounded Q&A generation**: `qa/qa_generator.py` builds every
  answer directly from fields already on the knowledge unit — a
  quantitative question's answer quotes the matching structured fact
  and the evidence sentence; nothing is invented.
- **Provenance**: every knowledge unit carries `document_id`, `page`,
  `section`, a whole-document `source_hash`, and an `evidence_hash` of
  its own text, so it is always traceable to an exact page/section of
  a specific source PDF.
- **Duplicate detection**: `dedup/deduplicator.py` — documents are
  deduplicated by DOI, then normalized title+first-author, then full
  content hash; knowledge units are deduplicated by exact evidence
  hash and by Jaccard similarity of 5-word shingles within the same
  `primary_domain` (so two different experimental conditions are never
  merged away).
- **Quality control**: `validation/validator.py` assigns one of
  `validated / needs_review / rejected / evidence_only` to every unit
  based on evidence presence, provenance completeness, domain
  validity, importance score, and classification confidence. Nothing
  is deleted — the status is recorded and visible in
  `validation_report.json`.
- **Canonical dataset → FAISS mapping**: `dataset/writer.py` writes
  the canonical JSONL first; `embeddings/embedder.py` embeds
  `retrieval.semantic_text` for validated/needs_review/evidence_only
  units only; `index/faiss_builder.py` persists an explicit
  `id_map.json` (FAISS row position → `knowledge_unit_id`) alongside
  `faiss.index`, so the index can always be traced back to, and
  rebuilt from, the canonical dataset.

## 6. Folder structure produced by a build

```
dataset/
    documents.jsonl              # one row per source PDF (metadata only)
    knowledge_units.jsonl        # the canonical corpus — one row per knowledge unit
    qa.jsonl                     # flattened Q&A pairs, joined to knowledge_unit_id
    dataset_statistics.json
    validation_report.json
    build_metadata.json          # reproducibility: versions, config, timestamp
    taxonomy.json                # copy of the taxonomy used for this build

index/
    faiss.index
    vectors.npy
    id_map.json                  # FAISS row -> knowledge_unit_id
    index_metadata.json          # embedding model/config used
```

## 7. Installation

```bash
pip install -r requirements.txt
```

`sentence-transformers` is optional — omit it and always pass
`--offline` to use the built-in TF-IDF embedder with zero downloads
and no commercial API dependency (master-spec section 28).

## 8. Commands

```bash
# Build the canonical dataset only
python build_dataset.py --input_dir ./papers --output_dir ./dataset

# Build dataset + FAISS index in one go
python build_dataset.py --input_dir ./papers --output_dir ./dataset --build-index

# Fully offline (TF-IDF, no downloads)
python build_dataset.py --input_dir ./papers --output_dir ./dataset --build-index --offline

# Use BGE-M3 or a multilingual model instead of the default MiniLM
python build_dataset.py --input_dir ./papers --output_dir ./dataset --build-index --model BAAI/bge-m3

# Force a full rebuild
python build_dataset.py --input_dir ./papers --output_dir ./dataset --rebuild

# Re-run validation only on an existing dataset
python build_dataset.py --output_dir ./dataset --validate

# Print dataset statistics
python build_dataset.py --output_dir ./dataset --stats

# Query the index (Top-K retrieval test)
python build_dataset.py --output_dir ./dataset --index_dir ./index \
    --query "effect of drip irrigation on water use efficiency in cotton" --top_k 5

# Rebuild an existing offline TF-IDF index after retrieval changes
python build_dataset.py --output_dir ./dataset --index_dir ./index \
    --skip-build --build-index --offline
```

## 9. Chat answer behavior

The ask page sends the current question and up to six recent turns from
the same chat to `/api/ask`. Retrieved records include source evidence,
extracted findings, structured facts, and available crop/soil/season
context. Only validated or reviewable records are used to generate
answers; evidence-only fragments are excluded. The language model uses
those records to answer the current
question in its own wording and cite its sources; it should not reuse a
previous answer when it is not relevant. Follow-up questions and
formatting requests (such as "make it short" or "write in one line") use
the most recent relevant question and conversation turns, instead of being
treated as standalone agricultural questions. The chat sends recent turns
and the API keeps a bounded context window. For non-English chats, recent
turns are translated to English for the model and the answer is
translated back to the selected language. Chat model weights are loaded
from the local Hugging Face cache without network checks. By default the base
instruct model generates answers; the fine-tuned adapter is opt-in with
`python api.py --use_adapter` because its training data can produce
repetitive, template-like responses. While an answer is generating, the
composer button becomes a stop control; cancellation is sent to `/api/cancel`
and checked during model token generation. Listen buttons toggle speech
playback, and clicking an active Listen button stops playback and cancels
ongoing synthesis. Bengali, Marathi, and English use separate, small local
MMS-VITS models. Speech is synthesized one sentence-sized chunk at a time
and streamed into browser playback; the next chunk is synthesized while
the current one plays. The model weights (about 145 MB per language) are
downloaded and cached on first use; answer text is not sent to a speech
service. MMS-TTS weights are licensed CC-BY-NC 4.0, so review that license
before any commercial use or redistribution. Speech uses the existing
PyTorch and Transformers API environment and needs no separate Parler-TTS
package. Other languages continue to use a matching browser/OS speech voice
when one is installed.
Standalone questions must concern agricultural research and retrieve
relevant dataset evidence before the model is called; otherwise the API
returns a dataset-scope or no-evidence message. Conversation history is
used for clear follow-ups, not appended to every standalone search.
Potato irrigation questions are matched to potato-specific records;
irrigation timing and quantities are not inferred from adjacent crops when
the dataset does not report them. Run `python evaluate_rag.py --smoke` for
a live regression check, or `python evaluate_rag.py --limit 25` to sample
labeled dataset Q&A pairs. Custom JSONL cases can include relevant
knowledge-unit IDs, reference answers, chat history, and formatting
expectations. The JSON report includes precision/recall@k, MRR, nDCG,
grounded-answer rate, cited-source ratio, latency, and optional token-F1
and ROUGE-L answer similarity. Lexical answer scores penalize valid
paraphrases; factual faithfulness needs human review or a separately
calibrated LLM judge.

## 10. Reproducibility

Every build writes `dataset/build_metadata.json` with the software
version, taxonomy version, Python version, embedding model/offline
flag, and the exact chunking/extraction thresholds used, so a dataset
can always be traced back to the configuration that produced it.
Knowledge-unit IDs (`SWM_000001`, ...) are assigned deterministically
after deduplication, in document order.

## 11. Known limitations / honest caveats

- Paragraph-break detection uses vertical line-spacing gaps from
  PyMuPDF's layout data; PDFs with unusual layouts (dense two-column
  scans, no consistent spacing) will fall back to one paragraph per
  section rather than finer splits — the sentence-level packer inside
  the chunker still prevents oversized chunks in that case.
  `pymupdf_layout` (PyMuPDF's dedicated layout-analysis add-on) can be
  installed and wired into `ingestion/pdf_loader.py` for meaningfully
  better multi-column and scanned-PDF handling if needed.
  - `finding` text is intentionally terse and mechanical (built purely
  from structured facts) rather than fluent prose, by design — it is
  a safety-first traceable restatement, not a paraphrase, so it can
  never introduce information the evidence doesn't support. If more
  natural phrasing is desired, an optional local LLM pass could be
  added as a clearly-labelled "LLM-assisted" post-processing step
  (see the `ML-based extraction` vs `LLM-assisted` distinction called
  for in master-spec section 28) — none is wired in by default so the
  system requires no external API.
- OCR for scanned/image PDFs is not implemented in this version;
  `ingestion/pdf_loader.py` extracts text and vector-based tables only.
  Adding an OCR fallback (e.g. Tesseract) is a contained addition to
  that one module.
