# SAWVIAL change log

## 2026-10 (round 3): no API keys, GPU-first, one setup snippet

* Removed every API key and the key-based/optional search providers (Brave, SearXNG). Search = key-less OpenAlex (open-access works only) + Crossref.
* GPU: `SAWVIAL_DEVICE=cuda` is the default; a CPU-only PyTorch build is refused with the exact fix. The LLM runs in 16-bit on the GPU (4-bit only with `--4bit`);
  health reports the device and VRAM. `numpy<2` and `bitsandbytes` pins removed from the runtime requirements.
* The search index is built/repaired automatically at start-up (missing, half-committed, old format, other scikit-learn version, stale dataset).
* README reduced to one setup-and-run snippet; maintenance material moved to `docs/MAINTENANCE.md`.

## 2026-10 (round 2): language handling, research inside /api/ask, Tamil audio

* **Language**: `input_language` and `answer_language` are separate (`sawvial/language.py`); an English question with Tamil selected is no longer
  translated "from Tamil". Responses carry `input_language`, `answer_language` (requested), `text_language` (what the text really is), `translation`, `warnings`.
  Every early return (greeting, out-of-scope, no evidence, cancelled, mock) is translated consistently; failures are visible, never silent.
* **Research in the chat**: local first -> relevance check (crop + aspect + score) -> `sawvial/research/live.py` (search, fetch, extract, rank) -> grounded answer.
  Streaming endpoint `/api/ask/stream` shows real stages. New settings `SAWVIAL_ONLINE_RESEARCH`, `SAWVIAL_WEB_SEARCH_*`, `SAWVIAL_RESEARCH_*`; `.env` is now loaded.
* **Search/trust**: Brave/SearXNG (optional) + OpenAlex + Crossref; trust tier decided from the FINAL url; robots.txt, per-host delay, caching; HTML extractor strips boilerplate, decodes charsets.
* **Publishing** is a transaction (dataset snapshot/rollback, index swap, mark-published last).
* **Grounding**: overclaim words must appear in the evidence; verbatim-excerpt last resort instead of a generic refusal.
* **Frontend**: stage labels, `text_language`-based display, translation banner, source kind/origin badges, differing values, speech language chosen from the text itself.
* **Fixed**: `index/` half-committed (heavy files git-ignored) now gives a clear rebuild message; tomato *irrigation* passages no longer count as evidence for a tomato *soil* question.

---


## 2026-10 audit: grounding, retrieval, ingestion, research, frontend

See README.md for usage. Summary of what changed and why (details in the final audit report):

* **Adapter** loaded by default after a compatibility check; explicit `--base_only`; clear errors (`sawvial/model_config.py`, `llm/rag_chat.py`).
* **Index** rebuilt as a persisted TF-IDF vectorizer + sparse matrix of *answerable records only*, atomic swap, staleness detection; the old
  never-used `faiss.index` / `vectors.npy` (93 MB) are gone. Optional dense/hybrid via `--dense`.
* **Eligibility** single rule (`sawvial/eligibility.py`) for build, index and runtime; `needs_review`/`evidence_only`/`pending_review`/`rejected` no longer answerable.
* **Crop gating**: crop questions only get crop-specific evidence, otherwise "not enough evidence" (LLM not called).
* **Grounding** (`sawvial/grounding.py`): citations must exist, sentences must be supported by the cited text incl. numbers; unsupported sentences removed.
* **API**: metadata (authors/year/journal/DOI/URL/page) passed through, narrow inference lock, timings, validation, rate limit, CORS allow-list.
* **Frontend**: random demo-question fallback removed, specific mic errors, retry, safe source links, answer-language-aware speech.
* **PDF ingestion**: scanned/partial detection, optional OCR, table-error reporting, tables attached by reading position, ingestion report.
* **Dataset**: QA checks (title-only, reference lists, off-topic, fragments), `tools/audit_dataset.py`, honest statistics, held-out split shared with evaluation.
* **Research workflow** (`research_cli.py`, `sawvial/research/`): safe fetching, pending_review -> verified -> published.
* **Evaluation/tests** rewritten; 100+ pytest tests.

Removed on purpose: the hard-coded potato-irrigation answer (`crop_irrigation_summary`) and the dataset-wide crop summary that bypassed retrieval and
the LLM and marked itself grounded; dead `demoAnswers` data in the frontend; unused `STRONG` score setting.

---

## Earlier fixes (backend answer quality)

All in `api.py` and `llm/rag_chat.py`:

1. **Relevance cut-off too high** - `MIN_RETRIEVAL_SCORE` 0.11 -> 0.03. TF-IDF scores for good
   matches are 0.05-0.25, so ordinary questions ("how much water does my field need this week")
   were rejected with "couldn't find relevant, reliable evidence".
2. **Follow-up detection was too greedy** - words like "this", "that", "it" anywhere in a question
   made it a "follow-up", mixing the previous topic into retrieval. Pronouns now only count in
   short messages (<= 7 words) or after openers like "and", "what about", "why".
3. **Out-of-scope filter missed farming words** - rice, paddy, wheat, maize, tomato, seed, sow,
   harvest, weeds, pesticide, etc. are now recognised.
4. **Answers thrown away for small LLM slips** - invented citation ids, `[A_1, B_2]` style
   citations, empty output, or not repeating the crop name used to return the dead-end message.
   Now invented ids are ignored and, if the LLM output is unusable (or it says "not enough
   information" although retrieval was strong), the best matching evidence sentences are returned
   with their source id instead.
5. **`repetition_penalty` 1.1 -> 1.0** in `llm/rag_chat.py` - the penalty mangles repeated tokens
   such as `SWM_000123`, which caused the invented-citation rejections.
6. **Translator errors no longer crash the request** - falls back to the English answer.
7. **Misleading numbers** - dataset "findings" like "a range of 2.0-3.0 was reported" are auto-generated
   and unit-less; they are no longer shown to the LLM (`llm/rag_chat.py`).
8. **Answer style** - system prompt (`llm/common.py`) now asks for 2-4 sentences, no dangling
   "found that..." sentences, no author names, no unit-less numbers.
9. **Raw `[SWM_000671]` in canned answers** - citation tags are stripped from the potato-irrigation and
   crop-list answers (sources are still listed under "Sources").
