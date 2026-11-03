# SAWVIAL — Complete End-to-End Reliability and Answer-Quality Repair

Repository: https://github.com/KUNALDEBNATH/Sawvial
Base branch: main
Working branch: codex/sawvial-end-to-end-fixes
Base commit reviewed: f8cb62a441b3063afcd24fd688fe1771c16c49ea

## Mission
Inspect this current branch and IMPLEMENT every confirmed issue below. Do not stop at a report or plan. Make changes, add regression tests, run tests/evaluation where feasible, prepare a clean runnable source ZIP, commit the work to the working branch, and open a DRAFT pull request to main. Do not modify main directly, do not merge, and never claim a test passed unless it actually ran.

Keep the existing architecture and dependencies wherever practical. Avoid overengineering and unnecessary models/frameworks. Preserve source provenance, citation checks, SSRF protections, source trust checks and human review before publishing online evidence.

## Required end-to-end answer flow
Both local RAG evidence and live online-research evidence must pass through the same LLM generation path:

User query → language/intent detection → local RAG → question-aware evidence adequacy → if insufficient, live OpenAlex/Crossref search → safe fetch and PDF/HTML extraction → temporary retrieval and relevance validation → combine only relevant evidence → LLM generates a fresh, natural answer in its own words → claim-level citation/factual/intent grounding → exactly one bounded regeneration if validation fails → final answer or honest insufficient-evidence response → safe Markdown frontend rendering → translation → optional TTS.

Do not show raw retrieved paragraphs as if they were generated answers. Do not fabricate facts. Preserve sources/citations separately.

## Confirmed issues to fix

### 1. Retrieval relevance and false crop matches — CRITICAL
Inspect sawvial/extraction/context_extractor.py, sawvial/index/retriever.py, sawvial/index/passages.py and api.py.
A crop mention or crop label must not automatically mean the source provides evidence about that crop and the requested intent. The observed tomato question returned an unrelated list from a multi-crop yield/water-use meta-analysis.
Distinguish primary study crop / actual crop-specific finding, comparison crop, passing mention, multi-crop overview/list, and genuinely relevant crop-specific evidence. Preserve valid multi-crop findings where independently attributable. Add intent/aspect handling for soil suitability, pH/acidity, fertilizer/nutrients, irrigation methods, water requirements/scheduling, conservation, etc. A question such as 'Which soil is good for tomatoes?' must not be answered by a source merely listing tomato among crops. Grounding must separately verify that the response addresses query intent, not just lexical overlap.

### 2. Online research skipped when local passages are irrelevant — CRITICAL
Inspect api.py run_ask(), sawvial/research/live.py, sawvial/research/search.py, sawvial/settings.py and sawvial/js/sawvial-api.js.
Current auto decision uses local passage existence and score, so irrelevant evidence can be considered sufficient and OpenAlex/Crossref skipped.
Replace this with evidence adequacy based on intent, crop/condition coverage and relevance. In research='auto', call live_research when local evidence is missing, weak, irrelevant, wrong-crop, wrong-aspect or insufficient. 'always' must attempt research regardless of local evidence; 'never' must not call online search. Check SAWVIAL_ONLINE_RESEARCH, provider configuration and frontend payload; explicitly send research='auto' by default and report actual settings via /api/health. Avoid unnecessary online search when local evidence is clearly sufficient. Do not let irrelevant local passages crowd out better online passages.
Tests must prove: irrelevant local triggers discovery; relevant local avoids discovery in auto; always invokes discovery; never does not; disabled mode is honest; provider failure is honest; crop/aspect insufficiency triggers; cancellation works.

### 3. LLM natural answer generation — CRITICAL
Inspect llm/common.py, llm/rag_chat.py and api.py.
Both evidence origins use same LLM generation stage. The model must understand and explain relevant evidence in fresh wording, natural professional conversational style, with enough detail for the question. Use headings/bullets only when useful. Avoid filler and artificially inflating token count. Preserve citations, exact values/units/conditions, distinguish findings from recommendations, surface uncertainty and conflicts, and never invent advice. Do not force every answer into a rigid two-sentence template. Align generation arguments with greedy/sampling configuration to eliminate ignored-parameter warnings without unintentionally changing behaviour. Replace deprecated torch_dtype only if compatible with supported Transformers versions.

### 4. Lexical grounding is insufficiently question-aware — CRITICAL
Inspect sawvial/grounding.py and api.py. Separately validate citation identity, factual support from cited passage, numbers/units, crop/variety/region/season/experimental-condition consistency, relevance to query intent and whether the response answers the question. Keep claim-level checks. Do not claim lexical matching proves scientific correctness; document limitations.

### 5. Extractive fallback returns raw or incomplete research text — CRITICAL
Inspect api.py and sawvial/grounding.py. Current extractive_answer()/excerpt_answer() can return raw source text after model output fails (visible in screenshot). Instead: first generation failure → one controlled regeneration with concise validation feedback and same relevant evidence → validate. If it still fails, show honest insufficient-evidence/unable-to-verify text and keep source references separately. Never silently present raw passages as a generated answer. If an extractive mode is retained, it must be explicitly labelled, preserve sentence boundaries and not be misrepresented.

### 6. Frontend lacks safe Markdown rendering — HIGH
Inspect sawvial/js/chat.js, sawvial/js/sawvial-api.js, sawvial/ask.html and relevant CSS/i18n. messageHtml() currently escapes the entire answer as plain text, so Markdown is not rendered. Add safe Markdown rendering for paragraphs, headings, bold/italic, lists, and useful tables; sanitize HTML and external links. Use existing dependencies if suitable or add a small maintained dependency only if justified. Never inject raw LLM HTML. Preserve source cards, copy/listen/history/language/streaming/cancellation. Add frontend security/render tests.

### 7. PDF paragraph boundary, hyphenation and table identity — HIGH
Inspect sawvial/document_understanding/structure.py, sawvial/ingestion/pdf_loader.py, sawvial/pipeline.py and sawvial/dedup/deduplicator.py.
Flush paragraph buffers at page boundaries; preserve line breaks until dehyphenation; keep page attribution. PageContent.plain_text and RawDocument.full_plain_text include only text blocks, not tables, so document identity can ignore table-value differences. Include normalized table content in identity/hash while preserving page/row/column provenance; do not mix unrelated tables into prose.

### 8. Numeric extraction correctness — HIGH
Inspect sawvial/extraction/numeric.py. Regex can misparse ungrouped four-digit/larger numbers (e.g. 1200). Support arbitrary digits, decimals, signs, grouped thousands, numeric boundaries and units. Prevent confidence intervals being double-counted as ranges/measurements. Test 1200 kg/ha, 12500, 0.75, 1,200, ranges, percent, p-values and confidence intervals.

### 9. Deduplication may discard distinct findings — HIGH
Inspect sawvial/dedup/deduplicator.py. Jaccard near-duplicate logic may drop different studies with similar wording but different numeric values, crop, treatment, soil, location, season or conditions. Implement conservative structured comparison and provenance awareness; never discard distinct findings merely because prose is similar. Fix title+empty-author collisions. Add regression tests.

### 10. Online research time budget/cancellation — HIGH
Inspect sawvial/research/live.py, search.py, safe_http.py, workflow.py. Current deadline starts after provider discovery; cancellation is not propagated into discovery/fetch. Start total deadline before discovery, propagate cancellation/remaining budget and bound network requests by remaining time. Preserve SSRF safety.

### 11. ResearchStore concurrency — HIGH
Inspect sawvial/research/store.py. JSONL read-modify-write methods (add_source, add_records, set_status, etc.) share a temporary filename without locks. Concurrent requests can lose records/allocate duplicate IDs. Add process/thread-safe locking or a small SQLite transactional store if justified. Ensure atomic writes; add concurrency tests.

### 12. Validation overwrites human review — MEDIUM
Inspect build_dataset.py cmd_validate(). Preserve verified/rejected/published states; store automatic validation result separately. Never silently downgrade human decisions. Use atomic writes.

### 13. Offline model and dense index rebuild — MEDIUM
Inspect sawvial/embeddings/embedder.py, sawvial/index/retriever.py, faiss_builder.py, settings.py and llm/rag_chat.py. SentenceTransformer must respect offline/local-only mode; ensure_index_bundle currently rebuilds without preserving dense configuration. Respect offline mode and preserve/report lexical-only mode accurately. Never claim dense search when only lexical is active.

### 14. CPU/GPU LoRA training — MEDIUM
Inspect llm/train_lora.py. --force_cpu does not consistently control placement/dtype; CPU + 4-bit must be rejected; CPU should use float32; derive use_cuda consistently and handle empty datasets and eval scheduling for small datasets.

### 15. TTS memory — MEDIUM
Inspect llm/speech.py. Per-language cache can retain multiple GPU models. Bound cache or evict/unload prior models while preserving streaming/cancellation.

### 16. Evaluation correctness — MEDIUM
Inspect evaluate_rag.py. Multilingual metric currently checks answer_language (requested) instead of text_language (actual). Fix. Check passage IDs: source grouping can aggregate passages; recall must use actual relevant passage IDs, not only document representative IDs.

### 17. Privacy/model documentation — LOW/MEDIUM
Inspect sawvial/index.html, README.md, llm/sawvial-lora/README.md. Clarify local processing vs online search provider queries, API deployment, and browser ASR vendor. Fill model-card placeholders only with known facts; do not invent license, provenance or performance claims.

## Compatibility requirements
Preserve auto/always/never research modes, multilingual translation, local and browser TTS fallback, NDJSON streaming/cancellation, history/follow-ups, source metadata/pages/DOIs/URLs, pending_review→verified/rejected→publish workflow. Do not weaken safe_http SSRF/redirect/robots/trust checks. No production canned answers. No ungrounded answer when retrieval/search fails. Avoid retraining LoRA unless evidence demonstrates need; first fix retrieval, prompts, grounding and fallback.

## Test requirements
Add tests for tomato list false-positive and crop/intent relevance; auto online-search trigger/avoid, always/never/disabled; search provider failure, timeout, cancellation; combined local+web ranking; both sources reaching same LLM generation/grounding; single regeneration and no raw-text fallback; invalid citations, wrong source, invented numbers/units, irrelevant evidence; safe Markdown and sanitization; PDF page boundary/hyphenation/table hashing; large numbers and CI overlap; dedup distinct numeric/context findings; concurrent ResearchStore writes; human review status preservation; offline/dense rebuild; CPU/4-bit guards; TTS cache bounds; multilingual actual-text metric and recall IDs.

Run python -m pytest -q. Run frontend test/lint/build if present. When API/model is available run python evaluate_rag.py --api http://127.0.0.1:8000 --multilingual --api_limit 20. Run separate online-search integration tests (mock fixtures acceptable in CI, but clearly distinguish from live tests), plus syntax/import checks. If models/GPU/network/Tesseract are unavailable, document exactly which tests could not run. Never fake passing results.

## Required deliverables on this working branch
1. Actual implemented code changes, not just a plan.
2. Regression tests.
3. FIX_REPORT.md: each issue, root cause, files changed, solution, actual test commands/results and unresolved risks.
4. Updated ARCHITECTURE.md with end-to-end construction, online research and answer flows.
5. RUN_WINDOWS.md with exact PowerShell commands for venv, dependencies in correct order (CUDA PyTorch if needed), .env checks, model download/cache, dataset/index build if needed, API start, tests and evaluation.
6. SAWVIAL_FIXED_SOURCE.zip: clean runnable source snapshot excluding .venv, __pycache__, .pytest_cache, transient build/temp directories and secrets. Include dataset/index only if necessary and safe; document how to rebuild/download large models. Never include secrets. ZIP must not contain itself.
7. Commit all changes to branch codex/sawvial-end-to-end-fixes.
8. Create a DRAFT PR to main with summary and truthful test status. Do not merge.

Completion means code committed, tests run honestly, reports present, ZIP generated and downloadable as a GitHub Actions artifact or PR/release asset where supported, draft PR exists. If a delivery mechanism is unavailable, say so; still finish supported tasks.

Final message must include branch, draft PR, ZIP (only if actually uploaded and retrievable), report, architecture, exact test summary and remaining limitations.