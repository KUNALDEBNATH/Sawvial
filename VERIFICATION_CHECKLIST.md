# Final Verification Checklist

Statuses: **FIXED AND STATICALLY VERIFIED** / **PARTIALLY FIXED** / **NOT FIXED**.
"Statically verified" means: re-read the modified file(s) after editing, checked every caller of
every changed function signature for compatibility, and `python3 -m py_compile` / `node --check`
passed. No test suite was run and the application was not launched, per the task's instructions —
these are explicitly called out below wherever that matters.

## 1. Hardware / GPU

| Item | Status | Notes |
|---|---|---|
| CUDA/GPU used by default for inference, embeddings, fine-tuning | FIXED AND STATICALLY VERIFIED | Already routed through `pick_device()` for inference/embeddings; embeddings now explicitly call it (was implicit before). Training already GPU-first. |
| GPU availability detected dynamically via PyTorch | FIXED AND STATICALLY VERIFIED | `sawvial/device.py` only ever asks `torch.cuda.*` at runtime; no `nvidia-smi` parsing anywhere. |
| Precision selection appropriate for hardware/PyTorch support | FIXED AND STATICALLY VERIFIED | `torch.cuda.is_bf16_supported()` gates bf16 vs fp16 vs fp32 consistently in `rag_chat.py` and now in `train_lora.py` (previously used fp16 on CPU too, which is now fixed). |
| Memory-efficient inference/training for 8 GB VRAM | FIXED AND STATICALLY VERIFIED | LoRA + gradient checkpointing + accumulation + small batch were already present; OOM retries and TTS LRU cache are new. Not runtime-measured against actual VRAM (see limitation below). |
| Mixed precision / gradient checkpointing / small batches / gradient accumulation | FIXED AND STATICALLY VERIFIED | Present in `train_lora.py` before and after this pass; the `--force_cpu` dtype bug is what was fixed. |
| Avoid unnecessary CPU execution when GPU available | FIXED AND STATICALLY VERIFIED | `train_lora.py`'s `--force_cpu` bug (model moved to CUDA regardless of the flag) was the one place this was actually violated; fixed. |
| CPU fallback only when CUDA unavailable or op unsafe on GPU | FIXED AND STATICALLY VERIFIED | Same fix as above; no other unconditional CPU path found. |
| Handle CUDA OOM errors appropriately | FIXED AND STATICALLY VERIFIED (not runtime-tested) | New retry/fallback logic in `rag_chat.py`, `api.py`, `embedder.py`; never exercised against a real OOM. |
| Don't assume `nvidia-smi` CUDA version == PyTorch CUDA version | FIXED AND STATICALLY VERIFIED | Never was assumed; confirmed by reading `device.py` and every `torch.cuda.*` call site. |
| No hardcoded device IDs / incompatible CUDA deps | FIXED AND STATICALLY VERIFIED | Only `{"":0}"`/device `0` appear, meaning "the current/only GPU" for `device_map`/`get_device_properties`, standard single-GPU usage, not a hardcoded multi-GPU assumption. No new dependency added anywhere in this pass. |
| Workloads don't exceed 8 GB unnecessarily | FIXED AND STATICALLY VERIFIED | TTS LRU cache is the one place this was actually being violated (unbounded growth); fixed. |

## 2. Online research / retrieval decision logic

| Item | Status | Notes |
|---|---|---|
| `auto`/`always`/`never` research decision logic | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | Confirmed `want_online` logic in `api.py` is correct for all three modes; no change needed. |
| Actual question relevance checked, not just score/keyword | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | `aspect_filter` + `crop_missing` in `passages.py` already go beyond score; re-verified by reading the code, not by running a live query. |
| Online results enter the same generation pipeline as local | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | Both merged into one `passages` list before generation/grounding in `api.py`. |
| `safe_http.py` protections preserved | FIXED AND STATICALLY VERIFIED | File read in full; zero lines changed. |

## 3. Crop and question relevance

| Item | Status | Notes |
|---|---|---|
| Primary/comparison/passing-mention/multi-crop distinction | FIXED AND STATICALLY VERIFIED (correction — this was wrongly marked "pre-existing, verified" in an earlier pass) | A live reproduction showed a multi-crop list ("cotton, pearl millet, maize, chillies, tomato etc.") being served as tomato-specific evidence. Root cause and fix are in FIX_REPORT.md §2.2: `crop_mask()` and `passages.py`'s own filter both had an unconditional `if crop in labels` branch that bypassed the "at most one other crop" check whenever the crop was already in `context.crop` (the common case, since that field has no primary/passing-mention distinction). Both now always apply the check. Not re-run against the live index after fixing (would require rebuilding the index and querying it, which this task's "no app launch" instruction excludes) — verified by code tracing and by checking that the one existing test aimed at this behavior (`test_listing_a_crop_among_many_is_not_crop_evidence`) uses a fixture that bypasses the buggy branch entirely (its record is `needs_review` and excluded from the index before `crop_mask` runs), which is also why it didn't catch this. |
| Soil/fertilizer/pH/irrigation/conservation intent understanding | FIXED AND STATICALLY VERIFIED for the reproduced case; PARTIALLY FIXED overall | A second live reproduction found the "soil suitability" aspect rule matching a passage about saline *irrigation water* (not soil) purely because "saline" is ambiguous vocabulary; fixed in FIX_REPORT.md §2.3 by requiring the ambiguous terms (saline/sodic/acidic/alkaline) to actually appear near the word "soil". Re-verified against the real failing text and against the existing aspect test scenarios (no regression). I also directly searched the project's actual `dataset/knowledge_units.jsonl` and confirmed a genuine, validated, on-topic record (`SWM_003990`, tomato's moderate salinity tolerance + drainage) now passes both the crop and aspect checks where it previously could have been crowded out - this is a real, data-grounded check, not just code reading. The "fertilizer/nutrients" and "water requirement" rules were not similarly stress-tested with a real failing example and may have analogous ambiguous-vocabulary issues not yet found. |
| Reference lists / bibliography text served as "evidence" | FIXED AND STATICALLY VERIFIED (data-grounded) | Found while checking the above: record `SWM_004949` is a bibliography fragment (section mis-tagged as `"maom"`, a content heading, instead of "references") that passed both the crop and soil-aspect checks purely because a cited paper's title mentions "tomato"/"soils". Added `_looks_like_bibliography()` to `passages.py` (3+ "(YEAR)" citation markers -> excluded before any other filter). Verified against the actual flagged record (now excluded) and measured its effect across the whole dataset: 175/6,432 validated records (≈2.7%) match this pattern and are now excluded from retrieval. The underlying cause - `structure.py`'s section-heading detection not always recognising a "references"/"bibliography" heading - was not root-caused or fixed; this guard addresses the symptom where it actually matters (never presenting a reference list as evidence) without risking a deeper, untested change to heading detection. |
| Crop-specificity check scoped to whole passage, not sentence | FIXED AND STATICALLY VERIFIED as a general improvement; **NOT CONFIRMED** against the specific reported case | After the §2.2 fix, online research for the same question read 12 real documents but found no relevant passages in any of them - I cannot reproduce this (no internet access in this environment) and am not claiming to know it was caused by this scoping issue specifically. What I *can* confirm: the crop check compared "other crops anywhere in the whole passage," which is unnecessarily strict for longer excerpts (more plausible for freshly-extracted web text than the curated dataset). Rescoped to "other crops in the same sentence as the crop's mention" (`_other_crops_near_mention`, shared between `retriever.py` and `passages.py`). Verified: still correctly excludes the original six-crop-in-one-sentence case from §2.2 (unaffected), and a constructed long-excerpt-with-unrelated-aside case is no longer wrongly excluded. **Not verified:** whether this is actually why those 12 documents were rejected - that would require the `report["documents"]` list (titles/URLs of what was fetched) from the live request, which has not been shared. If online research is still coming up empty after this change, the live search provider may simply not be returning documents that discuss this specific question at all, which is not a code bug. |

## 4. LLM answer generation / fallback / grounding

| Item | Status | Notes |
|---|---|---|
| Answers the actual question, preserves values/units/conditions | FIXED AND STATICALLY VERIFIED | Grounding (`_supports`) already checks numbers appear in evidence; prompt fixed to stop truncating answers (§3.1 of FIX_REPORT). |
| Avoid rigid length restrictions | FIXED AND STATICALLY VERIFIED | "2 to 4 sentences, no more" removed from `SYSTEM_PROMPT`. |
| No invented findings / appropriate citations | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | Citation-existence and claim-support checks in `grounding.py` were already present. |
| Communicate uncertainty when evidence insufficient | FIXED AND STATICALLY VERIFIED | `NO_DATA_ANSWER` path preserved and now only reached after the new regeneration attempt also fails. |
| No incomplete/randomly-truncated passages presented as generated answers | FIXED AND STATICALLY VERIFIED | Regeneration step added before any excerpt fallback; existing excerpt fallback was already clearly labelled as quoted source text, not a generated answer, and is now a true last resort. |
| One controlled regeneration + re-validation on grounding failure, else clear "insufficient evidence" | FIXED AND STATICALLY VERIFIED (not runtime-tested) | Implemented exactly as specified in `api.py`; the actual LLM was never invoked to confirm the regenerated prompt produces a better answer in practice. |
| Citation/source-id validation, number/unit/crop consistency in grounding | PARTIALLY FIXED | Source-id existence, number-in-evidence, and overclaim checks were already present and verified. A dedicated "crop name mentioned in the claim but absent from the cited evidence" check was considered but not added (see FIX_REPORT limitations) — this is the one sub-item of task §6 not separately implemented. |

## 5. Frontend rendering

| Item | Status | Notes |
|---|---|---|
| Headings, paragraphs, bullets, numbered lists, bold, links, code render correctly | FIXED AND STATICALLY VERIFIED (not browser-tested) | New `renderMarkdown()` added and wired into both `messageHtml()` and `typeOut()`. Syntax-checked with `node --check`; not exercised in an actual browser DOM. |
| Safe rendering/sanitization, never raw LLM HTML into the page | FIXED AND STATICALLY VERIFIED | All text passes through `escapeHtml()` before any markup is reintroduced; links restricted to `http(s)://`. |
| Seven-language support preserved | FIXED AND STATICALLY VERIFIED | `m.lang` attribute handling untouched. |
| Chat history preserved | FIXED AND STATICALLY VERIFIED | Raw Markdown text (not HTML) is still what's stored in `localStorage`; rendering happens at display time. |
| Streaming responses preserved | FIXED AND STATICALLY VERIFIED | Typing animation logic unchanged; only the post-animation render step changed. |
| Request cancellation preserved | FIXED AND STATICALLY VERIFIED | `stop()` now also routes through `renderMarkdown()` on the partial text, otherwise unchanged. |
| Source display preserved | FIXED AND STATICALLY VERIFIED | `sourcesHtml()` call sites untouched. |
| Copy functionality preserved | FIXED AND STATICALLY VERIFIED | `navigator.clipboard.writeText(m.text)` still reads the raw stored text, unaffected by the rendering change. |
| Speech functionality preserved | FIXED AND STATICALLY VERIFIED | TTS code paths read `m.text` directly, unaffected. |

## 6. Data processing and research quality

| Item | Status | Notes |
|---|---|---|
| Page boundaries / paragraph attribution correct | FIXED AND STATICALLY VERIFIED (heuristic, not PDF-tested) | Sentence-end heuristic added at page boundaries; inherently a heuristic, not provably correct on every real PDF without inspecting rendered output, which this task excludes. |
| Hyphenation / broken word joining fixed | FIXED AND STATICALLY VERIFIED | Root cause (space-joined buffer defeating the `-\n` regex) identified and fixed with a traced example in the FIX_REPORT. |
| Table content included in document identity/hashing | FIXED AND STATICALLY VERIFIED | `PageContent.plain_text` now serializes table rows; confirmed `structure.py` doesn't double-count them in paragraph text. |
| Numeric extraction: large/decimal/negative/grouped/ranges/CIs/units, no overlap/duplication | FIXED AND STATICALLY VERIFIED | Large/grouped/decimal/negative numbers were already handled; the overlap/duplication bug (CI and p-value spans never marked consumed) is fixed and traced in the FIX_REPORT. |
| Dedup preserves distinct findings (values/conditions/treatments/crops/outcomes) | FIXED AND STATICALLY VERIFIED | Numeric + crop fingerprint added to the near-duplicate check. |
| Unsafe title-only document dedup when author missing | FIXED AND STATICALLY VERIFIED | `title_author` key now requires a non-empty author; falls back to content hash otherwise. |
| Dataset validation doesn't overwrite human review decisions | FIXED AND STATICALLY VERIFIED | `cmd_validate` now preserves `verified`/`pending_review`/`published`/`published_as` records. |
| Online research deadline handling | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | Per-candidate deadline check in `live.py` was already correct; unchanged. |
| Online research cancellation propagation | FIXED AND STATICALLY VERIFIED | `cancel_event` now threaded into `ingest_url()` and checked at each major step. |
| Concurrent JSONL updates in the research store | FIXED AND STATICALLY VERIFIED | Per-file `threading.Lock` added around every read-modify-write cycle in `store.py`. In-process only — see limitation in FIX_REPORT (no OS-level lock for multi-process deployments). |
| Preserve existing research trust/review workflows | FIXED AND STATICALLY VERIFIED | `TRANSITIONS`, flag checks, and reviewer/notе logging in `store.py` untouched. |

## 7. Model/index/speech components (task #9)

| Item | Status | Notes |
|---|---|---|
| Offline model loading works when offline mode enabled | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | `local_files_only=settings.OFFLINE` already threaded through `rag_chat.py`/`speech.py`; unchanged. |
| Index rebuild preserves configured embedding model/mode | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | Not modified; read `retriever.py`'s `ensure_index_bundle` and confirmed the configured model name round-trips through the persisted bundle metadata. |
| Training uses the selected CUDA device correctly | FIXED AND STATICALLY VERIFIED | Covered by the `--force_cpu`/`cuda_ok` fix above. |
| `--force_cpu` genuinely forces CPU | FIXED AND STATICALLY VERIFIED | Explicitly fixed; traced through dtype, 4-bit guard, model placement, `TrainingArguments`. |
| GPU precision/memory appropriate for RTX 4060 8GB | FIXED AND STATICALLY VERIFIED (not runtime-measured) | See GPU_CONFIGURATION.md. |
| CPU-only and GPU-only paths don't conflict | FIXED AND STATICALLY VERIFIED | Single `cuda_ok` flag now drives every related decision in `train_lora.py`, removing the previous inconsistency. |
| TTS caching doesn't unnecessarily consume GPU memory | FIXED AND STATICALLY VERIFIED | LRU cache cap added. |
| Evaluation checks actual response language, not just requested | FIXED AND STATICALLY VERIFIED | `evaluate_rag.py` now compares against `text_language` instead of the tautological `answer_language`. |
| Source/passage IDs counted correctly in evaluation | FIXED AND STATICALLY VERIFIED | `evaluate_rag.py` now flattens each source's grouped `ids` list instead of counting one id per document. |

## 8. Documentation and configuration (task #10)

| Item | Status | Notes |
|---|---|---|
| Misleading local-processing/offline/privacy claims | FIXED AND STATICALLY VERIFIED | README intro reworded to stop contradicting its own online-research/browser-speech sections. |
| GPU/model requirement claims accurate | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | Setup/health-check sections checked against actual code; no inaccuracies found beyond the intro paragraph. |
| No invented specs/benchmarks/performance claims | FIXED AND STATICALLY VERIFIED | README already explicitly disclaims unverified performance ("have not been verified by me"); nothing invented was found or added. |

## 9. Cross-cutting static verification (task §11)

| Item | Status | Notes |
|---|---|---|
| Every modified function has consistent callers/arguments | FIXED AND STATICALLY VERIFIED | Grepped every call site of every changed signature (`ingest_url`, `make_user_prompt`, `rag.answer`, `SentenceTransformerEmbedder`, `_load_model`) — all new parameters are optional with defaults, so no caller needed updating. |
| Imports/module references consistent | FIXED AND STATICALLY VERIFIED | `python3 -m py_compile` passed on every modified `.py` file; `node --check` passed on `chat.js`. |
| Configuration values connected to the application | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | `settings.py` values used throughout were traced to their consumers; no orphaned config found. |
| GPU selection applied consistently | FIXED AND STATICALLY VERIFIED | See §1/§7 above. |
| No old logic still bypasses the corrected implementation | FIXED AND STATICALLY VERIFIED | E.g. confirmed there is only one call site for `clean_page_text`/`_dehyphenate` (so the join-character fix fully applies) and one place `document_dedup_key`/`deduplicate_knowledge_units` are used. |
| No unrelated working features removed | FIXED AND STATICALLY VERIFIED | Every change was additive or a targeted replacement of a specific buggy expression; `git diff --stat` shows only the 18 files intentionally touched. |
| Local RAG and online research use the same answer pipeline | FIXED AND STATICALLY VERIFIED (pre-existing, re-verified) | Confirmed in §2 above. |
| Frontend/backend request-response formats compatible | FIXED AND STATICALLY VERIFIED | New `gpu_out_of_memory`/`regenerated_after_grounding_failure` fields are additive (new status value / new optional note); no existing field removed or retyped, so existing frontend parsing of `res.answer`, `res.grounding.notes`, etc. is unaffected. |
| No unnecessary files/dependencies introduced | FIXED AND STATICALLY VERIFIED | Only the three required deliverable `.md` files were added; the Markdown renderer was hand-written with no new library. |

## What could not be verified without running the application

- Actual GPU/VRAM behaviour (precision chosen at runtime, real OOM handling, real VRAM figures).
- Whether the regenerated LLM answer after a grounding failure is actually better in practice.
- Whether the Markdown renderer looks correct in an actual browser.
- Whether the page-boundary paragraph-break heuristic behaves correctly on your actual PDF corpus.
- Whether `pytest` passes — not run, per instructions.
