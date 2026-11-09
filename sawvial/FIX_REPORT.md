# SAWVIAL — Fix Report

Scope note: this repository already shows multiple prior rounds of fixes (see `CHANGES_FIXES.md`) —
GPU defaults, aspect/crop-aware retrieval, grounding/citation checks, SSRF protection, etc. This
pass re-inspected the whole flow named in the task brief, verified what was already correct, and
fixed the genuine remaining defects below. Items that were already correctly implemented are noted
as "verified, not changed" rather than re-flagged as bugs.

---

## 1. GPU / hardware (RTX 4060, 8 GB)

### 1.1 `--force_cpu` did not actually force CPU — **FIXED**
**File:** `llm/train_lora.py`
**Root cause:** dtype selection and `model.to("cuda")` only checked `torch.cuda.is_available()`,
ignoring `args.force_cpu`. On a machine with a GPU, passing `--force_cpu` still moved the model to
CUDA while `TrainingArguments.use_cpu` was set from the same flag — a device mismatch / contradicted
intent.
**Fix:** introduced a single `cuda_ok = torch.cuda.is_available() and not args.force_cpu` flag used
consistently for dtype (`bf16`/`fp16`/`float32`), 4-bit guard (QLoRA now refuses to run under
`--force_cpu` with a clear error instead of silently misbehaving), model placement, and
`TrainingArguments(use_cpu=not cuda_ok, fp16=cuda_ok and not bf16)`.

### 1.2 No CUDA out-of-memory handling anywhere — **FIXED**
**Files:** `llm/rag_chat.py`, `api.py`, `sawvial/embeddings/embedder.py`
**Root cause:** the task explicitly requires "Handle CUDA out-of-memory errors appropriately"; no
code in the project caught `torch.cuda.OutOfMemoryError` anywhere.
**Fix:**
- `rag_chat.answer()`: on OOM, frees the cache and retries generation once with half the
  `max_new_tokens` budget.
- `api.py` `run_ask()`: if generation OOMs even after that in-function retry, the request returns a
  clean `gpu_out_of_memory` status with a user-facing message instead of a 500/crash. The new
  regeneration call (see §5.3) is wrapped the same way.
- `SentenceTransformerEmbedder.encode()`: on OOM, halves its batch size and retries (useful when
  embedding many/long passages during index building on an 8 GB card).

### 1.3 Dense embedder did not use the project's central device logic — **FIXED**
**File:** `sawvial/embeddings/embedder.py`
**Root cause:** `SentenceTransformer(model_name)` let the library auto-pick its device instead of
going through `sawvial.device.pick_device()`, so it could silently diverge from `SAWVIAL_DEVICE`
and from the device the LLM is using.
**Fix:** `SentenceTransformerEmbedder.__init__` now calls `pick_device()` and passes an explicit
`device=` to `SentenceTransformer(...)`, falling back to the library's own default only if
`pick_device()` itself raises (e.g. GPU required but unavailable).

### 1.4 TTS models cached without bound, unnecessarily holding GPU memory — **FIXED**
**File:** `llm/speech.py`
**Root cause:** all 7 per-language VITS models were cached forever in `_MODELS`, each on the GPU if
one was available, with no eviction — unbounded VRAM growth as users switch languages.
**Fix:** added a small LRU cache (`_MAX_CACHED_MODELS = 2`); loading a model now evicts the
least-recently-used one (and calls `torch.cuda.empty_cache()` if the evicted model was on the GPU)
once the cap is reached.

### 1.5 Device detection, precision selection, device IDs, memory-efficient training — **verified, not changed**
`sawvial/device.py` dynamically detects CUDA via `torch.cuda.is_available()`, never hardcodes a
device index, and does not assume `nvidia-smi`'s reported CUDA version equals PyTorch's. LLM
loading (`llm/rag_chat.py`) already picks `bfloat16`/`float16` based on actual hardware support and
loads the ~3 GB Qwen2.5-1.5B model comfortably inside 8 GB. `llm/train_lora.py` already uses LoRA,
gradient checkpointing, mixed precision and gradient accumulation with a small batch size. No
changes were needed in these areas beyond the `--force_cpu`/OOM fixes above.

---

## 2. Online research, retrieval, and crop/question relevance (tasks #3, #4)

### 2.1 `auto`/`always`/`never` research-decision logic — **verified, not changed**
**File:** `api.py` (`run_ask`)
`want_online = research == "always" or (research == "auto" and not local_ok)`, and `local_ok`
already requires non-empty passages, `not local["crop_missing"]`, and a minimum retrieval score —
i.e. it already goes beyond a bare score threshold by using the aspect/crop-relevance gate described
below. `research == "never"` already disables online research unconditionally. Online and local
results are merged into one `passages` list before generation, so they share the same
answer-generation/grounding pipeline (verified; see also §5).

### 2.2 Crop relevance: a crop named in a multi-crop list was treated as crop-specific evidence — **FIXED (correction to this report)**
**Files:** `sawvial/index/retriever.py` (`crop_mask`), `sawvial/index/passages.py` (`retrieve_passages`)
**This was originally signed off as "verified, not changed" in an earlier pass of this report — that
sign-off was wrong.** A live reproduction (asking "Which soil is suitable for tomato cultivation?")
returned a passage about ridge-and-furrow land configuration that only lists tomato among
"cotton, pearl millet, maize, chillies, tomato etc." as crops the layout suits — not a passage about
soil suitability for tomato at all.
**Root cause:** `crop_mask()`'s documented invariant is "mentions the crop AND at most one other
crop," but the code only enforced that check in an `elif` branch reached when the crop was found by
a raw text search. When the crop was instead already present in the record's `context.crop` labels —
which `sawvial.extraction.context_extractor` populates via plain vocabulary matching with **no**
primary-subject/passing-mention distinction, so almost every record that names a crop anywhere ends
up with it in `context.crop` — the code took an `if` branch that set the match to `True`
unconditionally, completely bypassing the multi-crop check:
```python
if crop in labels:
    mask[pos] = True                                    # no multi-crop check reached
elif pat.search(ku["knowledge"]["evidence"]):
    mask[pos] = len(text_crops[pos] - {crop}) <= 1       # the real check, but only reached here
```
`sawvial/index/passages.py`'s own independent crop-relevance filter had the identical shortcut:
`crop_labeled` (context-label match) alone was enough to pass the filter, with no check on how many
*other* distinct crops the passage also named.
**Fix:** both now always apply the "at most one other crop" check, regardless of which signal
(context label or text search) found the match:
```python
if crop in labels or pat.search(ku["knowledge"]["evidence"]):
    mask[pos] = len(text_crops[pos] - {crop}) <= 1
```
and in `passages.py`, a new `other_crops_mentioned` count (using the same `CROP_SYNONYMS`
vocabulary) is now required to be `<= 1` in addition to the existing `crop_labeled`/`crop_mentions`
checks. A record listing six crops in passing no longer qualifies as evidence for any single one of
them, in either code path.
**Why my first pass missed this:** I read `crop_mask`'s docstring and the `elif` branch's logic,
confirmed the multi-crop check existed, and concluded it was being applied — without tracing which
branch actually fires for a *typical* record (one where `context.crop` already contains the crop,
which is the common case, not the exception). An existing test,
`test_listing_a_crop_among_many_is_not_crop_evidence` in `tests/test_retrieval.py`, looked like it
covered this, but its fixture record is marked `needs_review` and therefore excluded from the index
by eligibility *before* `crop_mask` ever runs on it — so it never actually exercised the buggy branch.
I did not catch that gap without a live example to drive it.

### 2.3 Aspect filter matched "soil suitability" on passages that were actually about water, not soil — **FIXED (second correction, same live reproduction)**
**File:** `sawvial/index/passages.py` (`ASPECT_RULES`, now `_soil_aspect_evidence`)
**Root cause:** after fixing §2.2, the same live question ("Which soil is suitable for tomato
cultivation?") still returned a wrong single-source answer — this time a passage about protected
cultivation using **saline irrigation water** (record `SWM_003986`, primary domain
`water_management`, topic: salinity/water scarcity/polyhouses). It mentions only tomato + chilli, so
it correctly passed the now-fixed crop check, but it is not about soil at all: the only occurrence
of the string "soil" in its evidence is "soil**less** media" (a hydroponic growing medium — the
opposite of soil-based cultivation). It passed the "soil suitability" aspect filter anyway, because
that filter's evidence vocabulary included the bare words `saline`, `sodic`, `acid\w*`, `alkalin\w*`
with no requirement that they actually modify "soil" — and all four are at least as commonly used to
describe **water** quality ("saline water", "alkaline water") as soil condition ("saline soil"). The
passage's "groundwater is saline" / "saline water blending" is what matched.
**Fix:** split the aspect's evidence vocabulary into two tiers: unambiguous soil-only terms (loam,
clay, texture, pH, drainage, alluvial, laterite, "black/red soil", organic matter/carbon, etc.) still
match on their own; the four ambiguous terms (`saline`/`sodic`/`acid*`/`alkalin*`) now only count when
the word "soil(s)" appears within roughly 40 characters either side of them (`_soil_aspect_evidence`
in `passages.py`), so "saline soils reduced yield" and "the soil was found to be moderately saline"
both still match, but "groundwater is saline" / "saline water blending" do not. `aspect_filter()` was
generalized to accept this as a plain function alongside the other rules' compiled regexes (checked
via `hasattr(ev_rule, "search")`), so the two existing regex-based rules (water requirement,
fertilizer/nutrients) are completely unchanged.
**Verified by direct reproduction** (not just reading the code, after the §2.2 mistake): ran
`_soil_aspect_evidence` and the full `aspect_filter()` against the actual failing passage text and
confirmed it is now excluded, and re-ran it against the three scenarios in
`tests/test_retrieval.py`'s existing aspect tests (irrigation-only passage excluded, soil+irrigation
mix keeps only the soil one, a water-requirement question still keeps the irrigation passage) to
confirm no regression — all three still behave exactly as before.
**Why this took two passes to find:** these are two independent bugs in two different filters
(crop-specificity in §2.2, topic/aspect-matching here) that both needed to be wrong for this one
query to produce a bad answer; fixing only the first one was not enough, and only a second live
reproduction against the running app surfaced the second one. I do not have a way to replay the
full local index against every possible question without the application running, so further
latent issues of this kind (an aspect rule's vocabulary accidentally overlapping with an unrelated
topic) may still exist for aspects or crops not exercised by the questions tried so far.

### 2.3 Online research concurrency/safety (task #8's note on research files)
See §4 below (store.py / workflow.py / live.py) — these are filed under data-processing/research
quality but affect retrieval correctness (stale or lost records would otherwise be served to the
retriever).

### 2.4 `safe_http.py` — **verified, NOT modified, per explicit instruction**
Reviewed in full: scheme/port/credential checks, blocked internal hostnames, public-IP validation
repeated at both `validate_url()` and the connect-time guard (DNS-rebinding defence), manual bounded
redirect following with re-validation per hop, proxy env vars ignored, content-type allow-list, PDF
magic-byte check, streaming size cap, and an overall time budget are all intact. No changes made.

---

## 3. LLM answer generation, fallback, and grounding (tasks #5, #6)

### 3.1 Rigid "2 to 4 sentences, no more" length cap — **FIXED**
**File:** `llm/common.py` (`SYSTEM_PROMPT`)
**Root cause:** the system prompt hard-capped every answer at 2–4 sentences, which the task
explicitly flags as something to avoid ("Avoid rigid answer-length restrictions that reduce answer
quality") — a multi-part question, a comparison across studies, or an answer needing several
conditions/units could be truncated purely by this rule.
**Fix:** replaced the fixed sentence cap with an instruction to use as many sentences as the
question and evidence genuinely need, explicitly forbidding cutting a numeric value, unit, condition
or part of a multi-part question short just to stay brief. `settings.MAX_NEW_TOKENS` (300, a
generous ~200+ word budget appropriate for the 8 GB card) remains the only hard limit, and is a
resource budget, not an arbitrary stylistic rule.

### 3.2 No controlled regeneration after a failed grounding check — **FIXED**
**File:** `api.py` (`run_ask`), `llm/common.py` (`make_user_prompt`), `llm/rag_chat.py` (`answer`)
**Root cause:** task §5 explicitly specifies: (1) attempt one controlled regeneration using
validation feedback, (2) validate the regeneration, (3) only then say evidence was insufficient. The
existing code skipped straight from "no claim in the first answer was supported" to quoting raw
passage text (`extractive_answer`/`excerpt_answer`) — i.e. it never gave the model a chance to fix a
bad first attempt, and what it fell back to was source text dressed up inside the answer field.
**Fix:**
- `make_user_prompt()` gained an optional `feedback` parameter; when present, it tells the model
  exactly why its previous attempt failed (invalid citation ids, which supportedness checks failed)
  and asks for a corrected answer or an honest "not enough evidence" statement.
- `rag_chat.answer()` passes this through to the prompt.
- `api.py`: when the first answer has zero supported claims, it now builds that feedback from the
  grounding report and calls `rag.answer(...)` **once more** (guarded against cancellation and OOM
  the same way as the first attempt). The regenerated answer is re-assessed with
  `grounding.assess()`/`compose_supported_answer()`. Only if that retry *also* produces no supported
  text does the code fall through to the existing `extractive_answer`/`excerpt_answer`/`NO_DATA_ANSWER`
  chain (kept as a last-resort safety net rather than removed outright, since it is clearly labelled
  — "According to the retrieved research records:" / "The most relevant passage found says:" — and
  never claims to be a generated answer; removing it entirely seemed like unnecessary rewriting of a
  working safety net). The response now carries a `regenerated_after_grounding_failure` note so
  clients/evaluators can see when this path was taken.

### 3.3 Grounding/citation validation (`sawvial/grounding.py`) — **verified, not changed**
Already checks: cited ids actually exist in the supplied passages (`invalid_citations`), numeric
values in a claim appear in the cited evidence (`_supports`'s number check), absolute/overclaiming
words appear in the evidence vocabulary, and whether the evidence supports each sentence
individually (not just keyword overlap across the whole answer). `evidence_level()` flags
single-source answers. No bug was found here that the task's listed symptoms pointed to; the
regeneration flow in §3.2 now sits in front of it so a failing report is acted on before falling
back to excerpts.

---

## 4. Data processing and research quality (task #8)

### 4.1 Hyphenated words split across a PDF line-wrap were never rejoined — **FIXED**
**Files:** `sawvial/document_understanding/structure.py`, `sawvial/preprocessing/cleaner.py`
**Root cause:** `_dehyphenate()` in `cleaner.py` only rejoins a hyphen immediately followed by `\n`
(`"irriga-\ntion"` → `"irrigation"`), but `structure.py`'s `flush_buffer()` joined the paragraph's
lines with `" ".join(...)` before calling `clean_page_text()` — so by the time dehyphenation ran,
every line break was already a space, not a `\n`, and the regex could never match. Every
line-wrap-hyphenated word in the dataset was silently left broken (`"irriga- tion"`).
**Fix:** `structure.py` now joins the buffered lines with `"\n"` (so `_dehyphenate()` actually sees
line boundaries), and `clean_page_text()` flattens any remaining real line wraps (i.e. ones that
were *not* hyphenated, and so weren't rejoined) back into single spaces once dehyphenation has had
its chance — preserving the original flowing-paragraph output format for everything else.

### 4.2 Paragraphs silently glued across a page boundary — **FIXED**
**File:** `sawvial/document_understanding/structure.py`
**Root cause:** the vertical-gap-based paragraph-break heuristic resets `prev_bbox_bottom`/
`prev_line_height` to `None` at the start of every page (bounding-box coordinates aren't comparable
across pages), which means the heuristic is simply disabled for the first block of every new page —
so the last paragraph of a page and the first paragraph of the next were always merged into one
buffer unless a heading happened to fall between them, even when they were unrelated, and the
merged paragraph's `page` attribution reflected only the page the paragraph *started* on.
**Fix:** before resetting per-page state, if the buffered text already looks like it ended a
sentence (matches a sentence-terminal-punctuation regex) and doesn't end on a hyphen (which would
indicate a genuine hyphenation continuation, handled by §4.1), the buffer is flushed as its own
paragraph before the new page's blocks are processed. A buffer that looks mid-sentence is still left
to continue, which is the common and correct case for a paragraph that genuinely spans a page break.

### 4.3 Table content excluded from document identity/hashing — **FIXED**
**File:** `sawvial/ingestion/pdf_loader.py`
**Root cause:** `PageContent.plain_text` (and therefore `RawDocument.full_plain_text`, which feeds
`document_hash()` for `source_hash` and document-level dedup) was built only from text blocks,
never from `page.tables`. Two documents differing only in a data table (same narrative text, e.g. a
template report with different trial results) would hash identically.
**Fix:** `plain_text` now also serializes each table's rows (`" | "`-joined cells) into the page
text used for hashing/repeated-line detection. `structure.py`'s own paragraph/table handling is
unaffected — it reads `page.blocks`/`page.tables` directly and already skips blocks inside a table's
bounding box, so paragraph text is not duplicated.

### 4.4 Numeric extraction: overlapping/duplicated facts from p-values and confidence intervals — **FIXED**
**File:** `sawvial/extraction/numeric.py`
**Root cause:** `consumed_spans` (used to stop the generic range/measurement patterns from
re-matching a span another pattern already claimed) was only populated by the percent-effect and
range patterns. P-value and confidence-interval matches never marked their spans as consumed, so,
e.g., the "1.2–3.4" inside `"95% CI: 1.2-3.4"` was *also* independently extracted as a duplicate
"range" fact, and the "95" in "95% CI" could also be extracted a second time as a bogus "measurement"
fact of 95%.
**Fix:** reordered extraction so the more specific patterns (percent-effect, confidence interval,
p-value) run first and each now marks its own span consumed and skips if it overlaps an
already-consumed span; range and measurement extraction run last and skip anything already claimed.
Large/grouped numbers, decimals, negative values via the `[-+]?` sign in `_NUMBER`, and unit
handling were already correct and are unchanged.

### 4.5 Deduplication could merge genuinely different findings, and title-only document matching was unsafe — **FIXED**
**File:** `sawvial/dedup/deduplicator.py`
**Root cause (document dedup):** `document_dedup_key()` used a `title_author` key even when
`first_author` was empty (no author metadata), so two different papers sharing a short/generic title
with missing author info would collide and be wrongly treated as the same document.
**Fix:** the `title_author` key now requires a non-empty first author; otherwise the function falls
back to the full-document content hash, which cannot produce a false cross-document match.
**Root cause (near-duplicate knowledge units):** `deduplicate_knowledge_units()` only compared
domain + shingle (wording) similarity, so two passages with near-identical sentence templates but
different crops, doses, or outcomes (e.g. two fertilizer-rate trials written the same way with
different kg/ha numbers) could be merged away, losing a distinct scientific finding.
**Fix:** added a numeric fingerprint (all numbers mentioned in the evidence) and a crop fingerprint
(from `unit["context"]["crop"]`) to the near-duplicate check; a near-duplicate is now only dropped
when wording similarity is high **and** the numbers match **and** the crop(s) match.

### 4.6 `build_dataset.py validate` overwrote human review decisions — **FIXED**
**File:** `build_dataset.py` (`cmd_validate`)
**Root cause:** the command unconditionally set `ku["quality"]["validation_status"] = result.status`
for every unit from the automatic validator, including units already marked `"verified"` by a human
reviewer through the online-research publish workflow (`sawvial/research/workflow.py`
`publish_verified` → `sawvial/dataset/writer.py` `merge_into_dataset`, which preserves
`forced_status`). Re-running `validate` after publishing verified research would silently downgrade
those units back to an automatic status, discarding the human decision.
**Fix:** `cmd_validate` now skips re-validating any unit whose current `validation_status` is
`"verified"`, `"pending_review"` or `"published"`, or that carries `quality.published_as` — those
are preserved as-is. (`validate_knowledge_unit()` itself only ever returns `"validated"`,
`"needs_review"`, `"evidence_only"` or `"rejected"`, so any of the protected statuses above can only
have been set by a human/the publish transaction, never by this command.)

### 4.7 Online research deadline/cancellation/concurrency — **FIXED**
**Files:** `sawvial/research/store.py`, `sawvial/research/workflow.py`, `sawvial/research/live.py`
**Root cause (concurrency):** `ResearchStore`'s mutating methods (`add_records`, `set_status`,
`mark_published`, `add_source`) each did read-all-rows → mutate → write-all-rows with no locking.
`os.replace()` makes the *write* atomic, but not the read-modify-write cycle: two concurrent
requests that both trigger online research (e.g. two simultaneous `/api/ask` calls) could each read
the same snapshot and the second writer would silently clobber the first writer's additions — a
classic lost-update race, exactly what task #8 names under "concurrent JSONL updates."
**Fix:** added a process-wide `threading.Lock` per records/sources file, held around each full
read-modify-write cycle in all four mutating methods. This protects concurrent threads within one
process (how `api.py`/uvicorn runs); a true multi-process deployment would need an OS-level file
lock instead, which is called out as a known limitation below.
**Root cause (cancellation):** `cancel_event` was threaded through `live_research`'s own loop
(checked before starting the next candidate) but never into `ingest_url()` itself, so a cancelled
request still ran robots/throttle checks, the whole HTTP fetch, and document processing to
completion.
**Fix:** `ingest_url()` now accepts an optional `cancel_event` and checks it before each major step
(cached-lookup, robots/throttle, DOI/metadata lookup, and immediately before document processing),
returning a `"cancelled"` outcome as soon as it's noticed. A single in-flight socket read still
cannot be interrupted mid-byte — that's bounded by `safe_http`'s existing per-request timeout, which
was correct already and left untouched.
**Deadline handling:** `live.py`'s per-candidate deadline check (`time.monotonic()` against a budget)
was already correct and is unchanged; it is now backed by the cancellation propagation above so a
user-cancelled request also stops faster, not just a timed-out one.

---

## 5. Frontend response rendering (task #7)

### 5.1 Markdown never rendered — answers always showed raw `**bold**`, `# heading`, etc. — **FIXED**
**File:** `sawvial/js/chat.js`
**Root cause:** every place an assistant message's text reached the DOM used either
`escapeHtml(m.text)` (stored/replayed messages in `messageHtml()`) or `node.textContent = text`
(the live typing animation in `typeOut()`). No Markdown parser existed anywhere in the project (no
`marked`, no `DOMPurify`, nothing hand-rolled) — headings, bullet/numbered lists, bold text, links
and code formatting from the LLM's Markdown-formatted answers were never rendered; they showed up as
literal `#`, `**`, `-`, `` ` `` characters.
**Fix:** added a small, dependency-free `renderMarkdown()`/`renderInlineMarkdown()` pair directly in
`chat.js` (no new library/CDN dependency introduced). It escapes everything via the existing
`escapeHtml()` first and only then re-introduces safe tags for `#`–`######` headings, paragraphs,
`**bold**`, `` `code` ``, fenced ```` ``` ```` code blocks, `-`/`*` bullet lists, `1.` numbered lists,
and `[text](url)` links — links are restricted to `http(s)://` targets only (never
`javascript:`/`data:`), so nothing from the model can inject executable markup. `messageHtml()` now
calls `renderMarkdown(m.text)` instead of `escapeHtml(m.text)`. `typeOut()` still types the raw text
via `textContent` during the animation (safe by construction, no markup yet) and switches to
`node.innerHTML = renderMarkdown(finalText)` only once typing finishes or is stopped — so untrusted
LLM output is still never written into `innerHTML` directly, only the output of this escaping
renderer is.
**Preserved, verified unaffected by this change:** seven-language support (`m.lang` attribute still
set), chat history (raw Markdown text, not HTML, is what's stored in `localStorage` and re-rendered
through the same function), streaming/typing animation, request cancellation (`stop()` now also
routes through the renderer), source display, copy (`navigator.clipboard.writeText(m.text)` still
copies the raw Markdown text, unaffected), and speech (TTS still reads `m.text`, unaffected).

---

## 6. Documentation and configuration (task #10)

### 6.1 README overstated "everything runs on your own GPU" / implied fully offline — **FIXED**
**File:** `README.md`
**Root cause:** the intro paragraph said "No API key is used anywhere; everything runs on your own
GPU," immediately after describing the **online** research feature (OpenAlex/Crossref web fetches)
in the same paragraph — self-contradictory, and misleading about data privacy/offline operation.
Browser-based voice input (documented further down, correctly, as using the browser's online speech
recogniser) is a second place audio leaves the machine that the intro's blanket claim ignored.
**Fix:** reworded the intro to state plainly that the LLM/translator/local voices run on the user's
GPU, while SAWVIAL is explicitly **not** a fully offline/local-only system — it fetches from the open
web when local evidence is insufficient, and browser voice input uses an online speech service — and
to note that "live web" answers should be treated accordingly. No other GPU/model/offline claims
elsewhere in the README were found to be inaccurate; the GPU setup instructions, health-check
guidance, and CUDA-PyTorch install notes were checked and are consistent with the actual code.

---

## Known limitations (could not be fully verified without running the app)

- The `ResearchStore` locking fix (§4.7) is in-process only (a `threading.Lock`); it does not protect
  against two separate OS processes writing the same dataset directory at once. The project's
  existing design (a single `api.py`/uvicorn process) matches this, but it's worth stating
  explicitly since it wasn't rearchitected into a cross-process file lock.
- The page-boundary paragraph-break heuristic (§4.2) is still a heuristic: it uses sentence-final
  punctuation as a proxy for "this paragraph really ended," which will occasionally be wrong in
  either direction (e.g. a paragraph ending on an abbreviation, or a genuinely new paragraph that
  happens to start with a continuation-looking lowercase word). It is strictly better than the prior
  behaviour (which never broke across a page), but only a real PDF with print-outs or page-image
  inspection (which this report was told not to do) would confirm it on real documents.
- The CUDA OOM retry/fallback paths (§1.2) and the GPU-only code paths throughout (§1.5) are written
  to the project's existing conventions and were checked line-by-line for correctness, but were not
  executed — there is no way to runtime-confirm actual GPU behaviour without starting the
  application, which this task explicitly asked not to do.
- The Markdown renderer (§5.1) was syntax-checked with `node --check` and traced by hand against the
  supported syntax list; it was not exercised in a real browser DOM.
