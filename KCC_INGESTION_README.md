# What was fixed, and how your Excel/CSV data is now wired in

## Update: two more bugs found & fixed after testing the banana question
Testing "What type of soil is best for banana?" surfaced two bugs in the
*ingestion script itself* (not the retrieval/grounding code):
1. **The question was echoed back in the answer.** The first version of this
   script stored each record's evidence as `"{question}. {answer}"`. When the
   LLM's own generated answer failed the grounding check, the app's
   extractive-fallback (`sawvial.grounding.extractive_answer` -- a deliberate
   anti-hallucination safety net, not a bug) quotes whichever evidence
   sentence has the most word-overlap with your question -- and the
   evidence's first "sentence" literally *was* your question. **Fixed:**
   evidence is now the KCC answer only; the question is preserved separately
   as a verbatim Q/A pair, never inside the quotable evidence text.
2. **A garbled CSV row ("organci soils ,fertile soils ,ondru nelalu etc.,")
   outranked a clean one.** That row is genuinely broken source data (typos,
   part-romanized-Tamil fragment, no real grammar) -- TF-IDF ranked it highly
   purely on keyword overlap, and being that garbled is *why* the LLM
   couldn't produce a groundable sentence from it, triggering the raw-quote
   fallback in the first place. **Fixed:** a new filter
   (`_looks_like_garbled_fragment_list`) rejects verbless, comma-joined
   fragment answers at ingestion time, before they ever reach the index.

Re-running ingestion on the full CSV with both fixes: candidate rows dropped
from 11,249 to 4,461 (the garbled ones are gone), and 462 pass full
validation and are indexed (fewer than the 1,111 from the first pass, but
clean). Re-verified both the tomato and banana questions directly against
`sawvial.index.passages.retrieve_passages` -- no echoed question, no garbled
text, correct passage ranked first for both.


## The bug you hit
"Which soil is suitable for tomato cultivation?" failed because your **local
knowledge base simply had no record that actually said what soil suits
tomato** — the crop/aspect filtering bugs from the earlier fix pass are
already fixed in this code (`crop_mask`, `_soil_aspect_evidence`); I verified
both fixes are still present. The remaining problem was a data gap, not a
retrieval bug: nothing in your indexed PDFs stated tomato's soil preference,
and the live-web research step couldn't find/ground a usable passage either.

## The fix: your KCC CSV is now real, queryable knowledge
`kcc_index_ready_final.csv` (Kisan Call Centre farmer Q&A) **does** contain
the answer:

> *"farmer asked query on suitable lands for tomato?"* → *"Tomatoes can grow
> in a variety of soils, but they grow best in well-drained, light loam soil
> with a pH of 5 to 7."*

I built `tools/ingest_kcc_csv.py`, a new ingestion script that feeds this CSV
through **the exact same pipeline** your PDFs go through — no shortcuts, no
hand-written answers:

- `sawvial.classification.domain_classifier` assigns the real domain
  (soil_management / water_management / ...),
- `sawvial.extraction.context_extractor` tags crop/soil-type/etc.,
- `sawvial.extraction.importance` scores it,
- `sawvial.knowledge.knowledge_builder` builds the canonical record
  (evidence is the verbatim KCC text — nothing paraphrased or invented),
- `sawvial.qa.qa_generator` adds grounded Q&A,
- `sawvial.validation.validator` assigns `validated` / `needs_review` /
  `evidence_only` / `rejected` — same rules as everywhere else,
- `sawvial.dataset.writer.merge_into_dataset` appends it to your dataset
  **without touching or duplicating what's already there**,
- `sawvial.index.faiss_builder` rebuilds the lexical index.

So a KCC-derived answer is governed by the identical `is_answerable` /
grounding gate as everything else — it can't bypass the "never hallucinate"
checks the rest of the app relies on.

Of ~493k CSV rows, after filtering to the ones actually usable as evidence
(non-empty English Q+A, decent length, not a scheme/weather/price query,
dedup'd): **11,249 records** were built, of which **1,111** passed full
validation and are now indexed (soil_management: 807, water_management: 302,
soil_water_conservation: 2). The rest are kept as `evidence_only` /
`needs_review` for manual triage, not silently discarded.

**Verified fix** — I ran the real retrieval code
(`sawvial.index.passages.retrieve_passages`, exactly what `api.py` calls) for
your failing question and it now returns:

```
SWM_003753  Tomato — farmer asked query on suitable lands for tomato?
"Tomatoes can grow in a variety of soils, but they grow best in
well-drained, light loam soil with a pH of 5 to 7."
```

plus clean crop-filtered results for other test questions (wheat irrigation
timing, banana soil, tomato wilt disease, etc.).

## ⚠️ Important — merge this into your real dataset, don't just drop it in
The zip you gave me did **not** include your real `dataset/knowledge_units.jsonl`
(the ~5,568-unit file your PDFs built — it was presumably stripped out before
zipping, since it's large). So the `dataset/` and `index/` folders in *this*
zip only contain the 1,111 KCC-derived units, not your PDF knowledge.

**Do this on your own machine instead of copying my dataset/index folders
over yours:**

```
.\.venv\Scripts\python.exe tools\ingest_kcc_csv.py --csv kcc_index_ready_final.csv --build-index
```

Run it from inside your real project folder (the one with your real
`dataset/knowledge_units.jsonl`). `merge_into_dataset` de-duplicates against
what's already there (by document/evidence hash), assigns continuing IDs, and
`--build-index` rebuilds `index/` from the combined set — so you keep your
existing PDF-derived knowledge **and** gain the 1,111 KCC records. No GPU
needed for this step (it's a pure lexical TF-IDF rebuild); your LLM/GPU is
only used later, at answer-generation time, same as before.

Useful flags:
- `--dry-run` — see row/domain counts without writing anything.
- `--min_latin_fraction` (default 0.8) — most of the CSV is Hindi; the
  taxonomy vocabulary is English, so non-Latin rows are skipped by default
  rather than being mis-classified. Translate and re-run to bring the Hindi
  answers in once you have a translation step you trust.
- `--min_importance` / `--review_importance` — same thresholds as
  `build_dataset.py`, applied identically.

You can re-run the script with new/updated CSV exports any time — it only
ever adds, never overwrites.
