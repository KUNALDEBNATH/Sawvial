# SAWVIAL - multilingual, voice-enabled assistant for soil & water management

SAWVIAL answers farmers' questions about soil, irrigation and water use **from a knowledge base of
agricultural research papers**, shows the sources under each answer, and says so when the evidence is missing.

> **Not scientifically validated.** Records marked `validated` passed *automatic* checks only. Passing the
> automated tests does not mean the advice is agronomically correct. Have an agronomist review answers
> before anyone relies on them.

## How it works

```
question (typed, or voice -> browser speech recognition)
  -> [translate to English if needed]            NLLB-200, local
  -> retrieve answerable records                 TF-IDF index (+ optional dense index = hybrid)
       crop named?  only records genuinely about that crop; none -> "not enough evidence" (LLM is not called)
  -> generate answer                             Qwen2.5-1.5B + LoRA adapter, local
  -> grounding check                             every sentence needs a REAL citation + supporting text + matching numbers
       unsupported sentences are removed; nothing left -> quoted evidence or "insufficient evidence"
  -> [translate answer back]  -> answer + sources (title, authors, year, journal, page, DOI/URL links)
```

Document pipeline: `PDF -> text/tables/metadata -> structured sections -> knowledge records -> validation -> index`.

### Record status (one meaning everywhere)

| status | meaning | can support an answer? |
|---|---|---|
| `validated` | passed automatic checks, **not human-reviewed** | yes |
| `verified` | approved by a named human reviewer (research workflow) | yes |
| `needs_review` | borderline score or incomplete extraction (e.g. scanned/partial PDF) | no (default) |
| `evidence_only` | solid text, no domain assigned | no |
| `pending_review` | newly downloaded online record, awaiting a human | no |
| `rejected` | failed checks / rejected by reviewer | no |

Only `validated` + `verified` records are **put into the index at all**, so unsuitable records can never use up
top-k slots. Change with `SAWVIAL_ANSWERABLE_STATUSES` (adding `needs_review` lowers reliability).

## Requirements

* Python **3.10-3.12**, Windows 10/11 (Linux/macOS work too).
* ~5 GB disk for models, a few GB RAM. An NVIDIA GPU is strongly recommended for answer speed (CPU works, slowly).
* Optional: **Tesseract OCR** for scanned PDFs; **Node.js** only to run the optional frontend DOM tests.

## One-time setup (Windows PowerShell)

```powershell
# 1. Get the code
git clone https://github.com/KUNALDEBNATH/Sawvial.git
cd Sawvial

# 2. Create and activate a virtual environment (keeps packages separate from other projects)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
#    If PowerShell blocks the script:  Set-ExecutionPolicy -Scope Process Bypass   (then run the line above again)

# 3. Install packages
#    NVIDIA GPU? install the CUDA build of PyTorch FIRST (otherwise you get the slow CPU build):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-llm.txt

# 4. Download the model weights ONCE (they are not in git; needs internet + ~5 GB).
#    After this SAWVIAL runs fully offline (except browser voice input, see below).
python tools/download_models.py

# 5. Clean the dataset's QA pairs and recompute its statistics (recommended, ~10 s).
#    Reads/writes dataset/knowledge_units.jsonl, qa.jsonl, dataset_statistics.json; log in dataset/qa_rejected.jsonl.
python tools/audit_dataset.py --fix

# 6. Build the search index (required once after cloning: the index files in git are the OLD format). ~10-20 s.
python build_dataset.py --skip-build --build-index
#    (optional) also delete leftovers of the old, never-used index:   git rm index/faiss.index index/vectors.npy
```

The LoRA adapter (`llm/sawvial-lora`) is part of the repository. At startup SAWVIAL checks it matches the base
model (name, hidden size, layer count). **It is loaded by default.** A missing adapter prints a loud warning and runs the
base model (visible at `/api/health`); an incompatible adapter is a hard error. Use `--base_only` to deliberately skip it
and `--require_adapter` to fail instead of falling back.

## Every time you start

```powershell
cd Sawvial
.\.venv\Scripts\Activate.ps1
python api.py                  # loads models once (1-2 min), then serves API + web pages
```

Open **http://localhost:8000**. The same process serves the frontend - there is no separate frontend server.
Other start options:

```powershell
python api.py --mock           # canned answers, instant start - for testing the web page only
python api.py --no_translate   # English only, skips the translation model
python api.py --base_only      # run WITHOUT the LoRA adapter (explicit)
python api.py --host 0.0.0.0   # reachable from other machines: read "Security" first
```

## Using it

* **Text**: type a question, press Enter. Pick the language on the language page (English, Hindi, Bengali, Tamil, Telugu, Kannada, Marathi).
* **Voice input**: tap the microphone and speak. Uses the *browser's* speech recognition (Chrome / Edge; Firefox has none) which
  **sends audio to the browser vendor's online service**, so it needs internet and is not offline. If recognition fails you get a
  specific message (blocked microphone, nothing heard, offline, unsupported language) and can retry or type. SAWVIAL never submits
  a substitute question.
* **Listen**: the speaker button reads the answer in the language the answer is actually written in. A local voice (MMS-TTS, fetched
  by `download_models.py --speech`) is used for languages whose model is cached on the server (`/api/health` -> `speech_local`);
  otherwise the **browser's own voice** is used, if your device has a voice for that language (otherwise you get a message).
  Voices are single robotic voices - no regional accents - and Indian-language browser voices vary by device. Hindi/Tamil/Telugu/Kannada
  local voices are unverified here (see Known limitations).
* **Sources** appear under each answer with authors, year, journal, page and clickable DOI / URL **only when the record has them**;
  otherwise "no online link available". Metadata is never invented. Notes tell you when only one source supports an answer, when
  unsupported parts were removed, or when the text is quoted directly.
* If the request fails you get a **Try again** button. Rate limit: 30 requests/min per IP (configurable).

## Add a PDF

```powershell
# adds the PDF to the existing dataset (no full rebuild; existing IDs unchanged; duplicates skipped) and refreshes the index safely
python build_dataset.py --add_pdf "C:\path\to\paper.pdf" --build-index
```

A report is printed per file: `ok`, `partial` (incomplete text or table errors; records become `needs_review`, never auto-validated),
`no_text` (probably scanned; **nothing is added**), or `failed`. A full rebuild from `papers/` writes `dataset/ingestion_report.json`
listing every problematic PDF: `python build_dataset.py --rebuild --build-index` (slow; renumbers IDs).

### OCR for scanned PDFs (optional)

1. Install Tesseract (Windows installer from the UB Mannheim builds) and tick the languages you need; add its folder to `PATH`.
   If it complains about language data set `$env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"`.
2. `python build_dataset.py --add_pdf "scan.pdf" --ocr --ocr_language eng --build-index`

OCR errors are recorded in the report, not hidden. OCR text is lower quality (tables especially) - review such records.

## Online research (optional)

```powershell
python research_cli.py search "effect of drip irrigation on groundnut yield"   # find, download, extract, compare
python research_cli.py url "https://example.org/paper.pdf"                     # ingest one specific document
python research_cli.py list --status pending_review
python research_cli.py show SWR_000001
python research_cli.py verify SWR_000001 SWR_000002 --reviewer "Your Name"     # YOU read the evidence first
python research_cli.py reject SWR_000003 --reviewer "Your Name" --note "wrong crop"
python research_cli.py publish                                                 # verified -> dataset, index rebuilt safely
```

* **Search** uses the key-less scholarly APIs **OpenAlex** (default) or **Crossref** (`SAWVIAL_SEARCH_PROVIDER`). They index papers; they are
  **not a general web search**. Only open-access/accessible documents can be downloaded; paywalled ones are reported as failed.
  If the search or a download fails, the command says exactly that and stores nothing. `none` disables search.
* Everything downloaded is stored as **`pending_review`** with its source URL, DOI, retrieval time and file hash; the original file is kept in
  `research_store/downloads/`. The printed comparison is **extractive** (quoted source text, per source), flags *possible* contradictions
  (opposite effect directions on the same parameter - a person must check crops/conditions) and lists evidence gaps. It is not saved as
  knowledge. Generated answers are never converted into verified records.
* A record becomes answerable only after a named human runs `verify` **and** `publish`. Records containing instruction-like text
  ("ignore previous instructions...") are flagged and cannot be verified. Downloaded text only ever reaches the model as quoted evidence.
* Safety: http/https only, public addresses only (loopback/private/link-local/metadata addresses refused, re-checked at connect time and on
  every redirect, max 3), 20 s timeout, 25 MB limit, allowed content types (PDF/HTML/text), PDF magic-byte check, environment proxies ignored.
  Optionally restrict to domains: `SAWVIAL_RESEARCH_ALLOWED_DOMAINS=icar.org.in,fao.org`. ICAR and publisher domains are ranked first.
* This command-line workflow is the only way to change the knowledge base; the web API deliberately cannot trigger downloads.

## Evaluation and tests

```powershell
pip install -r requirements-dev.txt
python -m pytest tests -q                      # 100+ automated tests; no models, GPU or internet needed
# optional real-browser-behaviour tests (mic failure, retry, safe links) in a DOM simulator:
npm install jsdom ; python -m pytest tests/test_frontend.py -q

python evaluate_rag.py                         # retrieval on HELD-OUT documents + crop + unsupported-crop checks (no LLM)
python evaluate_rag.py --api http://localhost:8000                  # end-to-end through the running server: citations, refusals, latency
python evaluate_rag.py --api http://localhost:8000 --multilingual   # also hi/bn (needs the translator loaded)
```

Evaluation separates **retrieval** (recall/MRR/nDCG, document hit rate) from **citation behaviour** (invalid citations, supported-claim
coverage, partially-supported/extractive-fallback counts) and **latency** (mean/p50/p95 excluding the first request, plus per-stage means).
Held-out cases come only from documents excluded from fine-tuning data (`llm/common.py: is_heldout_doc`, same split as `prepare_sft_data.py`).
Results are written to `eval_results/`. The old `evaluation-results.legacy.json` is kept only as history: it used 7 hand-picked cases, hard-coded IDs
and was not separated from training data.

### Retraining the adapter (optional, needs a GPU)

The shipped adapter was trained on the *old* data (73% of its non-refusal training answers were mid-sentence fragments). To retrain on the cleaned data:

```powershell
python llm/prepare_sft_data.py          # writes llm/sft/train.jsonl + val.jsonl (overwrites)
python llm/train_lora.py --4bit         # writes llm/sawvial-lora
```

Keep the default `--val_pct 5` for both scripts so the held-out evaluation split stays valid.

## Configuration

Everything has a working default. Optional overrides are environment variables - see `.env.example` (no secrets exist in this project).
PowerShell example: `$env:SAWVIAL_OFFLINE = "0"; python api.py`.

## Security notes

* The server listens on `127.0.0.1` by default. CORS allows only `http://localhost:8000`/`127.0.0.1:8000` by default (`SAWVIAL_CORS_ORIGINS`).
  Before exposing it publicly: set your real https origin, put it behind HTTPS and a reverse proxy/firewall, and use the proxy's own rate limiting
  (the built-in limiter keys on the connecting IP, which behind a proxy is the proxy). There is no user authentication.
* Inputs are length-limited and validated; external links are only rendered for http(s) URLs and valid DOIs, with `rel="noopener noreferrer"`.
* Index/vectorizer files are loaded with `joblib` (pickle): only use index files you built yourself.

## Common errors

| message | fix |
|---|---|
| `No index at ... Build it with: python build_dataset.py --skip-build --build-index` / `old format` | run step 6 above |
| `Base model ... is not in the local cache` | `python tools/download_models.py` (needs internet once) |
| `Adapter ... does not match base model` | set `SAWVIAL_BASE_MODEL` to the model named in the message, or start with `--base_only` |
| `ADAPTER NOT FOUND ... running the BASE MODEL ONLY` | restore `llm/sawvial-lora/` (it is in git), or use `--base_only` knowingly, or `--require_adapter` to fail |
| `Model libraries are missing (No module named 'torch')` | `pip install -r requirements-llm.txt` (CUDA torch first, see setup) |
| log: `knowledge_units.jsonl changed since the index was built` | re-run step 6 (new/changed records are not searchable until then) |
| translation unavailable note / Hindi answered in English | translator model not cached: `python tools/download_models.py --only translator`; or start with `--no_translate` |
| `No local voice model for 'xx'` | `python tools/download_models.py --speech`; otherwise the browser voice is used |
| voice input does nothing / says unsupported | use Chrome or Edge, allow the microphone, check internet; or type |
| `429 Too many requests` | wait a minute or raise `SAWVIAL_RATE_LIMIT_PER_MIN` |
| `SEARCH FAILED: ... cannot resolve host` | no internet or provider blocked; use `research_cli.py url <URL>` or another `SAWVIAL_SEARCH_PROVIDER` |
| PDF reported `no_text` | scanned PDF: install Tesseract and add `--ocr` |
| PowerShell: running scripts is disabled | `Set-ExecutionPolicy -Scope Process Bypass` |

## Known limitations (honest list)

* Grounding is a **lexical** check (valid citation, shared content words, matching numbers). It catches fabricated sources, uncited claims and invented numbers; it cannot prove a paraphrase is scientifically right.
* Retrieval is TF-IDF with agricultural synonym expansion (paddy->rice, WUE, FYM ...). A dense/hybrid index is implemented (`--dense`) but **not evaluated** here. On the held-out set recall@5 is modest (see `eval_results/`); many auto-generated questions are generic templates.
* Crop gating knows ~35 crops; a crop outside that list is treated as a general question.
* Metadata in the existing dataset is incomplete (most records lack authors, journal and URL; DOIs exist for roughly a third). Fixing that needs re-extraction with better metadata sources; a rebuild applies the improved title/author filters.
* The shipped LoRA adapter predates the data cleaning (see retraining). Latency of the real LLM was **not measured** in the audit environment (no GPU/models); stage timings are logged and returned as `timings_ms`.
* Voice input requires an online browser service. Local speech output for hi/ta/te/kn uses `facebook/mms-tts-*` checkpoints that were not downloaded or listened to during the audit.
* NLLB-200 (translation) is licensed CC-BY-NC 4.0 (non-commercial).

## Project layout (what is where)

```
api.py                 FastAPI server + answer pipeline        research_cli.py   online research CLI
build_dataset.py       build / add-PDF / index CLI             evaluate_rag.py   evaluation
llm/                   rag_chat.py (model loading, generation), speech.py (TTS), SFT + LoRA training, common.py
sawvial/               settings, eligibility, grounding, model_config, index/ (builder, retriever, passages),
                       ingestion/ document_understanding/ chunking/ extraction/ validation/ qa/ research/ + frontend (html, js/, css/)
tools/                 download_models.py, audit_dataset.py
tests/                 pytest suite (+ tests/frontend/dom_test.js, tests/eval_cases.jsonl)
dataset/  index/  research_store/   data (index is derived and rebuildable)
```
