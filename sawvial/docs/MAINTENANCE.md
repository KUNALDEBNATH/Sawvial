# Maintenance (none of this is needed to run SAWVIAL)

All commands run from the project folder with `.\.venv\Scripts\python.exe` (works in Command Prompt and PowerShell).

## Add a PDF to the knowledge base
```
.\.venv\Scripts\python.exe build_dataset.py --add_pdf "C:\path\paper.pdf" --build-index
```
Scanned PDFs report `no_text`; install Tesseract and add `--ocr --ocr_language eng` to read them.

## Review and publish online research
The chat stores what it reads in `research_store/` as `pending_review`. To make records permanent knowledge, a person must verify them:
```
.\.venv\Scripts\python.exe research_cli.py list --status pending_review
.\.venv\Scripts\python.exe research_cli.py show SWR_000001
.\.venv\Scripts\python.exe research_cli.py verify SWR_000001 --reviewer "Your Name"
.\.venv\Scripts\python.exe research_cli.py reject SWR_000002 --reviewer "Your Name" --note "wrong crop"
.\.venv\Scripts\python.exe research_cli.py publish
```
`publish` is a transaction: dataset snapshot -> merge -> index rebuild -> only then records are marked published. On failure the dataset is restored,
the previous index is kept and the records stay `verified`/unpublished. `research_cli.py search "<question>"` runs the same research as the chat and prints a report.

## Tests and evaluation
```
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe evaluate_rag.py                              # retrieval on held-out documents (no LLM)
.\.venv\Scripts\python.exe evaluate_rag.py --api http://localhost:8000   # end to end, server must be running
```
The browser-behaviour tests need Node.js and `npm install jsdom`; they are skipped otherwise.

## Settings
Defaults are fine. `.env.example` lists every setting (copy to `.env` to change one). Notable: `SAWVIAL_ONLINE_RESEARCH=0` = local knowledge base only;
`SAWVIAL_RESEARCH_MIN_TRUST` = `official` | `reputable` | `any`; `SAWVIAL_DEVICE=auto|cpu` to allow CPU.

## Record statuses
`validated` (automatic checks, not human-reviewed) and `verified` (human-approved) can support answers; `pending_review` (online, awaiting a person) only supports
the current live answer; `needs_review`, `evidence_only`, `rejected` never do.

## Retraining the adapter (GPU)
```
.\.venv\Scripts\python.exe llm\prepare_sft_data.py
.\.venv\Scripts\python.exe -m pip install bitsandbytes
.\.venv\Scripts\python.exe llm\train_lora.py --4bit
```
Keep the default `--val_pct 5` in both scripts so the held-out evaluation split stays valid.
