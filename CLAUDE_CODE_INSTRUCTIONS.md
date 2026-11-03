# Claude Code Instructions

Work on the SAWVIAL repository: https://github.com/KUNALDEBNATH/Sawvial

Use the complete issue list and acceptance criteria in the existing repair specification:
https://github.com/KUNALDEBNATH/Sawvial/blob/codex/sawvial-end-to-end-fixes/CODEX_FIX_PROMPT.md

Treat that file as the technical specification, but execute it with Claude Code. Work on branch `claude/sawvial-end-to-end-fixes`, not `main`. First inspect git status and preserve all uncommitted user changes. Read any CLAUDE.md instructions. Then inspect the current source and implement the fixes in the specification; do not merely summarize or produce a plan.

Requirements:
1. Fix the complete PDF → dataset → index pipeline, local RAG, online research fallback, shared LLM generation, grounding, translation, frontend Markdown rendering and TTS flow.
2. Specifically fix the bug where irrelevant local RAG passages are treated as sufficient and automatic OpenAlex/Crossref research is skipped. In `auto`, search online when local evidence does not answer the query; `always` must search; `never` must not.
3. Ensure the LLM writes a natural answer in its own words from relevant evidence. Never silently display raw extracted text as a generated answer. Retry generation once after a grounding failure, then return an honest insufficient-evidence response.
4. Fix all other confirmed data-quality, deduplication, numeric extraction, PDF page-boundary, research cancellation/concurrency, offline/index, CPU/GPU training, TTS memory, multilingual evaluation and privacy-documentation issues listed in the linked specification.
5. Add regression tests. Run the full test suite, frontend checks and end-to-end evaluation where possible. Clearly report anything that could not be tested; never invent passing results.
6. Preserve SSRF protections, source trust checks, citations, provenance and human review before publishing online research. Do not retrain the LoRA model unless clearly necessary.
7. Create `FIX_REPORT.md`, update `ARCHITECTURE.md`, and create `RUN_WINDOWS.md` with exact PowerShell setup/run/test commands.
8. Produce `SAWVIAL_FIXED_SOURCE.zip`, excluding virtual environments, caches, temporary files and secrets. Include instructions for rebuilding any excluded index/model assets.
9. Commit all completed changes to `claude/sawvial-end-to-end-fixes`. Open a draft PR to `main`, but do not merge it.

Before finishing, verify the working tree, inspect the final diff, run tests, and ensure the ZIP is actually present and usable. Final response must include branch and PR links, report/architecture/runbook links, ZIP link, exact test results and remaining limitations. Do not claim completion until the code, tests, reports and ZIP are actually delivered.