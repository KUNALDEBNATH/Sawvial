# SAWVIAL - multilingual, voice-enabled assistant for soil & water management

Ask a question about crops, soil, irrigation or water in English, Hindi, Bengali, Tamil, Telugu, Kannada or Marathi (typed or spoken).
SAWVIAL searches its local research knowledge base first; if that cannot answer, it **searches online** (OpenAlex / Crossref, no API key),
reads the open-access papers and pages, and writes an evidence-based answer **in the language you selected**, with clickable sources and
a spoken version. It says so honestly when it cannot find evidence. No API key is used anywhere; everything runs on your own GPU.

> Not scientifically validated: answers come from automatic checks, and live web evidence is shown as *not reviewed by an expert*.
> Have an agronomist review the advice.

## Run it (Windows, Command Prompt or PowerShell) - everything you need, in this order

Needs: Python 3.10-3.12, an NVIDIA GPU with a current driver (yours: RTX 4060, 8 GB), internet for the first run only (~6 GB of model downloads).
Open a terminal **inside the unzipped `Sawvial` folder** and run:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-llm.txt
.\.venv\Scripts\python.exe tools\download_models.py
.\.venv\Scripts\python.exe api.py
```

Then open **http://localhost:8000** (press Ctrl+F5 once so the browser loads the new page code).

* The first four lines and the model download are needed **once**. Next time, only the last line: `.\.venv\Scripts\python.exe api.py` (from the same folder).
* Line 3 installs the **CUDA** build of PyTorch. SAWVIAL uses the GPU and **refuses to start on the CPU** by accident; if PyTorch cannot see your GPU it stops
  and prints the exact fix. The LLM (about 3 GB), the translator and the Tamil voice all run on the GPU and fit in 8 GB.
* The search index is built or repaired automatically at start-up (about 15 seconds when needed). Nothing else to run.

Check that it is really using the GPU and everything loaded (a second terminal, while the server runs):

```
curl http://localhost:8000/api/health
```
Look for `"device"` (your GPU name), `"translator": true`, `"speech_local"` (contains `"ta"` for Tamil), `"index"` with `"stale": false`.
If `translator` is false, `translator_error` contains the exact fix.

## How an answer is produced

1. **Language** - the question's language is detected from its script; the answer language is the one you selected. They are independent
   (an English question can get a Tamil answer; a Tamil question a Tamil answer). If translation is unavailable the answer stays English, is labelled English,
   and a visible banner says so.
2. **Local first** - the knowledge base must have evidence that is about the right crop **and** the right topic (a tomato *irrigation* passage does not answer a tomato *soil* question).
3. **Online research** - otherwise it searches, opens only sources it may read (public, robots.txt-respecting, trusted by their **final** site, not just the link they came from), and extracts the relevant passages.
4. **Answer** - written from those passages and checked: every sentence needs a real citation, supporting text, matching numbers and no unsupported overclaims; unsupported sentences are removed.
   Sources are labelled *official guidance / research paper / general article* and *local / live web*; differing values between sources are shown.
5. **Listen** - the answer is spoken in the language it is written in: local voice (Tamil: `facebook/mms-tts-tam`) -> your browser's voice -> a clear message.
6. Evidence found online is saved in `research_store/` as **pending review** and never becomes permanent knowledge without a human decision.

## Honest limits

* Search is key-less scholarly search (OpenAlex, Crossref): good for papers and open-access documents, **not a general web search** - extension pages
  (e.g. TNAU, Vikaspedia) are found only when a paper links to them. Web pages not in English are skipped.
* Grounding is a lexical check: it catches fabricated sources, invented numbers and overclaims; it cannot prove that a paraphrase is scientifically right.
* The shipped LoRA adapter was trained before the dataset clean-up.
* Voice input uses the browser's online speech recogniser (Chrome / Edge). Romanised text ("thakkali mann") is treated as English.
* The model download, GPU speed, Tamil translation/voice quality and live online search have **not been verified by me** (no GPU/internet where this was built) - please try them and tell me what you see.

## Problems

| you see | do |
|---|---|
| `No usable GPU: ... CPU-only build` | run the 3rd line of the snippet again (it reinstalls CUDA PyTorch), then start again |
| `Base model ... not in the local cache` / `Translation model ... not in the local cache` | run the `tools\download_models.py` line |
| banner "answer could not be translated" | `curl http://localhost:8000/api/health` -> read `translator_error` |
| answer says the online search could not be completed | no internet/blocked; the reason is in the message |
| page looks old after an update | stop the server (Ctrl+C), start it again, press Ctrl+F5 |

Maintenance (add a PDF, review and publish collected research, tests, evaluation, retraining): see [docs/MAINTENANCE.md](docs/MAINTENANCE.md).
