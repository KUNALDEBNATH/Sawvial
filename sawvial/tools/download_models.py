#!/usr/bin/env python3
"""One-time download of the model weights SAWVIAL needs (they are NOT in the git repository).

    python tools/download_models.py                 # base LLM + translator + all speech models
    python tools/download_models.py --only base     # just the LLM (English-only use)
    python tools/download_models.py --speech        # just the local text-to-speech models

Files go into the normal Hugging Face cache (~/.cache/huggingface). After this, SAWVIAL runs with
no internet access (SAWVIAL_OFFLINE=1, the default). Needs internet ONCE and roughly 4-5 GB of disk.
Licences: NLLB-200 is CC-BY-NC 4.0 (non-commercial); check the Qwen and MMS-TTS licences before
commercial use.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm"))
from sawvial import settings
from sawvial.model_config import inspect_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["base", "translator", "speech"])
    ap.add_argument("--speech", action="store_true", help="shortcut for --only speech")
    a = ap.parse_args()
    only = "speech" if a.speech else a.only
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is missing. Run the install lines from README.md first (CUDA PyTorch, then requirements-llm.txt).")
    from speech import MODEL_IDS

    jobs = []
    if only in (None, "base"):
        jobs.append(("base LLM", settings.BASE_MODEL, ["*.json", "*.safetensors", "merges.txt", "*.txt", "*.jinja"]))
    if only in (None, "translator"):
        jobs.append(("translator", settings.TRANSLATOR_MODEL, None))
    if only in (None, "speech"):
        jobs += [(f"speech [{code}]", mid, None) for code, mid in MODEL_IDS.items()]
    failed = []
    for label, repo, patterns in jobs:
        print(f"-> {label}: {repo}")
        try:
            snapshot_download(repo, allow_patterns=patterns)
        except Exception as exc:
            print(f"   FAILED: {exc}")
            failed.append(repo)
    chk = inspect_adapter(settings.ADAPTER_DIR, settings.BASE_MODEL)
    from sawvial.model_config import hf_cached
    print(f"\ntranslator cached: {hf_cached(settings.TRANSLATOR_MODEL)}   " +
          "   ".join(f"voice[{c}]: {hf_cached(m)}" for c, m in MODEL_IDS.items()))
    print(f"\nLoRA adapter ({settings.ADAPTER_DIR}): {chk.status}" + ("" if chk.ok else f" - {chk.message()}"))
    if failed:
        sys.exit(f"\nDownload failed for: {', '.join(failed)}. Check your internet connection and retry.")
    print("\nAll requested models are cached.")


if __name__ == "__main__":
    main()
