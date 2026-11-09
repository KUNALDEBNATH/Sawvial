# GPU Configuration — NVIDIA GeForce RTX 4060 (8 GB)

This describes how SAWVIAL uses your RTX 4060 after this fix pass. It documents what the code
*does*; it was checked statically (code inspection), not by running the application, per this
task's instructions.

## Driver / CUDA version note

`nvidia-smi` reporting CUDA 13.1 is the **maximum CUDA version your driver (591.59) supports** — it
is not necessarily the CUDA version your installed PyTorch build was compiled against, and the code
never assumes otherwise. `sawvial/device.py` and the model-loading code never read or branch on the
`nvidia-smi` CUDA version at all; they only ever ask PyTorch itself, at runtime, whether a GPU is
usable (`torch.cuda.is_available()`) and what it supports (`torch.cuda.is_bf16_supported()`,
`torch.cuda.get_device_properties(0)`). Whatever CUDA build of PyTorch you install (11.8, 12.1, 12.4,
etc., per the README's setup snippet) will work as long as it's new enough for an Ada Lovelace card
like the 4060 and no newer than what driver 591.59 supports — that compatibility is PyTorch/driver's
job, not something this codebase checks or needs to.

## Central device selection: `sawvial/device.py`

`pick_device(torch_module=None)` is the single source of truth for device selection:
- Returns `{"device": "cpu"}` immediately if `SAWVIAL_DEVICE=cpu` is set (explicit CPU opt-out).
- Otherwise calls `torch.cuda.is_available()`; if true, reads the real device name and VRAM via
  `torch.cuda.get_device_properties(0)` (on your machine: "NVIDIA GeForce RTX 4060", ~8188 MB) and
  returns `{"device": "cuda", ...}`.
- If a GPU is required (the default) but unavailable, raises a clear `ModelConfigError` rather than
  silently falling back to CPU — this is the "refuses to start on the CPU by accident" behaviour the
  README describes.
- Never hardcodes a device index beyond `"cuda"` (device 0, the only GPU on a single-GPU machine);
  nothing in the project assumes a specific device ordinal beyond that.

Every GPU-capable component below goes through this function (or, for `llm/train_lora.py`, the
equivalent `torch.cuda.is_available()` check combined with the new `--force_cpu`-respecting logic
from this fix pass).

## Per-workload configuration

| Workload | File | Device selection | Precision | Memory notes |
|---|---|---|---|---|
| LLM inference (chat/API) | `llm/rag_chat.py` `load_llm()` | `pick_device(torch)` | `bfloat16` if `torch.cuda.is_bf16_supported()` else `float16` on GPU, `float32` on CPU | Qwen2.5-1.5B is ~3 GB in 16-bit — comfortable headroom in 8 GB. `--4bit` (QLoRA-style `BitsAndBytesConfig`) is available for a smaller footprint if you ever swap to a larger base model; it now refuses to combine with `--force_cpu`. LoRA adapter is merged into the base weights (`merge_and_unload()`) when not running 4-bit, removing per-token adapter overhead. |
| LLM generation (per request) | `llm/rag_chat.py` `answer()` | uses the already-loaded model's device | same as load | **New in this pass:** catches `torch.cuda.OutOfMemoryError`, frees the cache, and retries once with half `max_new_tokens`; `api.py` catches a second OOM and returns a clean error response instead of crashing the request. |
| Embeddings (dense index) | `sawvial/embeddings/embedder.py` | **New in this pass:** explicit `pick_device()` instead of letting sentence-transformers auto-pick | library default (fp32) | **New in this pass:** OOM retry that halves the batch size (default 32) and retries, rather than failing the whole index build. |
| LoRA fine-tuning | `llm/train_lora.py` | `torch.cuda.is_available() and not args.force_cpu` (new `cuda_ok` flag) | `bfloat16` if supported, else `float16` on GPU, `float32` on CPU (previously `float16` even on CPU, which the new logic fixes) | Already used LoRA (small trainable parameter count), `gradient_checkpointing_enable()`, batch size 2, and gradient accumulation (`grad_accum` arg) before this pass — all appropriate for 8 GB. **New in this pass:** `--force_cpu` now genuinely keeps the model off the GPU even when one is present; `--4bit` now refuses to run under `--force_cpu` instead of silently requesting a `device_map={"":0}` that `cuda_ok=False` would make meaningless. |
| Text-to-speech (7 languages) | `llm/speech.py` | `"cuda" if (settings.DEVICE != "cpu" and torch.cuda.is_available()) else "cpu"` per model load | library default | **New in this pass:** models are evicted LRU-style once more than 2 are cached simultaneously (previously all 7 language models stayed resident forever), each eviction calling `torch.cuda.empty_cache()` if that model was on the GPU — bounds VRAM used by TTS instead of letting it grow unbounded across a session. |
| Evaluation harness | `evaluate_rag.py` | N/A (calls the already-running API over HTTP) | N/A | Not a model-loading workload itself. |

## CPU fallback policy

CPU is used only when:
1. `SAWVIAL_DEVICE=cpu` is explicitly set, or
2. `torch.cuda.is_available()` is false (no usable GPU/driver/CUDA build), or
3. `--force_cpu` is explicitly passed to `llm/train_lora.py` (now genuinely honoured end-to-end,
   see above).

There is no other place in the reviewed code that silently prefers CPU when a GPU is available.

## Out-of-memory handling (new in this pass)

Three places now explicitly catch `torch.cuda.OutOfMemoryError` instead of letting it propagate as
an unhandled exception:
1. `llm/rag_chat.py` `answer()` — retry generation once at half the token budget, freeing the CUDA
   cache first.
2. `api.py` `run_ask()` — if generation OOMs twice (including the one from item 1 failing again),
   or if the new one-shot regeneration call (task §5 of the brief) OOMs, the request returns a
   `gpu_out_of_memory` status with a clear message instead of a server error.
3. `sawvial/embeddings/embedder.py` `encode()` — halves its batch size and retries once per OOM,
   recursively (bounded by batch size reaching 1), for building/rebuilding the dense index.

## What was NOT runtime-verified

Everything above was verified by reading the code; none of it was executed, per this task's
instruction not to launch the application, start training, or download models. In particular:
- Actual VRAM usage figures, actual generation latency, and whether `torch.cuda.is_bf16_supported()`
  returns true for your specific driver/CUDA-build combination were not measured.
- The OOM retry paths were not exercised against a real low-memory condition.
- `/api/health`'s reported `"device"` field (mentioned in `README.md`) should show
  `"NVIDIA GeForce RTX 4060"` once you start the server, which is the one easy way to confirm all of
  the above end-to-end yourself.
