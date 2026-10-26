#!/usr/bin/env python3
"""Ask questions: retrieve from the SAWVIAL index, then answer with the fine-tuned LLM.

    python llm/rag_chat.py -q "Effect of drip irrigation on cotton WUE?"
    python llm/rag_chat.py --base_only -q "..."        # base model without the LoRA adapter
    python llm/rag_chat.py --require_adapter           # fail instead of falling back to base

Adapter policy
--------------
* Default ("auto"): the trained adapter is loaded if it exists AND is compatible with the
  base model. An adapter that is present but incompatible is a hard error (never ignored).
  A missing adapter falls back to the base model with a loud warning, which /api/health
  also reports.
* ``--base_only`` is the explicit way to run without the adapter.
* ``--require_adapter`` turns a missing adapter into an error as well.
"""
import argparse
import logging
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root -> sawvial/
sys.path.insert(0, str(Path(__file__).resolve().parent))          # llm/ -> common

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sawvial import settings
from sawvial.index.passages import get_passages, retrieve_passages        # noqa: F401  (re-exported)
from sawvial.index.retriever import load_index_bundle
from sawvial.model_config import ModelConfigError, inspect_adapter
from common import SYSTEM_PROMPT, format_context, make_user_prompt

log = logging.getLogger("sawvial.llm")
DOWNLOAD_HINT = "python tools/download_models.py"


def load_retriever(index_dir, dataset_dir):
    """Returns (bundle, embedder). embedder is None for a lexical-only index."""
    bundle = load_index_bundle(index_dir, dataset_dir)
    return bundle, bundle.get("dense_embedder")


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def load_llm(args):
    """Load the base model (+ adapter) ONCE. Returns (tokenizer, model, status_dict)."""
    base = getattr(args, "base_model", None) or settings.BASE_MODEL
    adapter_dir = Path(getattr(args, "adapter", None) or settings.ADAPTER_DIR)
    base_only = bool(getattr(args, "no_adapter", False))
    require = bool(getattr(args, "require_adapter", False))
    local_only = settings.OFFLINE

    status = {"base_model": base, "adapter_path": str(adapter_dir), "mode": "base_only",
              "adapter_status": "disabled", "message": "adapter disabled by --base_only",
              "merged": False, "four_bit": False}

    check = None
    if not base_only:
        check = inspect_adapter(adapter_dir, base)
        if check.status == "missing":
            if require:
                raise ModelConfigError(f"--require_adapter set but {check.message()}")
            log.warning("ADAPTER NOT FOUND (%s) - running the BASE MODEL ONLY. Answers will not "
                        "reflect fine-tuning.", check.message())
            status.update(adapter_status="missing", message=check.message())
            check = None
        elif check.status == "invalid":
            raise ModelConfigError(
                f"Adapter at {adapter_dir} is not usable: {check.message()}. Fix the problem or "
                f"start with --base_only to run without the adapter.")

    try:
        from transformers import AutoConfig
        base_cfg = AutoConfig.from_pretrained(base, local_files_only=local_only)
    except OSError as exc:
        raise ModelConfigError(
            f"Base model {base!r} is not in the local cache ({exc.__class__.__name__}). "
            f"Download it once with: {DOWNLOAD_HINT}") from exc

    if check is not None:                                   # cross-check shapes vs real base config
        check = inspect_adapter(adapter_dir, base, getattr(base_cfg, "hidden_size", None),
                                getattr(base_cfg, "num_hidden_layers", None))
        if not check.ok:
            raise ModelConfigError(f"Adapter at {adapter_dir} does not match base model {base!r}: "
                                   f"{check.message()}. Use --base_only to run without the adapter.")

    use_4bit = bool(getattr(args, "four_bit", False)) or bool(check and check.four_bit_trained)
    cuda = torch.cuda.is_available()
    if use_4bit and not (cuda and _bnb_available()):
        if cuda:
            log.warning("4-bit loading requested but bitsandbytes is not installed; using 16-bit weights")
        use_4bit = False
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    kw = {"torch_dtype": dtype if cuda else torch.float32}
    if use_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": 0}
    try:
        tok = AutoTokenizer.from_pretrained(base, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(base, local_files_only=local_only, **kw)
    except OSError as exc:
        raise ModelConfigError(f"Could not load base model {base!r} from the local cache: {exc}. "
                               f"Run: {DOWNLOAD_HINT}") from exc

    if check is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        status.update(mode="adapter", adapter_status="ok", message="LoRA adapter loaded")
        if not use_4bit:
            model = model.merge_and_unload()               # no per-token LoRA overhead
            status["merged"] = True
    if cuda and not use_4bit:
        model.to("cuda")
    model.eval()
    status["four_bit"] = use_4bit
    log.info("LLM ready: %s", status)
    return tok, model, status


@torch.no_grad()
def answer(tok, model, question, passages, max_new_tokens, history=None,
           cancel_event: threading.Event | None = None, do_sample: bool = False):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(
                format_context(passages), question, history=history)}]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
              add_special_tokens=False, return_tensors="pt").to(model.device)
    if cancel_event is not None and cancel_event.is_set():
        return ""
    stopping_criteria = None
    if cancel_event is not None:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class CancelOnEvent(StoppingCriteria):
            def __call__(self, input_ids, _scores, **_kwargs):
                return torch.full((input_ids.shape[0],), cancel_event.is_set(),
                                  dtype=torch.bool, device=input_ids.device)

        stopping_criteria = StoppingCriteriaList([CancelOnEvent()])
    gen_kw = dict(do_sample=True, temperature=0.35, top_p=0.9) if do_sample else dict(do_sample=False)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         stopping_criteria=stopping_criteria, **gen_kw)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from sawvial import grounding
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question")
    ap.add_argument("--dataset_dir", default=str(settings.DATASET_DIR))
    ap.add_argument("--index_dir", default=str(settings.INDEX_DIR))
    ap.add_argument("--adapter", default=str(settings.ADAPTER_DIR))
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--base_only", "--no_adapter", dest="no_adapter", action="store_true",
                    help="run the base model without the LoRA adapter")
    ap.add_argument("--require_adapter", action="store_true")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--top_k", type=int, default=settings.TOP_K)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    args = ap.parse_args()

    bundle, embedder = load_retriever(args.index_dir, args.dataset_dir)
    try:
        tok, model, status = load_llm(args)
    except ModelConfigError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"[model] mode={status['mode']} adapter={status['adapter_status']}")

    def run(q):
        res = retrieve_passages(bundle, q, args.top_k, embedder)
        if res["crop_missing"]:
            print(f"\nThe knowledge base has no evidence specific to: {', '.join(res['crops'])}\n")
            return
        passages = res["passages"]
        if not passages:
            print("\nNo usable passages found.\n")
            return
        raw = answer(tok, model, q, passages, args.max_new_tokens)
        report = grounding.assess(raw, passages)
        text, ids = grounding.compose_supported_answer(report)
        print("\n" + (text or "(model output could not be verified against the sources)"))
        print(f"[grounding] supported {report['n_supported']}/{report['n_claims']} sentences; "
              f"invalid citations: {report['invalid_citations'] or 'none'}")
        print("\nSources:")
        for p in passages:
            print(f"  [{p['id']}] {p['title']} (p.{p['page']})  score={p['score']:.3f}"
                  f"{'  <- cited' if p['id'] in ids else ''}")
        print()

    if args.question:
        run(args.question)
    else:
        while True:
            try:
                q = input("Question (blank to quit): ").strip()
            except EOFError:
                break
            if not q:
                break
            run(q)


if __name__ == "__main__":
    main()
