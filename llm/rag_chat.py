#!/usr/bin/env python3
"""Ask questions: retrieve from the SAWVIAL FAISS index, then answer with the fine-tuned LLM.

    python llm/rag_chat.py --offline                       # interactive
    python llm/rag_chat.py --offline -q "Effect of drip irrigation on cotton WUE?"
    python llm/rag_chat.py --offline --no_adapter -q "..."  # base model, for comparison
"""
import argparse, json, sys
import re
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root -> sawvial/

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sawvial.index.retriever import load_index_bundle, query as retrieve
from sawvial.embeddings.embedder import TFIDFEmbedder, load_embedder
from common import SYSTEM_PROMPT, format_context, make_user_prompt, is_usable_evidence

QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "between", "by",
    "did", "do", "does", "during", "effect", "find", "found", "for", "from",
    "give", "how", "in", "into", "is", "it", "of", "on", "or", "over",
    "report", "reported", "result", "results", "say", "show", "source",
    "study", "the", "their", "these", "this", "those", "to", "was", "were",
    "what", "when", "where", "which", "while", "who", "why", "with",
    "agriculture", "agricultural", "crop", "crops", "farming", "research",
    "system", "systems", "year", "years",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def _normalize_evidence_term(word):
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("oes") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 4:
        return word[:-1]
    return word


def _focus_crop_labeled_evidence(
    evidence: str, crops: list[str], requested_crops: set[str],
) -> tuple[str, bool]:
    markers = []
    for crop in crops:
        crop_name = str(crop).strip().lower()
        if not crop_name:
            continue
        plural = (crop_name[:-1] + "ies" if crop_name.endswith("y")
                  else crop_name + "es" if crop_name.endswith(("o", "ch", "sh", "x", "z", "s"))
                  else crop_name + "s")
        pattern = re.compile(
            rf"(?<!\w)(?:{re.escape(crop_name)}|{re.escape(plural)})\s*:",
            re.I,
        )
        markers.extend(
            (match.start(), match.end(), crop_name)
            for match in pattern.finditer(evidence)
        )
    markers.sort()
    if not markers:
        return evidence, False

    targets = requested_crops.intersection(crops)
    target_markers = [marker for marker in markers if marker[2] in targets]
    if not targets or {marker[2] for marker in target_markers} != targets:
        return evidence, False

    segments = []
    for start, end, crop in target_markers:
        next_marker = next(
            (marker_start for marker_start, _, _ in markers if marker_start > start),
            len(evidence),
        )
        detail = evidence[end:next_marker].strip(" \t\r\n;,:")
        if detail:
            segments.append(f"{crop}: {detail}")
    return ("; ".join(segments), True) if segments else (evidence, False)


def load_retriever(index_dir, dataset_dir):
    bundle = load_index_bundle(index_dir, dataset_dir)
    meta = bundle["index_meta"]["embedding"]
    embedder = load_embedder(meta["model_name"], meta["offline"])
    return bundle, embedder


def get_passages(bundle, embedder, question, top_k):
    candidate_count = min(bundle["index"].ntotal, max(top_k * 25, 100))
    retrieval_query = question
    expansion_terms = set()
    if isinstance(embedder, TFIDFEmbedder):
        lower_question = question.lower()
        if re.search(r"\bpotatoes?\b", lower_question) and re.search(
            r"\b(water|irrigat\w*)\b", lower_question,
        ):
            retrieval_query += " potato irrigation"
            expansion_terms.update({"potato", "irrigation"})
        if re.search(r"\bsoil\b", lower_question) and re.search(
            r"\b(fertil\w*|nutrient\w*)\b", lower_question,
        ):
            retrieval_query += (
                " soil fertility improvement manure compost organic amendments"
            )
            expansion_terms.update(
                {"fertility", "manure", "compost", "organic", "amendments"}
            )
        if re.search(r"\birrigat\w*\b", lower_question) and re.search(
            r"\b(soil|water)\b", lower_question,
        ):
            retrieval_query += (
                " irrigation efficiency soil moisture water retention "
                "infiltration water management"
            )
            expansion_terms.update(
                {"efficiency", "moisture", "retention", "infiltration", "management"}
            )
        if re.search(r"\b(water|irrigat\w*)\b", lower_question) and re.search(
            r"\b(conserv\w*|sav\w*|benefit|efficient\w*)\b", lower_question,
        ):
            retrieval_query += (
                " water conservation water-saving irrigation scheduling "
                "soil moisture monitoring efficient irrigation water use "
                "drip sprinkler mulching"
            )
            expansion_terms.update({
                "conservation", "saving", "irrigation", "scheduling",
                "moisture", "monitoring", "efficiency", "efficient",
                "drip", "sprinkler", "mulching", "water",
            })
    hits = retrieve(bundle, embedder, retrieval_query, top_k=candidate_count)
    question_words = set(re.findall(r"[a-z]+", question.lower()))
    indexed_crops = {
        crop.lower()
        for ku in bundle["units_by_id"].values()
        for crop in ku.get("context", {}).get("crop", [])
        if isinstance(crop, str) and crop.strip()
    }
    mentioned_crops = {
        crop for crop in indexed_crops
        if crop in question_words
        or (crop.endswith("y") and crop[:-1] + "ies" in question_words)
        or (crop.endswith(("o", "ch", "sh", "x", "z", "s")) and crop + "es" in question_words)
        or (not crop.endswith(("y", "o", "ch", "sh", "x", "z", "s"))
            and crop + "s" in question_words)
    }
    compare_crops = bool(re.search(
        r"\b(other|another|different|more)\s+(kind\s+of\s+)?crops?\b",
        question.lower(),
    ))
    if mentioned_crops and not compare_crops:
        crop_hits = [
            hit for hit in hits
            if mentioned_crops.intersection(
                str(crop).lower() for crop in hit["context"].get("crop", [])
            )
        ]
        if crop_hits:
            hits = crop_hits

    query_terms = {
        _normalize_evidence_term(word)
        for word in re.findall(r"[a-z]{2,}", question.lower())
        if word not in QUERY_STOPWORDS and word not in mentioned_crops
    }
    query_terms.update(expansion_terms)
    out = []
    for r in hits:
        if r["validation_status"] not in {"validated", "needs_review"}:
            continue
        title = r["source"].get("title")
        if not is_usable_evidence(r["evidence"], title):    # skip reference lists / title-only hits
            continue
        text = r["evidence"]
        evidence_terms = {
            _normalize_evidence_term(word)
            for word in re.findall(r"[a-z]{2,}", text.lower())
        }
        if query_terms and not query_terms.intersection(evidence_terms):
            continue
        passage_crops = [
            str(crop).strip().lower()
            for crop in r["context"].get("crop", [])
            if isinstance(crop, str) and crop.strip()
        ]
        focused_text, crop_focused = _focus_crop_labeled_evidence(
            text, passage_crops, mentioned_crops,
        )
        passage_context = r["context"]
        if crop_focused:
            focused_crops = sorted(set(passage_crops).intersection(mentioned_crops))
            passage_context = {"crop": focused_crops}
        out.append({"id": r["knowledge_unit_id"], "title": title,
                    "page": r["provenance"].get("page"), "section": r["provenance"].get("section"),
                    "text": focused_text,
                    "finding": None if crop_focused else r["finding"],
                    "recommendation": None if crop_focused else r["recommendation"],
                    "context": passage_context,
                    "structured_facts": [] if crop_focused else r["structured_facts"],
                    "crop_focused": crop_focused,
                    "score": r["score"]})
        if len(out) == top_k:
            break
    return out


def load_llm(args):
    info = {}
    adapter = Path(args.adapter)
    if not args.no_adapter and (adapter / "sawvial_train_info.json").exists():
        info = json.load(open(adapter / "sawvial_train_info.json"))
    base = args.base_model or info.get("base_model") or "Qwen/Qwen2.5-1.5B-Instruct"
    use_4bit = args.four_bit or info.get("four_bit", False)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    kw = {"torch_dtype": dtype if torch.cuda.is_available() else torch.float32}
    if use_4bit and torch.cuda.is_available():
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": 0}
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, **kw)
    if not args.no_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    if torch.cuda.is_available() and not use_4bit:
        model.to("cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def answer(tok, model, question, passages, max_new_tokens, history=None,
           cancel_event: threading.Event | None = None):
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
                return torch.full(
                    (input_ids.shape[0],), cancel_event.is_set(),
                    dtype=torch.bool, device=input_ids.device,
                )

        stopping_criteria = StoppingCriteriaList([CancelOnEvent()])
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=0.35, top_p=0.9, repetition_penalty=1.1,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         stopping_criteria=stopping_criteria)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question")
    ap.add_argument("--dataset_dir", default="./dataset")
    ap.add_argument("--index_dir", default="./index")
    ap.add_argument("--adapter", default="./llm/sawvial-lora")
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--no_adapter", action="store_true")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--offline", action="store_true", help="(kept for symmetry; index metadata decides)")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    args = ap.parse_args()

    bundle, embedder = load_retriever(args.index_dir, args.dataset_dir)
    tok, model = load_llm(args)

    def run(q):
        passages = get_passages(bundle, embedder, q, args.top_k)
        if not passages:
            print("\nNo usable passages found.\n"); return
        print("\n" + answer(tok, model, q, passages, args.max_new_tokens))
        print("\nSources:")
        for p in passages:
            print(f"  [{p['id']}] {p['title']} (p.{p['page']})  score={p['score']:.3f}")
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
