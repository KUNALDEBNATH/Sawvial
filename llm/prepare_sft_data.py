#!/usr/bin/env python3
"""Convert dataset/knowledge_units.jsonl -> chat-format SFT data (RAG-aware).

Each training example = retrieved-style context (the true passage + 0-2 distractors)
+ question -> grounded answer with [ID] citation. ~5% are "not in context" refusals.
Train/val split is by document so there is no leakage.

    python llm/prepare_sft_data.py --dataset_dir ./dataset --out_dir ./llm/sft
"""
import argparse, hashlib, json, random
from collections import defaultdict
from pathlib import Path

from common import (SYSTEM_PROMPT, NO_ANSWER, load_jsonl, usable_unit,
                    format_context, make_user_prompt, fix_grammar)


def passage(ku):
    s = ku["source"]
    return {"id": ku["id"], "title": s.get("title"), "page": s.get("page"),
            "section": s.get("section"), "text": ku["knowledge"]["evidence"]}


def is_val(doc_id, pct):
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 100 < pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="./dataset")
    ap.add_argument("--out_dir", default="./llm/sft")
    ap.add_argument("--include_needs_review", action="store_true")
    ap.add_argument("--max_samples", type=int, default=8000)
    ap.add_argument("--max_per_doc", type=int, default=60)
    ap.add_argument("--val_pct", type=int, default=5)
    ap.add_argument("--refusal_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    all_units = list(load_jsonl(Path(args.dataset_dir) / "knowledge_units.jsonl"))
    units = [k for k in all_units if usable_unit(k, args.include_needs_review)]
    print(f"{len(units)} / {len(all_units)} knowledge units usable "
          f"(dropped reference lists, title-only, short, non-validated)")

    by_domain = defaultdict(list)
    for k in units:
        by_domain[k["classification"]["primary_domain"]].append(k)
    domains = list(by_domain)

    def distractors(ku, n, other_domain=False):
        dom = ku["classification"]["primary_domain"]
        if other_domain:
            pool = [k for d in domains if d != dom for k in rnd.sample(by_domain[d], min(50, len(by_domain[d])))]
        else:
            pool = rnd.sample(by_domain[dom], min(50, len(by_domain[dom])))
        doc = ku["source"]["document_id"]
        pool = [k for k in pool if k["source"]["document_id"] != doc]
        return rnd.sample(pool, min(n, len(pool)))

    def example(ku, q, a, kind, ctx_units):
        rnd.shuffle(ctx_units)
        user = make_user_prompt(format_context([passage(u) for u in ctx_units]), q)
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user},
                             {"role": "assistant", "content": a}],
                "meta": {"unit": ku["id"], "doc": ku["source"]["document_id"], "kind": kind}}

    examples = []
    for ku in units:
        title = (ku["source"].get("title") or "").strip()
        for qa in ku.get("qa", [])[:2]:
            ans = (qa.get("answer") or "").strip()
            if len(ans) < 40 or ans == title:
                continue
            examples.append(example(ku, qa["question"], f"{fix_grammar(ans)} [{ku['id']}]",
                                    "qa", [ku] + distractors(ku, rnd.randint(0, 2))))
        k = ku["knowledge"]
        if k.get("recommendation") and k.get("recommendation_type") == "source_explicit":
            cl = ku["classification"]
            topic = (cl.get("topics") or cl.get("subdomains") or ["this topic"])[0].replace("_", " ")
            examples.append(example(ku, f"What does the source recommend regarding {topic}?",
                                    f"{k['recommendation'].strip()} [{ku['id']}]",
                                    "recommendation", [ku] + distractors(ku, rnd.randint(0, 2))))
        if rnd.random() < args.refusal_frac and ku.get("qa"):
            examples.append(example(ku, ku["qa"][0]["question"], NO_ANSWER, "refusal",
                                    distractors(ku, 3, other_domain=True)))

    rnd.shuffle(examples)
    per_doc, kept = defaultdict(int), []
    for ex in examples:
        d = ex["meta"]["doc"]
        if per_doc[d] >= args.max_per_doc:
            continue
        per_doc[d] += 1
        kept.append(ex)
    kept = kept[:args.max_samples]

    train = [e for e in kept if not is_val(e["meta"]["doc"], args.val_pct)]
    val = [e for e in kept if is_val(e["meta"]["doc"], args.val_pct)]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = defaultdict(int)
    for e in kept:
        kinds[e["meta"]["kind"]] += 1
    print(f"train={len(train)} val={len(val)} kinds={dict(kinds)} -> {out}")


if __name__ == "__main__":
    main()
