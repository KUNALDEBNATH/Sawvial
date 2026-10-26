#!/usr/bin/env python3
"""Evaluate SAWVIAL retrieval and answer behaviour. Retrieval and citation quality are reported
SEPARATELY, and nothing here claims scientific validity.

    python evaluate_rag.py                       # local, no LLM/GPU needed: retrieval + crop + unsupported
    python evaluate_rag.py --api http://localhost:8000      # end-to-end through a running API
    python evaluate_rag.py --api http://localhost:8000 --multilingual

Case sources
  * HELD-OUT cases: QA pairs from documents for which common.is_heldout_doc() is true. That is the same
    deterministic document split prepare_sft_data.py uses, so these documents are never in the
    training set (assuming the adapter was trained from data made with the default --val_pct 5).
  * CURATED cases (tests/eval_cases.jsonl): crop-specific, general, unsupported-crop, out-of-scope.
    Checked with invariants (crop evidence really concerns the crop; unsupported crops are refused),
    not with free-text judgement.

Honest limits: automatic gold labels come from template QA (many questions are generic), so unit-level
precision is expected to be low; doc_hit@k is the more meaningful retrieval number. Sets are small -
read the printed warnings. Final answer quality needs review by an agronomist.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "llm"))

from common import is_heldout_doc                                            # noqa: E402
from sawvial import settings                                                 # noqa: E402
from sawvial.eligibility import is_answerable                                # noqa: E402
from sawvial.validation.qa_checks import sft_usable                          # noqa: E402


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{n}: invalid JSON: {exc}") from exc


def heldout_cases(dataset_dir: Path, limit: int, seed: int, pct: int) -> tuple[list, dict]:
    cases, docs = [], set()
    for ku in load_jsonl(dataset_dir / "knowledge_units.jsonl"):
        doc = ku["source"]["document_id"]
        if not is_heldout_doc(doc, pct) or not is_answerable(ku):
            continue
        for qa in ku.get("qa", []):
            answer, _ = sft_usable(ku, qa)
            if answer:
                docs.add(doc)
                cases.append({"case_id": f"{ku['id']}:{len(cases)}", "type": "heldout", "question": qa["question"],
                              "relevant_ids": [ku["id"]], "relevant_doc": doc, "reference_answer": answer})
    random.Random(seed).shuffle(cases)
    return cases[:limit], {"heldout_documents": len(docs), "heldout_pool": len(cases)}


def retrieval_metrics(ids: list, relevant: set, k: int) -> dict:
    hits = [i for i, u in enumerate(ids[:k]) if u in relevant]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return {f"recall@{k}": len(hits) / len(relevant), "mrr": 1 / (hits[0] + 1) if hits else 0.0,
            f"ndcg@{k}": (sum(1 / math.log2(i + 2) for i in hits) / ideal) if ideal else 0.0}


def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.fmean(v), 4) if v else None


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 1)


# ------------------------------------------------------------------ local retrieval evaluation
def eval_retrieval(args, cases, curated):
    from sawvial.index.passages import retrieve_passages
    from sawvial.index.retriever import load_index_bundle
    bundle = load_index_bundle(args.index_dir, args.dataset_dir)
    k, rows, lat = args.top_k, [], []
    for c in cases:
        t = time.perf_counter()
        r = retrieve_passages(bundle, c["question"], k)
        lat.append((time.perf_counter() - t) * 1000)
        ids = [p["id"] for p in r["passages"]]
        docs = [p["document_id"] for p in r["passages"]]
        row = {"case_id": c["case_id"], **retrieval_metrics(ids, set(c["relevant_ids"]), k),
               "doc_hit": float(c["relevant_doc"] in docs[:k]), "n_returned": len(ids)}
        rows.append(row)
    heldout = {"n_cases": len(rows), f"recall@{k}": mean(rows, f"recall@{k}"), "mrr": mean(rows, "mrr"),
               f"ndcg@{k}": mean(rows, f"ndcg@{k}"), "doc_hit@k": mean(rows, "doc_hit"),
               "empty_result_rate": round(sum(r["n_returned"] == 0 for r in rows) / len(rows), 4) if rows else None}

    cur = []
    for c in curated:
        r = retrieve_passages(bundle, c["question"], k)
        ps, kind, ok, why = r["passages"], c["type"], True, []
        if kind == "crop":
            if not ps:
                ok, why = False, ["no passages for a crop the knowledge base should cover"]
            for p in ps:
                if not (p["crop_labeled"] or p["crop_mentions"] >= 2):
                    ok = False; why.append(f"{p['id']} is not crop-specific")
            if ps and not any(any(m.lower() in p["text"].lower() for m in c["must_mention_any"]) for p in ps):
                ok = False; why.append("no passage mentions the topic")
        elif kind == "unsupported":
            if ps or not r["crop_missing"]:
                ok, why = False, [f"expected 'no evidence for {c['crop']}', got {len(ps)} passages"]
        elif kind == "general":
            if not ps:
                ok, why = False, ["no passages"]
            elif not any(any(m.lower() in p["text"].lower() for m in c["must_mention_any"]) for p in ps):
                ok, why = False, ["no passage mentions the topic"]
        cur.append({"id": c["id"], "type": kind, "passed": ok, "problems": why, "n_passages": len(ps)})
    return {"heldout_retrieval": heldout, "curated": cur,
            "curated_pass_rate_by_type": {t: f"{sum(x['passed'] for x in cur if x['type']==t)}/{sum(1 for x in cur if x['type']==t)}"
                                          for t in sorted({x['type'] for x in cur if x['type'] != 'out_of_scope'})},
            "retrieval_latency_ms": {"mean": round(statistics.fmean(lat), 1) if lat else None,
                                     "p50": pct(lat, .5), "p95": pct(lat, .95)},
            "index": bundle["index_meta"]}


# ------------------------------------------------------------------ end-to-end (API) evaluation
def call_api(url, payload, timeout):
    req = Request(url.rstrip("/") + "/api/ask", data=json.dumps(payload).encode(), method="POST",
                  headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body, (time.perf_counter() - t) * 1000


def eval_api(args, cases, curated):
    health = json.loads(urlopen(args.api.rstrip("/") + "/api/health", timeout=10).read().decode())
    rows, k = [], args.top_k
    allc = [dict(c, case_id=c["case_id"]) for c in cases[:args.api_limit]] + \
           [dict(c, case_id=c["id"]) for c in curated]
    if args.multilingual:
        if not health.get("translator"):
            print("multilingual: SKIPPED - translator model not loaded on the server")
        else:
            allc += [{"case_id": f"lang-{lang}", "type": "multilingual", "language": lang,
                      "question": q} for lang, q in (("hi", "मिट्टी की उर्वरता कैसे बढ़ाएं?"),
                                                    ("bn", "মাটির উর্বরতা কীভাবে বাড়ানো যায়?"))]
    for i, c in enumerate(allc):
        try:
            res, ms = call_api(args.api, {"question": c["question"], "language": c.get("language", "en"), "top_k": k},
                               args.timeout)
        except (URLError, TimeoutError, OSError) as exc:
            rows.append({"case_id": c["case_id"], "type": c["type"], "error": str(exc)})
            continue
        g, srcs = res.get("grounding", {}), res.get("sources", [])
        retrieved = [s["id"] for s in srcs]
        row = {"case_id": c["case_id"], "type": c["type"], "latency_ms": round(ms, 1), "warmup": i == 0,
               "grounded": bool(res.get("grounded")), "status": g.get("status"),
               "timings_ms": res.get("timings_ms", {}),
               "invalid_citations": len(g.get("invalid_citations", [])),
               "citation_coverage": (g["supported_claims"] / g["claims"]) if g.get("claims") else None,
               "cited_source_ratio": (sum(s["cited"] for s in srcs) / len(srcs)) if srcs else None}
        if c["type"] == "heldout":
            row.update(retrieval_metrics(retrieved, set(c["relevant_ids"]), k))
        if c["type"] == "unsupported":
            row["correctly_refused"] = g.get("status") == "insufficient_evidence" and not res.get("grounded")
        if c["type"] == "out_of_scope":
            row["correctly_refused"] = not res.get("grounded")
        if c["type"] == "multilingual":
            row["answer_in_requested_language"] = res.get("answer_language") == c["language"]
        rows.append(row)
    ok = [r for r in rows if "error" not in r]
    steady = [r["latency_ms"] for r in ok if not r["warmup"]]
    answered = [r for r in ok if r["grounded"]]
    stage = {s: round(statistics.fmean(r["timings_ms"][s] for r in ok if s in r["timings_ms"]), 1)
             for s in ("translate_in", "retrieval", "generation", "grounding", "translate_out")
             if any(s in r["timings_ms"] for r in ok)}
    return {"server": {"llm": health.get("llm"), "translator": health.get("translator"), "index": health.get("index")},
            "n_requests": len(rows), "n_errors": len(rows) - len(ok),
            "citation": {"answered_rate": round(len(answered) / len(ok), 4) if ok else None,
                         "mean_citation_coverage": mean(answered, "citation_coverage"),
                         "invalid_citation_total": sum(r["invalid_citations"] for r in ok),
                         "partially_supported": sum(r["status"] == "partially_supported" for r in ok),
                         "extractive_fallback": sum(r["status"] == "extractive_fallback" for r in ok)},
            "heldout_retrieval_via_api": {f"recall@{k}": mean([r for r in ok if r["type"] == "heldout"], f"recall@{k}"),
                                          "mrr": mean([r for r in ok if r["type"] == "heldout"], "mrr")},
            "unsupported_refused": f"{sum(bool(r.get('correctly_refused')) for r in ok if r['type']=='unsupported')}/"
                                   f"{sum(1 for r in ok if r['type']=='unsupported')}",
            "out_of_scope_refused": f"{sum(bool(r.get('correctly_refused')) for r in ok if r['type']=='out_of_scope')}/"
                                    f"{sum(1 for r in ok if r['type']=='out_of_scope')}",
            "multilingual": [{r["case_id"]: r["answer_in_requested_language"]} for r in ok if r["type"] == "multilingual"],
            "latency_ms_excluding_first_request": {"mean": round(statistics.fmean(steady), 1) if steady else None,
                                                   "p50": pct(steady, .5), "p95": pct(steady, .95)},
            "mean_stage_ms": stage, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", help="base URL of a running api.py for end-to-end evaluation")
    ap.add_argument("--dataset_dir", default=str(settings.DATASET_DIR)); ap.add_argument("--index_dir", default=str(settings.INDEX_DIR))
    ap.add_argument("--limit", type=int, default=300, help="max held-out cases (local retrieval)")
    ap.add_argument("--api_limit", type=int, default=20, help="max held-out cases sent through the API (each runs the LLM)")
    ap.add_argument("--top_k", type=int, default=5); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_pct", type=int, default=5, help="must equal the value used for prepare_sft_data.py")
    ap.add_argument("--timeout", type=int, default=300); ap.add_argument("--multilingual", action="store_true")
    ap.add_argument("--out_dir", default=str(ROOT / "eval_results"))
    args = ap.parse_args()

    cases, pool = heldout_cases(Path(args.dataset_dir), args.limit, args.seed, args.val_pct)
    curated = list(load_jsonl(ROOT / "tests" / "eval_cases.jsonl"))
    warnings = ["Not scientifically validated: automatic gold labels + lexical checks only; agronomist review required."]
    if len(cases) < 30:
        warnings.append(f"only {len(cases)} held-out cases - too few for reliable averages")
    if args.api and len(cases[:args.api_limit]) < 30:
        warnings.append(f"API run uses {len(cases[:args.api_limit])} held-out cases (LLM calls are slow) - indicative only")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "mode": "api" if args.api else "retrieval_only",
              "split": {"val_pct": args.val_pct, **pool, "note": "held-out docs are excluded from SFT training"},
              "warnings": warnings}
    try:
        report["results"] = eval_api(args, cases, curated) if args.api else eval_retrieval(args, cases, curated)
    except (URLError, OSError) as exc:
        sys.exit(f"Could not reach the API at {args.api}: {exc}\nStart it first: python api.py")
    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    f = out / f"{report['mode']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in report["results"].items() if k not in ("rows", "curated", "index")}
    print(json.dumps({"mode": report["mode"], "split": report["split"], **summary}, indent=2, ensure_ascii=False))
    for w in warnings:
        print("WARNING:", w)
    bad = [c for c in report["results"].get("curated", []) if not c["passed"]]
    for c in bad:
        print(f"CURATED FAIL {c['id']}: {'; '.join(c['problems'])}")
    print(f"\nFull results: {f}")


if __name__ == "__main__":
    main()
