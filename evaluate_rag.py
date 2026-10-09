#!/usr/bin/env python3
"""Evaluate SAWVIAL retrieval and answer behavior against local gold examples.

Run ``python evaluate_rag.py --smoke`` for a small live-API regression check,
or ``python evaluate_rag.py --limit 25`` to sample labeled dataset Q&A pairs.
Custom JSONL cases can provide question, relevant_ids, reference_answer,
history, and expected_format (for example, "one_line").
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent

SMOKE_CASES = [
    {
        "case_id": "farm-water-conservation",
        "question": "Tell me how to conserve water in farming.",
        "relevant_ids": ["SWM_012405"],
    },
    {
        "case_id": "contextual-water-benefit",
        "question": "How to use water so that can benefit?",
        "history": [
            {
                "role": "user",
                "content": "Tell me how to conserve water in farming.",
            },
            {
                "role": "assistant",
                "content": (
                    "The dataset discusses irrigation scheduling and soil "
                    "moisture monitoring to conserve water [SWM_012405]."
                ),
            },
        ],
        "relevant_ids": ["SWM_012405"],
    },
    {
        "case_id": "water-one-line-after-followups",
        "question": "Write it in one line.",
        "history": [
            {
                "role": "user",
                "content": "Tell me how to conserve water in farming.",
            },
            {
                "role": "assistant",
                "content": (
                    "The dataset discusses irrigation scheduling and soil "
                    "moisture monitoring to conserve water [SWM_012405]."
                ),
            },
            {
                "role": "user",
                "content": "How to use water so that can benefit?",
            },
            {
                "role": "assistant",
                "content": (
                    "The records connect irrigation scheduling and soil "
                    "moisture monitoring with water conservation [SWM_012405]."
                ),
            },
            {"role": "user", "content": "Make it short."},
            {
                "role": "assistant",
                "content": (
                    "Use irrigation scheduling and soil moisture monitoring "
                    "to conserve water [SWM_012405]."
                ),
            },
        ],
        "relevant_ids": ["SWM_012405"],
        "expected_format": "one_line",
    },
    {
        "case_id": "potato-irrigation",
        "question": "How do I grow potatoes using water properly?",
        "relevant_ids": ["SWM_000671"],
    },
    {
        "case_id": "potato-one-line-followup",
        "question": "Write that in one line.",
        "history": [
            {"role": "user", "content": "How do I grow potatoes using water properly?"},
            {
                "role": "assistant",
                "content": (
                    "The records mention three irrigations for potato, but do not "
                    "specify their timing or amount [SWM_000671]."
                ),
            },
        ],
        "relevant_ids": ["SWM_000671"],
        "expected_format": "one_line",
    },
    {
        "case_id": "soil-fertility",
        "question": "I want to make my soil more fertile. How?",
        "relevant_ids": ["SWM_003637", "SWM_002322", "SWM_002555"],
    },
    {
        "case_id": "irrigation-efficiency",
        "question": "How can I improve irrigation efficiency?",
        "relevant_ids": ["SWM_012781", "SWM_014375", "SWM_012778"],
    },
]


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def usable_gold_units(dataset_dir: Path) -> set[str]:
    valid = set()
    for unit in load_jsonl(dataset_dir / "knowledge_units.jsonl"):
        evidence = unit.get("knowledge", {}).get("evidence") or ""
        source = unit.get("source", {})
        status = unit.get("quality", {}).get("validation_status")
        section = (source.get("section") or "").lower()
        if (status in {"validated", "needs_review"} and len(evidence) >= 80
                and len(evidence.split()) >= 12
                and section not in {"references", "reference", "bibliography", "literature cited"}):
            valid.add(unit["id"])
    return valid


def load_default_cases(dataset_dir: Path, limit: int, seed: int) -> list[dict]:
    valid_ids = usable_gold_units(dataset_dir)
    cases = []
    for row in load_jsonl(dataset_dir / "qa.jsonl"):
        unit_id = row.get("knowledge_unit_id")
        question = (row.get("question") or "").strip()
        reference = (row.get("answer") or "").strip()
        if unit_id in valid_ids and len(question) >= 12 and len(reference) >= 60:
            cases.append({
                "case_id": f"{unit_id}:{len(cases) + 1}",
                "question": question,
                "relevant_ids": [unit_id],
                "reference_answer": reference,
            })
    if not cases:
        raise ValueError(f"No usable labeled Q&A cases found under {dataset_dir}")
    random.Random(seed).shuffle(cases)
    return cases[:limit]


def load_custom_cases(path: Path) -> list[dict]:
    cases = list(load_jsonl(path))
    for number, case in enumerate(cases, 1):
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"{path}:{number}: each case needs a non-empty question")
        ids = case.get("relevant_ids", [])
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
            raise ValueError(f"{path}:{number}: relevant_ids must be a list of IDs")
        case["relevant_ids"] = ids
        if "history" in case and not isinstance(case["history"], list):
            raise ValueError(f"{path}:{number}: history must be a list of chat turns")
    return cases


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = Counter(tokens(prediction)), Counter(tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted or not expected or not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted, expected = tokens(prediction), tokens(reference)
    if not predicted or not expected:
        return 0.0
    previous = [0] * (len(expected) + 1)
    for predicted_token in predicted:
        current = [0]
        for index, expected_token in enumerate(expected, 1):
            if predicted_token == expected_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def retrieval_metrics(retrieved: list[str], relevant: set[str], k: int) -> dict:
    top_k = retrieved[:k]
    hits = [index for index, unit_id in enumerate(top_k) if unit_id in relevant]
    dcg = sum(1 / math.log2(index + 2) for index in hits)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        f"precision@{k}": len(hits) / k,
        f"recall@{k}": len(hits) / len(relevant) if relevant else None,
        "reciprocal_rank": 1 / (hits[0] + 1) if hits else 0.0,
        f"ndcg@{k}": dcg / ideal_dcg if ideal_dcg else None,
    }


def ask_api(api_url: str, case: dict, top_k: int, timeout: int) -> dict:
    payload = {
        "question": case["question"],
        "language": case.get("language", "en"),
        "top_k": top_k,
        "history": case.get("history", []),
    }
    request = Request(
        api_url.rstrip("/") + "/api/ask",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate_case(case: dict, response: dict, k: int, latency_ms: float) -> dict:
    answer = response.get("answer") or ""
    sources = response.get("sources") or []
    retrieved = [source.get("id", "") for source in sources if source.get("id")]
    result = {
        "case_id": case.get("case_id"),
        "question": case["question"],
        "answer": answer,
        "grounded": bool(response.get("grounded")),
        "latency_ms": round(latency_ms, 1),
        "retrieved_ids": retrieved,
        "metrics": {
            **(
                retrieval_metrics(retrieved, set(case["relevant_ids"]), k)
                if case["relevant_ids"] else {}
            ),
            "cited_source_ratio": (
                sum(bool(source.get("cited")) for source in sources) / len(sources)
                if sources else 0.0
            ),
        },
    }
    if case.get("reference_answer"):
        result["metrics"]["token_f1"] = token_f1(answer, case["reference_answer"])
        result["metrics"]["rouge_l_f1"] = rouge_l_f1(answer, case["reference_answer"])
    if case.get("expected_format") == "one_line":
        result["metrics"]["one_line_compliance"] = float("\n" not in answer.strip())
    return result


def mean_metric(results: list[dict], name: str) -> float | None:
    values = [
        row["metrics"][name] for row in results
        if name in row["metrics"] and row["metrics"][name] is not None
    ]
    return round(statistics.fmean(values), 4) if values else None


def summarize(results: list[dict], k: int) -> dict:
    return {
        "cases": len(results),
        f"precision@{k}": mean_metric(results, f"precision@{k}"),
        f"recall@{k}": mean_metric(results, f"recall@{k}"),
        "mrr": mean_metric(results, "reciprocal_rank"),
        f"ndcg@{k}": mean_metric(results, f"ndcg@{k}"),
        "grounded_rate": round(
            sum(row["grounded"] for row in results) / len(results), 4,
        ) if results else 0.0,
        "cited_source_ratio": mean_metric(results, "cited_source_ratio"),
        "token_f1": mean_metric(results, "token_f1"),
        "rouge_l_f1": mean_metric(results, "rouge_l_f1"),
        "one_line_compliance": mean_metric(results, "one_line_compliance"),
        "mean_latency_ms": round(
            statistics.fmean(row["latency_ms"] for row in results), 1,
        ) if results else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--cases", type=Path, help="Custom benchmark cases in JSONL format")
    parser.add_argument("--smoke", action="store_true", help="Run the built-in regression cases")
    parser.add_argument("--limit", type=int, default=25, help="Number of dataset Q&A samples")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation-results.json")
    args = parser.parse_args()

    if args.top_k < 1 or args.limit < 1 or args.timeout < 1:
        parser.error("--top-k, --limit, and --timeout must be positive")
    if args.smoke and args.cases:
        parser.error("--smoke and --cases cannot be used together")
    if args.smoke:
        cases = SMOKE_CASES
    elif args.cases:
        cases = load_custom_cases(args.cases)
    else:
        cases = load_default_cases(args.dataset_dir, args.limit, args.seed)

    results = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        try:
            response = ask_api(args.api_url, case, args.top_k, args.timeout)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"[{index}/{len(cases)}] {case.get('case_id', case['question'])}: {exc}",
                  file=sys.stderr)
            return 1
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = evaluate_case(case, response, args.top_k, elapsed_ms)
        results.append(result)
        mrr = result["metrics"].get("reciprocal_rank")
        print(
            f"[{index}/{len(cases)}] {case.get('case_id', case['question'][:48])} "
            f"grounded={result['grounded']} rr={mrr if mrr is not None else 'n/a'} "
            f"latency={elapsed_ms:.0f}ms"
        )

    report = {
        "api_url": args.api_url,
        "top_k": args.top_k,
        "seed": args.seed,
        "summary": summarize(results, args.top_k),
        "results": results,
        "metric_notes": {
            "retrieval": "Gold knowledge-unit IDs are used for precision/recall, MRR, and nDCG.",
            "answer_similarity": (
                "Token F1 and ROUGE-L compare wording to references; valid paraphrases may score low."
            ),
            "faithfulness": (
                "Not automatically judged: use human review or a calibrated LLM judge for factual "
                "faithfulness. Grounded rate and cited-source ratio are not faithfulness scores."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print("\nAggregate metrics")
    print(json.dumps(report["summary"], indent=2))
    print(f"Detailed report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
