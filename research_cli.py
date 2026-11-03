#!/usr/bin/env python3
"""SAWVIAL online research workflow (command line).

    python research_cli.py search "which soil is good for tomatoes"                # search + fetch + compare
    python research_cli.py url https://example.org/paper.pdf                       # ingest one document
    python research_cli.py list [--status pending_review]                          # see collected records
    python research_cli.py show SWR_000001                                         # read one record
    python research_cli.py verify SWR_000001 SWR_000002 --reviewer "Your Name"     # human approval
    python research_cli.py reject SWR_000003 --reviewer "Your Name" --note "wrong crop"
    python research_cli.py publish                                                 # verified -> dataset + index

New records are ALWAYS stored as pending_review; only `verify` + `publish` make them answerable.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sawvial import settings
from sawvial.research.store import ResearchStore


def _print_outcomes(rows):
    for o in rows:
        mark = {"ok": "OK  ", "duplicate": "DUP ", "cached": "HIT ", "failed": "FAIL"}.get(o["status"], o["status"])
        extra = f"+{o.get('records_added', 0)} records" if o["status"] == "ok" else (o.get("reason") or "")
        print(f"  [{mark}] {(o.get('title') or o['url'])[:80]}  {extra}")
        for w in o.get("warnings", []):
            print(f"         warning: {w}")
        if o.get("flagged_records"):
            print(f"         {o['flagged_records']} record(s) flagged for instruction-like text (cannot be verified)")


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("question"); s.add_argument("--max", type=int, default=5)
    s.add_argument("--provider", choices=["auto", "openalex", "crossref", "none"]); s.add_argument("--ocr", action="store_true")
    u = sub.add_parser("url"); u.add_argument("url"); u.add_argument("--ocr", action="store_true")
    l = sub.add_parser("list"); l.add_argument("--status")
    sh = sub.add_parser("show"); sh.add_argument("id")
    for name in ("verify", "reject"):
        v = sub.add_parser(name); v.add_argument("ids", nargs="+"); v.add_argument("--reviewer", required=True)
        v.add_argument("--note", default="")
    sub.add_parser("publish")
    a = ap.parse_args()
    store = ResearchStore()

    if a.cmd in ("search", "url", "publish"):
        from sawvial.research import workflow
        pipe = workflow.get_pipeline()
    if a.cmd == "search":
        rep = workflow.research(a.question, store, pipe, max_results=a.max, provider=a.provider, ocr=a.ocr)
        if rep["search"]["status"] != "ok":
            print(f"SEARCH FAILED: {rep['search']['reason']}\nNothing was fetched or stored.")
            sys.exit(2)
        for p in rep["search"].get("providers", []):
            print(f"  provider {p['provider']:<9} {p['status']:<14} results={p['results']}" + (f"  ({p['error']})" if p.get("error") else ""))
        print(f"Search ok: {rep['search']['candidates']} candidates; fetch results:")
        _print_outcomes(rep["fetched"])
        print("\nEvidence found (extractive quotes; NOT verified):")
        for f in rep["findings"]:
            print(f"  [{f['record_id']}] {f['title']} ({f['year']}) p.{f['page']} {f['doi'] or f['url'] or ''}\n      \"{f['quote'][:300]}\"")
        for c in rep["contradictions"]:
            print(f"  ! possible contradiction on '{c['parameter']}': {c['records']} - {c['note']}")
        for g in rep["gaps"]:
            print(f"  - gap: {g}")
        print(f"\nReport saved: {rep['report_file']}\nReview with: python research_cli.py list --status pending_review")
    elif a.cmd == "url":
        # you chose this URL yourself, so the trust-tier filter is off; the tier is still recorded on the record
        o = workflow.ingest_url(a.url, store, pipe, ocr=a.ocr, min_trust="any")
        _print_outcomes([o])
        sys.exit(0 if o["status"] in ("ok", "duplicate", "cached") else 2)
    elif a.cmd == "list":
        for r in store.records(a.status):
            print(f"{r['id']}  {r['quality']['validation_status']:<14} {(r['source'].get('title') or '')[:55]:<55} "
                  f"p.{r['provenance'].get('page')}  {'FLAGGED' if r['quality'].get('flags') else ''}")
    elif a.cmd == "show":
        match = [r for r in store.records() if r["id"] == a.id]
        print(json.dumps(match[0], indent=2, ensure_ascii=False) if match else f"unknown id {a.id}")
    elif a.cmd in ("verify", "reject"):
        try:
            changed = store.set_status(a.ids, "verified" if a.cmd == "verify" else "rejected", a.reviewer, a.note)
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")
        print(f"{a.cmd}: {len(changed)} record(s) updated. Run `python research_cli.py publish` to make verified records searchable.")
    elif a.cmd == "publish":
        res = workflow.publish_verified(store, settings.DATASET_DIR, settings.INDEX_DIR, pipe)
        print(json.dumps(res, indent=2))
        sys.exit(1 if res.get("rolled_back") else 0)


if __name__ == "__main__":
    main()
