"""
Live research used by /api/ask when the local knowledge base cannot answer.

    discover (OpenAlex / Crossref / optional web provider)  ->  fetch + extract the best few documents
    -> keep them as pending_review in the research store (NOT verified, NOT in the answerable index)
    -> rank their passages for THIS question with the same retrieval/crop-gating code as local evidence.

Everything returns a report (never raises for network problems) so the API can tell the user honestly what
happened: search failed, nothing relevant was found, or which sources were read.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from sawvial import settings
from sawvial.index.passages import retrieve_passages
from sawvial.index.retriever import bundle_from_units, detect_crops
from sawvial.research.search import SearchUnavailable, discover
from sawvial.research.store import ResearchStore
from sawvial.research import workflow

LIVE_STATUSES = frozenset({"pending_review", "verified"})


def live_research(question_en: str, top_k: int, store: Optional[ResearchStore] = None,
                  progress: Optional[Callable[..., None]] = None, cancel_event=None) -> Dict[str, Any]:
    """-> {"report": {...}, "passages": [...], "crop_missing": bool}"""
    emit = progress or (lambda *a, **k: None)
    report: Dict[str, Any] = {"attempted": True, "status": "search_failed", "providers": [], "candidates": 0,
                              "documents": [], "failures": [], "queries_for": question_en}
    store = store or ResearchStore()
    if not settings.ONLINE_RESEARCH:
        report.update(attempted=False, status="disabled")
        return {"report": report, "passages": [], "crop_missing": False}
    crops = detect_crops(question_en)
    emit("online_search")
    try:
        cands, providers = discover(question_en, crops, max_results=settings.RESEARCH_MAX_DOCS * 4)
    except SearchUnavailable as exc:
        report.update(status="search_failed", error=str(exc))
        return {"report": report, "passages": [], "crop_missing": False}
    report.update(providers=providers, candidates=len(cands))
    if not cands:
        report["status"] = "no_results"
        return {"report": report, "passages": [], "crop_missing": False}

    pipe = workflow.get_pipeline()
    deadline = time.monotonic() + settings.RESEARCH_TIME_BUDGET_S
    urls: List[str] = []
    attempts = 0
    emit("reading", n=min(len(cands), settings.RESEARCH_MAX_DOCS))
    for c in cands:
        if len(urls) >= settings.RESEARCH_MAX_DOCS or attempts >= settings.RESEARCH_MAX_DOCS * 3:
            break
        if time.monotonic() > deadline:
            report["failures"].append({"url": c.target, "reason": "research time budget exhausted"})
            break
        if cancel_event is not None and cancel_event.is_set():
            break
        attempts += 1
        o = workflow.ingest_url(c.target, store, pipe, hint=c)
        entry = {"url": o.get("final_url") or c.target, "title": o.get("title") or c.title, "status": o["status"],
                 "tier": o.get("tier"), "source_type": o.get("source_type"), "reason": o.get("reason")}
        report["documents"].append(entry)
        if o["status"] in ("ok", "cached", "duplicate") and o.get("source_url"):
            urls.append(o["source_url"])
        else:
            report["failures"].append({"url": c.target, "reason": o.get("reason")})

    records = store.records_for_urls(urls)
    bundle = bundle_from_units(records, LIVE_STATUSES, origin="web")
    if bundle is None:
        report["status"] = "no_usable_documents" if report["documents"] else "no_results"
        return {"report": report, "passages": [], "crop_missing": bool(crops)}
    found = retrieve_passages(bundle, question_en, top_k)
    passages = [p for p in found["passages"] if float(p["score"]) >= settings.MIN_RETRIEVAL_SCORE]
    report["status"] = "used" if passages else "no_relevant_passages"
    report["sources_used"] = sorted({p.get("document_id") for p in passages if p.get("document_id")})
    return {"report": report, "passages": passages, "crop_missing": found["crop_missing"] and not passages,
            "records_collected": len(records)}
