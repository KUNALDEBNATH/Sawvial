"""
Compare evidence passages from different sources for the current question (no LLM, nothing saved).

* groups passages by source document and labels each with its kind of statement
  (official guidance / research paper / informational article);
* finds numeric ranges that the sources state for the SAME quantity (the quantity name is simply the word
  next to the number in the text - nothing is hard-coded) and reports when sources give different values.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List

_RANGE = re.compile(r"(?P<label>[A-Za-z][A-Za-z]{1,24})\s*(?:of|between|from|at|is|was|should be|:|=|-)?\s*"
                    r"(?P<lo>\d{1,4}(?:\.\d+)?)\s*(?:-|–|—|to)\s*(?P<hi>\d{1,4}(?:\.\d+)?)\s*(?P<unit>%|°C|cm|mm|m|kg/ha|t/ha|ds/m)?")
_NOT_A_LABEL = {"of", "to", "and", "from", "between", "at", "is", "the", "in", "by", "for", "range", "ranging", "about",
                "around", "approximately", "over", "under", "than", "was", "are", "were", "with", "per", "page", "pp",
                "table", "fig", "figure", "up", "down", "days", "day", "weeks", "months", "year", "years"}
TYPE_LABELS = {"official_guidance": "official guidance", "research_paper": "research paper",
               "informational_article": "informational article"}


def _ranges(text: str) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    for m in _RANGE.finditer(text):
        label = m.group("label").lower()
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        if label in _NOT_A_LABEL or lo >= hi or hi - lo > 10000:
            continue
        out.setdefault(label, (lo, hi, m.group("unit") or ""))
    return out


def compare_passages(passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_doc: Dict[str, Dict[str, Any]] = {}
    for p in passages:
        doc = p.get("document_id") or p["id"]
        d = by_doc.setdefault(doc, {"document_id": doc, "title": p.get("title"), "url": p.get("url"),
                                    "source_type": p.get("source_type") or "research_paper",
                                    "trust_tier": p.get("trust_tier"), "ids": [], "ranges": {}})
        d["ids"].append(p["id"])
        for label, rng in _ranges(p.get("text") or "").items():
            d["ranges"].setdefault(label, rng)
    labelled: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in by_doc.values():
        for label, (lo, hi, unit) in d["ranges"].items():
            labelled[label].append({"title": d["title"], "source_type": d["source_type"], "low": lo, "high": hi,
                                    "unit": unit, "document_id": d["document_id"]})
    differences = []
    for label, vals in labelled.items():
        if len({v["document_id"] for v in vals}) < 2:
            continue
        if len({(v["low"], v["high"]) for v in vals}) > 1:
            differences.append({"quantity": label, "values": vals})
    types = sorted({d["source_type"] for d in by_doc.values()})
    return {"n_sources": len(by_doc), "source_types": types, "differences": differences,
            "single_source": len(by_doc) < 2,
            "sources": [{k: d[k] for k in ("document_id", "title", "url", "source_type", "trust_tier", "ids")}
                        for d in by_doc.values()]}
