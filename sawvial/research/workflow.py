"""
Online research workflow (reuses the normal ingestion pipeline):

    question -> search -> fetch (safe) -> extract (same pipeline as local PDFs)
             -> store as pending_review -> compare sources -> evidence report
    (human) verify -> publish -> dataset + atomic index rebuild -> answerable in later sessions

Rules enforced here:
  * a failed search or download is reported as failed - never papered over;
  * only source text is stored; the report's summary is extractive and is NOT saved as knowledge;
  * downloaded text is untrusted: instruction-like phrases flag the record (it cannot be
    verified) and the text only ever reaches the LLM as quoted evidence;
  * duplicates (same DOI / URL / file hash / evidence text) are skipped.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

from sawvial import grounding, settings
from sawvial.config import BuildConfig, ChunkingConfig, EmbeddingConfig, ExtractionConfig, Taxonomy
from sawvial.ingestion.pdf_loader import (ExtractionReport, PageContent, PDFLoadError, RawDocument,
                                          TextBlock, load_pdf)
from sawvial.research import safe_http
from sawvial.research.search import Candidate, SearchUnavailable, is_trusted, search_literature
from sawvial.research.store import ResearchStore

ALLOWED_DOC_TYPES = {"application/pdf", "text/html", "application/xhtml+xml", "text/plain"}


# ---------------------------------------------------------------- HTML -> text + metadata
class _Html(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg", "form", "aside"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth_skip, self.text, self.meta, self._title, self._in_title = 0, [], {}, [], False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self.depth_skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta" and a.get("name") and a.get("content"):
            self.meta.setdefault(a["name"].lower(), []).append(a["content"].strip())
        elif tag in ("p", "h1", "h2", "h3", "li", "div", "br", "tr") and not self.depth_skip:
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth_skip:
            self.depth_skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
        elif not self.depth_skip:
            self.text.append(data)


def html_to_document(html: bytes, url: str) -> tuple[RawDocument, Dict[str, Any]]:
    """Plain-text RawDocument + bibliographic metadata from <title>/citation_* meta tags."""
    p = _Html()
    p.feed(html.decode("utf-8", errors="replace"))
    lines = [re.sub(r"\s+", " ", l).strip() for l in "".join(p.text).split("\n")]
    lines = [l for l in lines if len(l) > 2]
    m = p.meta
    first = lambda k: (m.get(k) or [None])[0]
    meta = {"title": first("citation_title") or " ".join(p._title).strip() or None,
            "authors": m.get("citation_author", []), "doi": first("citation_doi"),
            "journal": first("citation_journal_title") or first("citation_publisher"),
            "year": int(re.search(r"(19|20)\d{2}", first("citation_publication_date") or first("citation_date") or "0000")
                        .group(0)) if re.search(r"(19|20)\d{2}", first("citation_publication_date") or first("citation_date") or "") else None}
    blocks = [TextBlock(text=l, page=1, font_size=10.0, bold=False, bbox=(0, i * 12.0, 500, i * 12.0 + 10))
              for i, l in enumerate(lines)]
    rep = ExtractionReport(pages_total=1, pages_with_text=1 if blocks else 0,
                           chars_total=sum(len(l) for l in lines))
    rep.warnings.append("html_page: page numbers are not available")
    return RawDocument(source_path=url, filename=url[-60:], pdf_metadata={"title": meta["title"] or ""},
                       pages=[PageContent(page_number=1, blocks=blocks)], n_pages=1, extraction=rep), meta


# ---------------------------------------------------------------- ingestion
class Pipeline:
    """Builds the extraction components once (same ones the PDF pipeline uses)."""
    def __init__(self):
        from sawvial.classification.domain_classifier import DomainClassifier
        from sawvial.extraction.context_extractor import ContextExtractor
        from sawvial.extraction.entities import ScientificEntityExtractor
        from sawvial.extraction.importance import ImportanceScorer
        root = settings.ROOT
        self.taxonomy = Taxonomy(root / "taxonomy.json")
        self.cfg = BuildConfig(taxonomy=self.taxonomy, chunking=ChunkingConfig(),
                               extraction=ExtractionConfig(), embedding=EmbeddingConfig(offline=True))
        ent = ScientificEntityExtractor(self.taxonomy)
        self.parts = (ent, ContextExtractor(self.taxonomy), ImportanceScorer(self.taxonomy),
                      DomainClassifier(self.taxonomy, ent))


def ingest_url(url: str, store: ResearchStore, pipe: Pipeline, hint: Optional[Candidate] = None,
               ocr: bool = False, allow_private: bool = False) -> Dict[str, Any]:
    """Download one document and store its evidence as pending_review. Always returns an outcome
    dict with status ok | duplicate | failed and a reason; it never raises for network problems."""
    from sawvial.pipeline import process_document
    out: Dict[str, Any] = {"url": url, "status": "failed", "reason": None, "records_added": 0}
    try:
        res = safe_http.fetch(url, allowed_types=ALLOWED_DOC_TYPES, allow_private=allow_private,
                              allowed_domains=settings.RESEARCH_ALLOWED_DOMAINS or None,
                              accept="application/pdf,text/html;q=0.9")
    except (safe_http.UnsafeURL, safe_http.FetchError) as exc:
        out["reason"] = f"{exc.__class__.__name__}: {exc}"
        return out
    dup = store.find_duplicate_source(hint.doi if hint else None, res.final_url, res.sha256)
    if dup:
        return {**out, "status": "duplicate", "reason": f"already collected ({dup['final_url']})"}

    doi_hint = hint.doi if hint else None
    try:
        if res.content_type == "application/pdf":
            store.downloads.mkdir(parents=True, exist_ok=True)
            path = store.downloads / f"{res.sha256[:16]}.pdf"
            path.write_bytes(res.content)
            raw = load_pdf(path, ocr=ocr, max_pages=settings.RESEARCH_MAX_PDF_PAGES)
            html_meta = {}
        else:
            raw, html_meta = html_to_document(res.content, res.final_url)
    except PDFLoadError as exc:
        out["reason"] = f"unreadable document: {exc}"
        return out
    if raw.extraction.status == "no_text":
        out["reason"] = "no extractable text (scanned PDF? retry with --ocr)"
        return out

    meta_override = {
        "document_id": f"web_{res.sha256[:12]}", "url": res.final_url,
        "title": (hint.title if hint else None) or html_meta.get("title"),
        "authors": (hint.authors if hint else None) or html_meta.get("authors"),
        "year": (hint.year if hint else None) or html_meta.get("year"),
        "doi": doi_hint or html_meta.get("doi"),
        "journal": (hint.venue if hint else None) or html_meta.get("journal"),
        "source_hash": res.sha256}
    doc, units = process_document(Path(raw.source_path), pipe.cfg, *pipe.parts, raw_doc=raw,
                                  meta_override=meta_override, initial_status="pending_review")
    now = datetime.now(timezone.utc).isoformat()
    trusted = is_trusted(res.final_url) or bool(hint and hint.trusted)
    for u in units:
        u["source"].update({"origin": "online_research", "retrieved_at": now, "trusted_source": trusted,
                            "final_url": res.final_url, "content_sha256": res.sha256,
                            "search_provider": hint.provider if hint else "direct_url"})
    counts = store.add_records(units)
    store.add_source({"final_url": res.final_url, "requested_url": url, "doi": meta_override["doi"],
                      "title": doc["metadata"].get("title"), "content_sha256": res.sha256,
                      "content_type": res.content_type, "bytes": len(res.content), "retrieved_at": now,
                      "trusted": trusted, "extraction": doc["metadata"]["extraction"]["status"],
                      "records": counts["added"]})
    flagged = sum(1 for u in units if u["quality"].get("flags"))
    return {**out, "status": "ok", "reason": None, "title": doc["metadata"].get("title"),
            "records_added": counts["added"], "duplicates_skipped": counts["duplicates_skipped"],
            "flagged_records": flagged, "extraction": doc["metadata"]["extraction"]["status"],
            "warnings": doc["metadata"]["extraction"]["warnings"], "trusted_source": trusted}


# ---------------------------------------------------------------- compare + report
def compare_sources(question: str, records: List[Dict[str, Any]], max_sources: int = 5) -> Dict[str, Any]:
    """Extractive comparison of the collected evidence (no LLM, nothing generated or saved)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not records:
        return {"findings": [], "contradictions": [], "gaps": ["no evidence records were collected"]}
    texts = [r["knowledge"]["evidence"] for r in records]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(texts + [question])
    sims = (mat[:-1] @ mat[-1].T).toarray().ravel()
    best_per_doc: Dict[str, int] = {}
    for i in sims.argsort()[::-1]:
        doc = records[i]["source"]["document_id"]
        if sims[i] > 0 and doc not in best_per_doc and len(best_per_doc) < max_sources:
            best_per_doc[doc] = int(i)
    findings = []
    for doc, i in best_per_doc.items():
        r = records[i]
        p = {"id": r["id"], "text": r["knowledge"]["evidence"], "document_id": doc}
        ex = grounding.extractive_answer(question, [p])
        findings.append({"record_id": r["id"], "title": r["source"].get("title"), "year": r["source"].get("year"),
                         "doi": r["source"].get("doi"), "url": r["source"].get("final_url") or r["source"].get("url"),
                         "page": r["provenance"].get("page"), "trusted_source": r["source"].get("trusted_source"),
                         "quote": ex[0].split(": ", 1)[-1] if ex else r["knowledge"]["evidence"][:300],
                         "effects": [(f.get("parameter"), f.get("effect"), f.get("magnitude"), f.get("unit"))
                                     for f in r.get("structured_facts", []) if f.get("fact_type") == "effect"]})
    contradictions = []
    for a in range(len(findings)):
        for b in range(a + 1, len(findings)):
            for pa, ea, *_ in findings[a]["effects"]:
                for pb, eb, *_ in findings[b]["effects"]:
                    if pa and pa == pb and {ea, eb} == {"increase", "decrease"}:
                        contradictions.append({"parameter": pa, "records": [findings[a]["record_id"], findings[b]["record_id"]],
                                               "note": "opposite effect directions reported; automatic flag - a person must check "
                                                       "whether crops/conditions differ"})
    gaps = []
    if len(findings) < 2:
        gaps.append("fewer than two independent sources - nothing to compare")
    if findings and not any(f["trusted_source"] for f in findings):
        gaps.append("none of the sources is from the preferred official/publisher domains")
    if not findings:
        gaps.append("no collected passage matches the question")
    return {"findings": findings, "contradictions": contradictions, "gaps": gaps}


def research(question: str, store: ResearchStore, pipe: Pipeline, max_results: int = 5,
             provider: Optional[str] = None, ocr: bool = False) -> Dict[str, Any]:
    report: Dict[str, Any] = {"question": question, "started": datetime.now(timezone.utc).isoformat(),
                              "provider": provider or settings.SEARCH_PROVIDER, "search": {}, "fetched": [],
                              "note": "Evidence is stored as pending_review. The summary below is extractive "
                                      "(quoted source text), not verified knowledge."}
    try:
        cands = search_literature(question, provider, max_results=max_results * 3)
        report["search"] = {"status": "ok", "candidates": len(cands)}
    except SearchUnavailable as exc:
        report["search"] = {"status": "failed", "reason": str(exc)}
        report["gaps"] = ["search failed - nothing was fetched or stored"]
        return _save(store, report)
    new_ids: List[str] = []
    tried = 0
    for c in cands:
        if tried >= max_results:
            break
        target = c.pdf_url or c.landing_url
        if not target:
            report["fetched"].append({"title": c.title, "status": "failed", "reason": "no accessible URL"})
            continue
        tried += 1
        outcome = ingest_url(target, store, pipe, hint=c, ocr=ocr)
        outcome["title"] = outcome.get("title") or c.title
        report["fetched"].append(outcome)
    # compare over everything collected by this run (+ earlier pending/verified records on the same docs)
    urls = {o["url"] for o in report["fetched"] if o["status"] in ("ok", "duplicate")}
    recs = [r for r in store.records()
            if r["quality"]["validation_status"] != "rejected"
            and (r["source"].get("final_url") in urls or r["source"].get("url") in urls)]
    report.update(compare_sources(question, recs))
    return _save(store, report)


def _save(store: ResearchStore, report: Dict[str, Any]) -> Dict[str, Any]:
    store.reports.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"
    (store.reports / name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_file"] = str(store.reports / name)
    return report


# ---------------------------------------------------------------- publish
def publish_verified(store: ResearchStore, dataset_dir: Path, index_dir: Path, pipe: Pipeline) -> Dict[str, Any]:
    """Merge human-verified records into the dataset and atomically rebuild the index."""
    from sawvial.dataset.writer import merge_into_dataset
    from sawvial.pipeline import run_embed_and_index
    units = store.publishable()
    if not units:
        return {"published": 0, "message": "no verified, unpublished records"}
    docs_by_id = {}
    for u in units:
        docs_by_id.setdefault(u["source"]["document_id"], {
            "document_id": u["source"]["document_id"], "full_text": "", "n_candidate_chunks": 0,
            "metadata": {k: u["source"].get(k) for k in ("title", "authors", "year", "doi", "url", "journal", "publisher",
                                                          "document_type")}
                        | {"source_hash": u["source"].get("content_sha256")}})
    for u in units:
        u["quality"]["forced_status"] = "verified"
    res = merge_into_dataset(str(dataset_dir), list(docs_by_id.values()), units, pipe.taxonomy,
                             pipe.cfg.extraction, id_prefix="SWR", preserve_ids=True)
    store.mark_published({i: i for i in res["new_unit_ids"]})
    cfg = pipe.cfg
    cfg.output_dir, cfg.index_dir = str(dataset_dir), str(index_dir)
    if res["units_added"]:
        run_embed_and_index(cfg)
    skipped = len(units) - res["units_added"]
    return {"published": res["units_added"], "skipped_as_duplicates": skipped,
            "index_rebuilt": bool(res["units_added"])}
