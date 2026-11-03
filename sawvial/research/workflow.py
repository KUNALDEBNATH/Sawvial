"""
Research workflow building blocks (shared by the chat API and research_cli.py):

    fetch (safe, polite) -> extract (PDF or HTML, no pretending) -> same pipeline as local PDFs
    -> store as pending_review  ...  (human) verify -> publish -> dataset + atomic index rebuild

Rules enforced here:
  * a failed search/download is reported as failed - never papered over;
  * trust is decided from the FINAL url after redirects (never from the search provider or doi.org);
  * robots.txt and a per-host delay are respected; paywalls/logins are not bypassed;
  * downloaded text is untrusted: instruction-like phrases flag the record (it cannot be verified) and the text
    only ever reaches the LLM as quoted evidence;
  * only source text is stored - generated answers are never saved as knowledge;
  * duplicates (same URL / DOI / file hash / evidence text) are skipped and unchanged documents are re-used.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from sawvial import settings
from sawvial.config import BuildConfig, ChunkingConfig, EmbeddingConfig, ExtractionConfig, Taxonomy
from sawvial.ingestion.pdf_loader import (ExtractionReport, PageContent, PDFLoadError, RawDocument,
                                          TextBlock, load_pdf)
from sawvial.language import script_counts
from sawvial.research import safe_http
from sawvial.research.search import (Candidate, SearchUnavailable, classify_source, discover, tier_allowed,
                                     trust_tier)
from sawvial.research.store import ResearchStore

ALLOWED_DOC_TYPES = {"application/pdf", "text/html", "application/xhtml+xml", "text/plain"}
MIN_PAGE_CHARS = 400            # less readable text than this cannot support an answer
ROBOTS_AGENT = "SAWVIAL-research"


# ---------------------------------------------------------------- HTML -> clean text + metadata
_VOID = {"br", "hr", "img", "input", "meta", "link", "source", "wbr", "area", "base", "col", "embed", "param", "track"}
_NOISE_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "iframe",
               "button", "select", "template", "dialog", "head"}
_NOISE_ROLES = {"navigation", "banner", "contentinfo", "complementary", "search"}
_NOISE_ATTR = re.compile(r"(?:^|[\s_-])(nav|navbar|menu|footer|cookie|cookies|consent|advert|advertisement|banner|"
                         r"breadcrumb|breadcrumbs|social|share|newsletter|popup|modal|comments|related)(?:$|[\s_-])", re.I)
_BLOCK = {"p", "li", "div", "section", "blockquote", "pre", "tr", "table", "ul", "ol", "dd", "dt", "figcaption"}


def decode_html(content: bytes, declared: Optional[str] = None) -> str:
    """Decode with the declared charset, then <meta charset>, then detection - not blindly as UTF-8."""
    tried = []
    head = content[:4096].decode("ascii", errors="ignore")
    m = re.search(r"<meta[^>]+charset=[\"']?([\w-]+)", head, re.I)
    for enc in (declared, m.group(1) if m else None, "utf-8"):
        if enc and enc.lower() not in tried:
            tried.append(enc.lower())
            try:
                return content.decode(enc)
            except (LookupError, UnicodeDecodeError):
                continue
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(content).best()
        if best is not None:
            return str(best)
    except ImportError:
        pass
    return content.decode("utf-8", errors="replace")


class _Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip: List[str] = []
        self.main_stack: List[str] = []
        self.meta: Dict[str, List[str]] = {}
        self.title_parts: List[str] = []
        self.in_title = False
        self.buf: List[str] = []
        self.kind = "p"
        self.row: Optional[List[str]] = None
        self.all: List[tuple] = []
        self.main: List[tuple] = []

    # -- output helpers
    def _emit(self, kind: str, text: str, min_len: Optional[int] = None):
        text = " ".join(text.split())
        if len(text) < (min_len or (3 if kind == "h" else 20)) or (self.all and self.all[-1][1] == text):
            return
        self.all.append((kind, text))
        if self.main_stack:
            self.main.append((kind, text))

    def _flush(self):
        text = "".join(self.buf)
        self.buf = []
        if self.row is not None:
            if text.strip():
                self.row.append(" ".join(text.split()))
        elif text.strip():
            self._emit(self.kind, text)
        self.kind = "p"

    # -- parser callbacks
    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("name") or a.get("property") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, []).append(a["content"].strip())
            return
        if tag == "title" and not self.skip:
            self.in_title = True
        if self.skip:
            if tag not in _VOID:
                self.skip.append(tag)
            return
        noisy = (tag in _NOISE_TAGS and tag != "head") or a.get("role", "").lower() in _NOISE_ROLES or (
            tag not in ("html", "body", "article", "main", "table", "tr", "td")
            and bool(_NOISE_ATTR.search(f"{a.get('class', '')} {a.get('id', '')}")))
        if tag == "head":
            noisy = False
        if noisy and tag not in _VOID:
            self._flush()
            self.skip = [tag]
            return
        if tag in ("article", "main") or a.get("role", "").lower() == "main":
            self._flush()
            self.main_stack.append(tag)
        elif tag in ("h1", "h2", "h3", "h4"):
            self._flush()
            self.kind = "h"
        elif tag == "tr":
            self._flush()
            self.row = []
        elif tag in ("td", "th"):
            self._flush()
        elif tag in _BLOCK or tag == "br":
            self._flush()

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if self.skip:
            if tag in self.skip:
                while self.skip and self.skip.pop() != tag:
                    pass
            return
        if tag in ("article", "main") and self.main_stack:
            self._flush()
            self.main_stack.pop()
        elif tag == "tr":
            self._flush()
            if self.row:
                row, self.row = self.row, None
                self._emit("p", " | ".join(row), min_len=5)     # table rows are short but carry the data
            self.row = None
        elif tag in ("td", "th") or tag in ("h1", "h2", "h3", "h4") or tag in _BLOCK:
            self._flush()

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        elif not self.skip:
            self.buf.append(data)

    def close(self):
        super().close()
        self._flush()


def html_to_document(content: bytes, url: str, charset: Optional[str] = None) -> tuple[RawDocument, Dict[str, Any]]:
    """Readable article text (navigation/ads/menus removed) as a RawDocument + bibliographic metadata."""
    p = _Page()
    p.feed(decode_html(content, charset))
    p.close()
    main_chars = sum(len(t) for _, t in p.main)
    lines = p.main if main_chars >= 500 else p.all
    m, first = p.meta, (lambda *keys: next((m[k][0] for k in keys if m.get(k)), None))
    date = first("citation_publication_date", "citation_date", "article:published_time", "dc.date", "date")
    ym = re.search(r"(?:19|20)\d{2}", date or "")
    doi = first("citation_doi", "dc.identifier", "prism.doi")
    meta = {"title": first("citation_title", "og:title") or " ".join(" ".join(p.title_parts).split()) or None,
            "authors": m.get("citation_author") or m.get("author") or [],
            "doi": (re.search(r"10\.\d{4,9}/\S+", doi or "") or [None])[0] if doi else None,
            "journal": first("citation_journal_title", "og:site_name", "citation_publisher"),
            "year": int(ym.group(0)) if ym else None, "published": (date or "")[:25] or None}
    blocks = []
    for i, (kind, text) in enumerate(lines):
        y = i * 26.0                                # one paragraph per source block (gap > 0.6 x line height)
        blocks.append(TextBlock(text=text, page=1, font_size=14.0 if kind == "h" else 10.0, bold=(kind == "h"),
                                bbox=(0, y, 500, y + 10)))
    chars = sum(len(b.text) for b in blocks)
    rep = ExtractionReport(pages_total=1, pages_with_text=1 if blocks else 0, chars_total=chars)
    rep.warnings.append("html_page: page numbers are not available")
    return RawDocument(source_path=url, filename=url[-60:], pdf_metadata={"title": meta["title"] or ""},
                       pages=[PageContent(page_number=1, blocks=blocks)], n_pages=1, extraction=rep), meta


# ---------------------------------------------------------------- politeness: robots.txt + per-host delay
_ROBOTS: Dict[str, tuple] = {}
_LAST_HIT: Dict[str, float] = {}
_LOCK = threading.Lock()


def robots_allowed(url: str, allow_private: bool = False) -> tuple[bool, str]:
    if not settings.RESPECT_ROBOTS:
        return True, ""
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    now = time.time()
    with _LOCK:
        cached = _ROBOTS.get(base)
    if cached and now - cached[0] < 3600:
        parser, why = cached[1], cached[2]
    else:
        parser, why = None, ""
        try:
            res = safe_http.fetch(base + "/robots.txt", allowed_types={"text/plain", "text/html", "application/octet-stream"},
                                  max_bytes=512 * 1024, timeout=10, allow_private=allow_private)
            parser = RobotFileParser()
            parser.parse(res.content.decode("utf-8", errors="replace").splitlines())
        except safe_http.FetchError as exc:
            if exc.status and exc.status >= 500:
                parser, why = False, f"robots.txt unavailable (HTTP {exc.status}); not crawling"
            # 4xx = no robots.txt = allowed; other problems = cannot parse = allowed
        except safe_http.UnsafeURL as exc:
            parser, why = False, str(exc)
        with _LOCK:
            _ROBOTS[base] = (now, parser, why)
    if parser is False:
        return False, why
    if parser is None:
        return True, ""
    return (True, "") if parser.can_fetch(ROBOTS_AGENT, url) else (False, "disallowed by robots.txt")


def throttle(host: str) -> None:
    delay = settings.RESEARCH_HOST_DELAY_S
    if delay <= 0:
        return
    with _LOCK:
        now = time.time()
        wait = max(0.0, _LAST_HIT.get(host, 0.0) + delay - now)
        _LAST_HIT[host] = now + wait
    if wait:
        time.sleep(min(wait, 5.0))


# ---------------------------------------------------------------- ingestion
class Pipeline:
    """Builds the extraction components once (the same ones the PDF pipeline uses)."""
    def __init__(self):
        from sawvial.classification.domain_classifier import DomainClassifier
        from sawvial.extraction.context_extractor import ContextExtractor
        from sawvial.extraction.entities import ScientificEntityExtractor
        from sawvial.extraction.importance import ImportanceScorer
        self.taxonomy = Taxonomy(settings.ROOT / "taxonomy.json")
        self.cfg = BuildConfig(taxonomy=self.taxonomy, chunking=ChunkingConfig(), extraction=ExtractionConfig(),
                               embedding=EmbeddingConfig(offline=True))
        ent = ScientificEntityExtractor(self.taxonomy)
        self.parts = (ent, ContextExtractor(self.taxonomy), ImportanceScorer(self.taxonomy),
                      DomainClassifier(self.taxonomy, ent))


_PIPE: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _PIPE
    with _LOCK:
        if _PIPE is None:
            _PIPE = Pipeline()
        return _PIPE


def _latin_share(raw: RawDocument) -> float:
    c = script_counts(" ".join(b.text for p in raw.pages[:3] for b in p.blocks)[:6000])
    total = sum(c.values())
    return c["latin"] / total if total else 1.0


def ingest_url(url: str, store: ResearchStore, pipe: Pipeline, hint: Optional[Candidate] = None,
               ocr: bool = False, allow_private: bool = False, min_trust: Optional[str] = None) -> Dict[str, Any]:
    """Download ONE document and store its evidence as pending_review. Never raises for network problems;
    returns {status: ok|cached|duplicate|failed, reason, source_url, ...}."""
    from sawvial.pipeline import process_document
    out: Dict[str, Any] = {"url": url, "status": "failed", "reason": None, "records_added": 0, "source_url": None}
    min_trust = min_trust or settings.RESEARCH_MIN_TRUST

    cached = store.find_cached_source(url, settings.RESEARCH_CACHE_DAYS)
    if cached:                                                       # unchanged document: no second download
        tier = cached.get("trust_tier") or trust_tier(cached.get("final_url"))
        if not tier_allowed(tier, min_trust):
            return {**out, "reason": f"cached source is from an untrusted domain ({tier})"}
        return {**out, "status": "cached", "title": cached.get("title"), "tier": tier,
                "source_type": cached.get("source_type"), "source_url": cached["final_url"],
                "records_added": 0, "reason": None}

    host = (urlparse(url).hostname or "").lower()
    if not allow_private:
        ok, why = robots_allowed(url)
        if not ok:
            return {**out, "reason": why}
        throttle(host)
    try:
        res = safe_http.fetch(url, allowed_types=ALLOWED_DOC_TYPES, allow_private=allow_private,
                              allowed_domains=settings.RESEARCH_ALLOWED_DOMAINS or None,
                              accept="application/pdf,text/html;q=0.9")
    except (safe_http.UnsafeURL, safe_http.FetchError) as exc:
        return {**out, "reason": f"{exc.__class__.__name__}: {exc}"}

    # Trust is judged on where we ACTUALLY ended up - a trusted search result or a doi.org link proves nothing.
    tier = trust_tier(res.final_url)
    if not tier_allowed(tier, min_trust):
        moved = f" (redirected from {urlparse(url).hostname})" if urlparse(res.final_url).hostname != host else ""
        return {**out, "tier": tier, "final_url": res.final_url,
                "reason": f"final site {urlparse(res.final_url).hostname}{moved} is not a trusted source "
                          f"(tier '{tier}', required '{min_trust}')"}
    dup = store.find_duplicate_source(hint.doi if hint else None, res.final_url, res.sha256)
    if dup:
        return {**out, "status": "duplicate", "reason": f"already collected ({dup['final_url']})",
                "source_url": dup["final_url"], "tier": tier, "title": dup.get("title")}

    try:
        if res.content_type == "application/pdf":
            store.downloads.mkdir(parents=True, exist_ok=True)
            path = store.downloads / f"{res.sha256[:16]}.pdf"
            path.write_bytes(res.content)
            raw, html_meta = load_pdf(path, ocr=ocr, max_pages=settings.RESEARCH_MAX_PDF_PAGES), {}
        elif res.content_type == "text/plain":
            text = decode_html(res.content, res.charset)
            raw, html_meta = html_to_document(f"<article>{''.join(f'<p>{l}</p>' for l in text.splitlines())}</article>".encode(),
                                              res.final_url)
        else:
            raw, html_meta = html_to_document(res.content, res.final_url, res.charset)
    except PDFLoadError as exc:
        return {**out, "reason": f"unreadable document: {exc}"}
    if raw.extraction.status == "no_text" and res.content_type == "application/pdf":
        return {**out, "reason": "no extractable text (scanned PDF? retry with --ocr)"}
    if raw.extraction.chars_total < MIN_PAGE_CHARS:
        return {**out, "reason": f"too little readable text ({raw.extraction.chars_total} characters)"}
    if _latin_share(raw) < 0.5:
        return {**out, "reason": "page is not in English (non-English sources are not supported yet)"}

    doi = (hint.doi if hint else None) or html_meta.get("doi")
    provider = hint.provider if hint else "direct_url"
    source_type = classify_source(res.final_url, doi, provider, res.content_type)
    meta_override = {
        "document_id": f"web_{res.sha256[:12]}", "url": res.final_url,
        "title": (hint.title if hint and hint.provider in ("openalex", "crossref") else None) or html_meta.get("title")
                 or (hint.title if hint else None),
        "authors": (hint.authors if hint else None) or html_meta.get("authors"),
        "year": (hint.year if hint else None) or html_meta.get("year"),
        "doi": doi, "journal": (hint.venue if hint else None) or html_meta.get("journal"),
        "source_hash": res.sha256}
    doc, units = process_document(Path(raw.source_path), pipe.cfg, *pipe.parts, raw_doc=raw,
                                  meta_override=meta_override, initial_status="pending_review")
    now = datetime.now(timezone.utc).isoformat()
    for u in units:
        u["source"].update({"origin": "online_research", "retrieved_at": now, "trusted_source": True,
                            "trust_tier": tier, "source_type": source_type, "final_url": res.final_url,
                            "content_sha256": res.sha256, "search_provider": provider,
                            "published": html_meta.get("published")})
    counts = store.add_records(units)
    store.add_source({"final_url": res.final_url, "requested_url": url, "doi": doi,
                      "title": doc["metadata"].get("title"), "content_sha256": res.sha256,
                      "content_type": res.content_type, "bytes": len(res.content), "retrieved_at": now,
                      "trust_tier": tier, "source_type": source_type,
                      "extraction": doc["metadata"]["extraction"]["status"], "records": counts["added"]})
    flagged = sum(1 for u in units if u["quality"].get("flags"))
    return {**out, "status": "ok", "reason": None, "title": doc["metadata"].get("title"), "tier": tier,
            "source_type": source_type, "source_url": res.final_url, "final_url": res.final_url,
            "records_added": counts["added"], "duplicates_skipped": counts["duplicates_skipped"],
            "flagged_records": flagged, "extraction": doc["metadata"]["extraction"]["status"],
            "warnings": doc["metadata"]["extraction"]["warnings"]}


# ---------------------------------------------------------------- CLI-style compare + research run
def compare_sources(question: str, records: List[Dict[str, Any]], max_sources: int = 5) -> Dict[str, Any]:
    """Extractive per-source comparison of collected records (no LLM; nothing generated or saved)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sawvial import grounding
    if not records:
        return {"findings": [], "contradictions": [], "gaps": ["no evidence records were collected"]}
    texts = [r["knowledge"]["evidence"] for r in records]
    mat = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts + [question])
    sims = (mat[:-1] @ mat[-1].T).toarray().ravel()
    best: Dict[str, int] = {}
    for i in sims.argsort()[::-1]:
        doc = records[i]["source"]["document_id"]
        if sims[i] > 0 and doc not in best and len(best) < max_sources:
            best[doc] = int(i)
    findings = []
    for doc, i in best.items():
        r = records[i]
        ex = grounding.extractive_answer(question, [{"id": r["id"], "text": r["knowledge"]["evidence"], "document_id": doc}])
        findings.append({"record_id": r["id"], "title": r["source"].get("title"), "year": r["source"].get("year"),
                         "doi": r["source"].get("doi"), "url": r["source"].get("final_url") or r["source"].get("url"),
                         "page": r["provenance"].get("page"), "trust_tier": r["source"].get("trust_tier"),
                         "source_type": r["source"].get("source_type"),
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
                                               "note": "opposite effect directions reported; automatic flag - a person must "
                                                       "check whether crops/conditions differ"})
    gaps = []
    if len(findings) < 2:
        gaps.append("fewer than two independent sources - nothing to compare")
    if not findings:
        gaps.append("no collected passage matches the question")
    return {"findings": findings, "contradictions": contradictions, "gaps": gaps}


def research(question: str, store: ResearchStore, pipe: Pipeline, max_results: int = 5,
             provider: Optional[str] = None, ocr: bool = False) -> Dict[str, Any]:
    """CLI research run: discover -> fetch -> store (pending_review) -> compare."""
    from sawvial.index.retriever import detect_crops
    report: Dict[str, Any] = {"question": question, "started": datetime.now(timezone.utc).isoformat(),
                              "provider": provider or settings.SEARCH_PROVIDER, "search": {}, "fetched": [],
                              "note": "Evidence is stored as pending_review. The summary below is extractive "
                                      "(quoted source text), not verified knowledge."}
    try:
        cands, prov = discover(question, detect_crops(question), max_results * 3, provider)
        report["search"] = {"status": "ok", "candidates": len(cands), "providers": prov}
    except SearchUnavailable as exc:
        report["search"] = {"status": "failed", "reason": str(exc)}
        report["gaps"] = ["search failed - nothing was fetched or stored"]
        return _save(store, report)
    tried = 0
    for c in cands:
        if tried >= max_results:
            break
        tried += 1
        outcome = ingest_url(c.target, store, pipe, hint=c, ocr=ocr)
        outcome["title"] = outcome.get("title") or c.title
        report["fetched"].append(outcome)
    urls = [o["source_url"] for o in report["fetched"] if o.get("source_url")]
    report.update(compare_sources(question, store.records_for_urls(urls)))
    return _save(store, report)


def _save(store: ResearchStore, report: Dict[str, Any]) -> Dict[str, Any]:
    store.reports.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"
    (store.reports / name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_file"] = str(store.reports / name)
    return report


# ---------------------------------------------------------------- publish (transactional)
_DATASET_FILES = ("documents.jsonl", "knowledge_units.jsonl", "qa.jsonl", "dataset_statistics.json")


def publish_verified(store: ResearchStore, dataset_dir: Path, index_dir: Path, pipe: Pipeline) -> Dict[str, Any]:
    """Merge human-verified records into the dataset and rebuild the index - as ONE transaction.

    1. snapshot the dataset files   2. merge (atomic file writes)   3. rebuild the index (atomic swap)
    4. ONLY THEN mark the records as published.
    If step 2 or 3 fails, the dataset snapshot is restored, the previous index stays in place and the records
    stay `verified` and unpublished, so nothing is ever marked published but unsearchable. Downloaded research
    files and the research store are never touched."""
    from sawvial.dataset.writer import merge_into_dataset
    from sawvial.pipeline import run_embed_and_index
    units = store.publishable()
    if not units:
        return {"published": 0, "message": "no verified, unpublished records"}
    dataset_dir, index_dir = Path(dataset_dir), Path(index_dir)
    docs_by_id: Dict[str, Any] = {}
    for u in units:
        docs_by_id.setdefault(u["source"]["document_id"], {
            "document_id": u["source"]["document_id"], "full_text": "", "n_candidate_chunks": 0,
            "metadata": {k: u["source"].get(k) for k in ("title", "authors", "year", "doi", "url", "journal",
                                                          "publisher", "document_type")}
                        | {"source_hash": u["source"].get("content_sha256")}})
    for u in units:
        u["quality"]["forced_status"] = "verified"

    snap = dataset_dir.with_name(dataset_dir.name + ".publish-backup")
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True)
    for n in _DATASET_FILES:
        if (dataset_dir / n).exists():
            shutil.copy2(dataset_dir / n, snap / n)
    try:
        res = merge_into_dataset(str(dataset_dir), list(docs_by_id.values()), units, pipe.taxonomy,
                                 pipe.cfg.extraction, id_prefix="SWR", preserve_ids=True)
        if res["units_added"]:
            cfg = pipe.cfg
            cfg.output_dir, cfg.index_dir = str(dataset_dir), str(index_dir)
            run_embed_and_index(cfg)
    except Exception as exc:
        for n in _DATASET_FILES:                                   # roll the dataset back
            if (snap / n).exists():
                shutil.copy2(snap / n, dataset_dir / n)
            elif (dataset_dir / n).exists():
                (dataset_dir / n).unlink()
        shutil.rmtree(snap, ignore_errors=True)
        return {"published": 0, "error": f"{exc.__class__.__name__}: {exc}", "rolled_back": True,
                "message": "publish failed; dataset restored, previous index kept, records stay verified/unpublished"}
    shutil.rmtree(snap, ignore_errors=True)
    in_dataset = {json.loads(l)["id"] for l in open(dataset_dir / "knowledge_units.jsonl", encoding="utf-8") if l.strip()}
    store.mark_published({u["id"]: u["id"] for u in units if u["id"] in in_dataset})
    return {"published": res["units_added"], "skipped_as_duplicates": len(units) - res["units_added"],
            "index_rebuilt": bool(res["units_added"])}
