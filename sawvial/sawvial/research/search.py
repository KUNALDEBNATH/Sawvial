"""
Online discovery for agricultural questions.

Sources (key-less; each failure is reported - never faked):
  * OpenAlex and Crossref  - key-less scholarly metadata (papers, many with open-access PDFs).
  No API key is used or stored anywhere. We deliberately do NOT scrape search-engine result pages
  (terms of service / CAPTCHA), so general web pages (e.g. TNAU / Vikaspedia extension pages) are only
  reachable when a paper or a scholarly record links to them.

Trust vs permission
  * "May we fetch it?"            safe_http (public addresses only, limits) + optional allowed-domain list.
  * "May it support an answer?"   trust_tier() of the FINAL url after redirects:
        official   research/extension institutions (ICAR, TNAU, FAO, CGIAR/ICRISAT ...)
        reputable  universities, learned publishers, Vikaspedia, PubMed/EuropePMC
        unknown    everything else
    The tier is NEVER inherited from the search provider, from doi.org, or from where a link was found.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

from sawvial import settings
from sawvial.research.safe_http import FetchError, UnsafeURL, fetch

OFFICIAL_DOMAINS = (
    "icar.gov.in", "icar.org.in", "epubs.icar.org.in", "krishi.icar.gov.in", "kvk.icar.gov.in",
    "agritech.tnau.ac.in", "tnau.ac.in", "icrisat.org", "iari.res.in", "fao.org", "cgiar.org",
    "ars.usda.gov", "nrcs.usda.gov", "extension.org",
)
REPUTABLE_DOMAINS = (
    "vikaspedia.in", "nih.gov", "europepmc.org", "sciencedirect.com", "springer.com", "nature.com",
    "wiley.com", "mdpi.com", "frontiersin.org", "plos.org", "tandfonline.com", "cambridge.org",
    "oup.com", "sagepub.com", "researchgate.net",
)
REPUTABLE_SUFFIXES = (".edu", ".ac.in", ".res.in", ".ac.uk", ".edu.au", ".ac.za", ".gov.in")
# Resolvers/aggregators: allowed to be FETCHED, but they say nothing about the destination's trustworthiness.
RESOLVER_DOMAINS = ("doi.org", "dx.doi.org", "hdl.handle.net")
TIER_RANK = {"unknown": 0, "reputable": 1, "official": 2}


class SearchUnavailable(RuntimeError):
    """No provider could be reached/used. The message lists every provider's reason."""


def _host(url: Optional[str]) -> str:
    return (urlparse(url or "").hostname or "").lower().rstrip(".")


def _in(host: str, domains) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def trust_tier(url: Optional[str]) -> str:
    h = _host(url)
    if not h or _in(h, RESOLVER_DOMAINS):
        return "unknown"
    if _in(h, OFFICIAL_DOMAINS):
        return "official"
    if _in(h, REPUTABLE_DOMAINS) or h.endswith(REPUTABLE_SUFFIXES):
        return "reputable"
    return "unknown"


def tier_allowed(tier: str, minimum: Optional[str] = None) -> bool:
    minimum = (minimum or settings.RESEARCH_MIN_TRUST).lower()
    return True if minimum == "any" else TIER_RANK[tier] >= TIER_RANK.get(minimum, 1)


def is_trusted(url: Optional[str]) -> bool:      # kept for callers: "trusted" = official or reputable
    return trust_tier(url) != "unknown"


def classify_source(final_url: str, doi: Optional[str], provider: str, content_type: str) -> str:
    """What kind of statement does this source make?
       research_paper       -> a finding from a specific study/experiment
       official_guidance    -> a recommendation published by an official research/extension body
       informational_article-> a general explanatory page"""
    scholarly = bool(doi) or provider in ("openalex", "crossref", "europepmc") or "journal" in _host(final_url)
    if scholarly or (content_type == "application/pdf" and trust_tier(final_url) == "reputable"):
        return "research_paper"
    if trust_tier(final_url) == "official":
        return "official_guidance"
    return "informational_article"


@dataclass
class Candidate:
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    venue: Optional[str] = None
    landing_url: Optional[str] = None
    pdf_url: Optional[str] = None
    provider: str = ""
    snippet: str = ""

    @property
    def target(self) -> Optional[str]:
        return self.pdf_url or self.landing_url

    @property
    def tier(self) -> str:                     # pre-fetch estimate only; the FINAL url decides later
        return trust_tier(self.target)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy(); d["tier"] = self.tier
        return d


def _clean_doi(doi: Optional[str]) -> Optional[str]:
    m = re.search(r"10\.\d{4,9}/\S+", doi or "")
    return m.group(0).rstrip(".,;)") if m else None


# ---------------------------------------------------------------- scholarly providers
def parse_openalex(payload: Dict[str, Any]) -> List[Candidate]:
    out = []
    for w in payload.get("results", []):
        if not w.get("title"):
            continue
        best, primary = w.get("best_oa_location") or {}, w.get("primary_location") or {}
        landing = best.get("landing_page_url") or primary.get("landing_page_url") \
            or (w.get("open_access") or {}).get("oa_url")
        out.append(Candidate(
            title=w["title"], year=w.get("publication_year"), doi=_clean_doi(w.get("doi")),
            authors=[a["author"]["display_name"] for a in w.get("authorships", [])
                     if a.get("author", {}).get("display_name")],
            venue=((primary.get("source") or {}).get("display_name")),
            landing_url=landing, pdf_url=best.get("pdf_url"), provider="openalex"))
    return out


def parse_crossref(payload: Dict[str, Any]) -> List[Candidate]:
    out = []
    for it in (payload.get("message") or {}).get("items", []):
        title = (it.get("title") or [""])[0]
        if not title:
            continue
        year = ((it.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        pdf = next((l["URL"] for l in it.get("link", [])
                    if l.get("content-type") == "application/pdf" and l.get("URL")), None)
        doi = _clean_doi(it.get("DOI"))
        out.append(Candidate(
            title=title, year=year, doi=doi, venue=(it.get("container-title") or [None])[0],
            authors=[" ".join(filter(None, [a.get("given"), a.get("family")])) for a in it.get("author", [])],
            landing_url=it.get("URL") or (f"https://doi.org/{doi}" if doi else None), pdf_url=pdf,
            provider="crossref"))
    return out


def _get_json(url: str, host_for_policy: str) -> Dict[str, Any]:
    try:
        res = fetch(url, allowed_types={"application/json"}, max_bytes=5 * 1024 * 1024, accept="application/json",
                    allowed_domains=[host_for_policy])
        return json.loads(res.content)
    except (FetchError, UnsafeURL, ValueError) as exc:
        raise SearchUnavailable(f"{_host(url) or url}: {exc}") from exc


# ---------------------------------------------------------------- query generation
_QUERY_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "best", "by", "can", "do", "does", "for", "from", "give", "good",
    "grow", "growing", "grown", "how", "i", "in", "is", "it", "kind", "me", "my", "of", "on", "or", "should",
    "tell", "the", "to", "type", "types", "use", "what", "when", "where", "which", "why", "will", "with", "would", "you",
}
# What a farmer asking about X usually needs evidence on; chosen from the QUESTION's own topic words,
# never from the crop. (These steer the search; they are not answers.)
TOPIC_TERMS = [
    (r"\bsoils?\b|\bmann\b|\bland\b", "soil requirements pH drainage texture organic matter"),
    (r"irrigat|water|moisture|drought|rain", "irrigation water requirement schedule"),
    (r"fertili[sz]|nutrient|manure|compost|npk|nitrogen|phosph|potash", "fertilizer nutrient management recommendation"),
    (r"salin|alkal|sodic|acid", "saline sodic acid soil management"),
    (r"erosion|conservation|mulch|tillage|runoff", "soil and water conservation practices"),
]


def build_queries(question: str, crops: Optional[List[str]] = None, max_queries: int = 3) -> List[str]:
    """Search strings derived from the user's own question (keywords + crop + topic-specific terms)."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]+", question) if w.lower() not in _QUERY_STOP]
    base = " ".join(dict.fromkeys(w.lower() for w in words))[:120]
    subject = " ".join(crops) if crops else " ".join(words[:3]).lower()
    queries = [base] if base else []
    for pattern, terms in TOPIC_TERMS:
        if re.search(pattern, question, re.I):
            queries.append(f"{subject} {terms}".strip())
    if len(queries) < 2 and subject:
        queries.append(f"{subject} cultivation guide agriculture")
    return list(dict.fromkeys(q for q in queries if q))[:max_queries]


def _overlap(c: Candidate, terms: set) -> float:
    text = f"{c.title} {c.snippet}".lower()
    return sum(t in text for t in terms) / max(len(terms), 1)


# ---------------------------------------------------------------- orchestration
def discover(question: str, crops: Optional[List[str]] = None, max_results: int = 12,
             provider: Optional[str] = None) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """Run every configured provider for each generated query. Returns (ranked candidates, per-provider report).
    Raises SearchUnavailable only if NO provider produced a usable response."""
    provider = (provider or settings.SEARCH_PROVIDER).lower()
    if provider == "none":
        raise SearchUnavailable("online search is disabled (SAWVIAL_SEARCH_PROVIDER=none)")
    queries = build_queries(question, crops)
    n = max_results
    jobs: List[Tuple[str, Callable[[], List[Candidate]]]] = []
    for q in queries:
        qq = quote(q)
        if provider in ("auto", "openalex"):
            jobs.append(("openalex", lambda qq=qq: parse_openalex(_get_json(
                f"https://api.openalex.org/works?search={qq}&per-page={n}&filter=open_access.is_oa:true",
                "api.openalex.org"))))                                   # open access = full text we may actually read
        if provider in ("auto", "crossref"):
            jobs.append(("crossref", lambda qq=qq: parse_crossref(_get_json(
                f"https://api.crossref.org/works?query={qq}&rows={n}", "api.crossref.org"))))
    report: Dict[str, Dict[str, Any]] = {}
    cands: List[Candidate] = []
    for name, job in jobs:
        r = report.setdefault(name, {"provider": name, "status": "failed", "results": 0, "error": None})
        try:
            got = job()
            r["results"] += len(got)
            r["status"], r["error"] = "ok", None
            cands.extend(got)
        except SearchUnavailable as exc:
            if r["status"] != "ok":
                r["error"] = str(exc)
    reports = list(report.values())
    if not any(r["status"] == "ok" for r in reports):
        raise SearchUnavailable("; ".join(f"{r['provider']}: {r['error']}" for r in reports) or "no provider configured")

    seen, unique = set(), []
    terms = {w.lower() for q in queries for w in q.split() if len(w) > 3}
    for c in cands:
        key = (c.doi or "").lower() or (c.target or "").split("#")[0].rstrip("/").lower() or c.title.lower()
        if key not in seen and c.target:
            seen.add(key); unique.append(c)
    unique.sort(key=lambda c: (-TIER_RANK[c.tier], -_overlap(c, terms), not c.pdf_url, -(c.year or 0)))
    return unique[:max_results], reports


def search_literature(question: str, provider: Optional[str] = None, max_results: int = 10) -> List[Candidate]:
    """Backward-compatible wrapper used by research_cli.py."""
    from sawvial.index.retriever import detect_crops
    return discover(question, detect_crops(question), max_results, provider)[0]
