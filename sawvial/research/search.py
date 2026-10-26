"""
Literature search through free, key-less scholarly metadata APIs.

Providers (select with SAWVIAL_SEARCH_PROVIDER):
  openalex  https://api.openalex.org  (default; open-access links included)
  crossref  https://api.crossref.org  (metadata + publisher links)
  none      search disabled - use `research_cli.py url <URL>` instead

These are scholarly indexes, NOT a general web search engine. Results are only as good as
their metadata. Nothing is fabricated: if the API is unreachable, ``SearchUnavailable`` is
raised with the reason and the caller reports it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlparse

from sawvial import settings
from sawvial.research.safe_http import FetchError, UnsafeURL, fetch

# Domains treated as preferred / official agricultural research sources (ranking boost only;
# other domains are allowed but marked trusted=False for the reviewer).
TRUSTED_DOMAINS = (
    "icar.org.in", "icar.gov.in", "epubs.icar.org.in", "krishi.icar.gov.in", "icrisat.org",
    "iari.res.in", "fao.org", "cgiar.org", "ars.usda.gov", "nih.gov", "europepmc.org",
    "doi.org", "sciencedirect.com", "springer.com", "nature.com", "wiley.com", "mdpi.com",
    "frontiersin.org", "plos.org", "tandfonline.com", "agupubs.onlinelibrary.wiley.com",
    "ijas.icar.org.in", "ac.in", "res.in", "edu",
)


class SearchUnavailable(RuntimeError):
    pass


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
    trusted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def is_trusted(url: Optional[str]) -> bool:
    host = (urlparse(url or "").hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def _clean_doi(doi: Optional[str]) -> Optional[str]:
    m = re.search(r"10\.\d{4,9}/\S+", doi or "")
    return m.group(0).rstrip(".,;)") if m else None


def parse_openalex(payload: Dict[str, Any]) -> List[Candidate]:
    out = []
    for w in payload.get("results", []):
        if not w.get("title"):
            continue
        best = w.get("best_oa_location") or {}
        primary = w.get("primary_location") or {}
        landing = best.get("landing_page_url") or primary.get("landing_page_url") \
            or (w.get("open_access") or {}).get("oa_url")
        pdf = best.get("pdf_url")
        out.append(Candidate(
            title=w["title"], year=w.get("publication_year"), doi=_clean_doi(w.get("doi")),
            authors=[a["author"]["display_name"] for a in w.get("authorships", [])
                     if a.get("author", {}).get("display_name")],
            venue=((primary.get("source") or {}).get("display_name")),
            landing_url=landing, pdf_url=pdf, provider="openalex",
            trusted=is_trusted(pdf) or is_trusted(landing)))
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
        landing = it.get("URL") or (f"https://doi.org/{doi}" if doi else None)
        out.append(Candidate(
            title=title, year=year, doi=doi, venue=(it.get("container-title") or [None])[0],
            authors=[" ".join(filter(None, [a.get("given"), a.get("family")])) for a in it.get("author", [])],
            landing_url=landing, pdf_url=pdf, provider="crossref",
            trusted=is_trusted(pdf) or is_trusted(landing)))
    return out


def _get_json(url: str, host: str) -> Dict[str, Any]:
    try:
        res = fetch(url, allowed_types={"application/json"}, allowed_domains=[host],
                    max_bytes=5 * 1024 * 1024, accept="application/json")
        return json.loads(res.content)
    except (FetchError, UnsafeURL, ValueError) as exc:
        raise SearchUnavailable(f"{host} search failed: {exc}") from exc


def search_literature(question: str, provider: Optional[str] = None, max_results: int = 10) -> List[Candidate]:
    provider = (provider or settings.SEARCH_PROVIDER).lower()
    mail = settings.SEARCH_CONTACT_EMAIL
    if provider == "none":
        raise SearchUnavailable("search is disabled (SAWVIAL_SEARCH_PROVIDER=none). Use "
                                "`python research_cli.py url <URL>` to ingest a specific document.")
    q = quote(re.sub(r"\s+", " ", question).strip()[:300])
    if provider == "openalex":
        url = f"https://api.openalex.org/works?search={q}&per-page={max_results}"
        if mail:
            url += f"&mailto={quote(mail)}"
        cands = parse_openalex(_get_json(url, "api.openalex.org"))
    elif provider == "crossref":
        url = f"https://api.crossref.org/works?query={q}&rows={max_results}"
        if mail:
            url += f"&mailto={quote(mail)}"
        cands = parse_crossref(_get_json(url, "api.crossref.org"))
    else:
        raise SearchUnavailable(f"unknown search provider {provider!r} (use openalex, crossref or none)")
    # trusted sources and documents we can actually download first
    return sorted(cands, key=lambda c: (not c.trusted, not c.pdf_url, -(c.year or 0)))
