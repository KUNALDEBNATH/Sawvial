"""
sawvial.preprocessing.metadata_extractor
=========================================
Pulls bibliographic metadata (title, authors, year, DOI, journal) from
PDF metadata first, falling back to heuristic scanning of the first
page's text when the embedded PDF metadata is missing or generic
(e.g. "Microsoft Word - filename.docx").
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List

from sawvial.ingestion.pdf_loader import RawDocument

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_GENERIC_TITLE_RE = re.compile(r"^(microsoft word|untitled|document\d*)", re.IGNORECASE)


def _first_page_text(doc: RawDocument, n_chars: int = 3000) -> str:
    if not doc.pages:
        return ""
    return doc.pages[0].plain_text[:n_chars]


def _guess_title(doc: RawDocument, pdf_meta_title: str) -> str:
    if pdf_meta_title and not _GENERIC_TITLE_RE.match(pdf_meta_title.strip()):
        return pdf_meta_title.strip()

    # Heuristic: largest-font line(s) near the top of page 1 are usually the title.
    if doc.pages:
        blocks = [b for b in doc.pages[0].blocks if b.text.strip()]
        blocks = [b for b in blocks if len(b.text.split()) >= 3][:20]
        if blocks:
            max_size = max(b.font_size for b in blocks)
            candidates = [b.text for b in blocks if b.font_size >= max_size - 0.5]
            title = " ".join(candidates[:3]).strip()
            if title:
                return title

    return Path(doc.filename).stem.replace("_", " ").replace("-", " ").strip()


def _guess_authors(pdf_meta_author: str) -> List[str]:
    if not pdf_meta_author:
        return []
    parts = re.split(r";|,|\band\b", pdf_meta_author)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]


def extract_metadata(doc: RawDocument) -> Dict[str, Any]:
    meta = doc.pdf_metadata or {}
    first_page = _first_page_text(doc)

    title = _guess_title(doc, meta.get("title", ""))
    authors = _guess_authors(meta.get("author", ""))

    year = None
    year_match = _YEAR_RE.search(first_page)
    if year_match:
        year = int(year_match.group(0))
    elif meta.get("creationDate"):
        y = _YEAR_RE.search(meta["creationDate"])
        if y:
            year = int(y.group(0))

    doi_match = _DOI_RE.search(first_page)
    doi = doi_match.group(0).rstrip(".,)") if doi_match else None

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "url": None,
        "journal": None,
        "publisher": None,
        "document_type": "research_paper",
        "n_pages": doc.n_pages,
        "filename": doc.filename,
        "source_path": doc.source_path,
    }
