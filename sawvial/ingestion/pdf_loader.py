"""
sawvial.ingestion.pdf_loader
=============================
Replaces the old flat `page.get_text("text")` join with structure-aware
extraction:

  - per-page text blocks (with font size, to help detect headings later)
  - simple table detection via PyMuPDF's `find_tables()` (falls back
    gracefully on PyMuPDF versions without it)
  - basic PDF metadata (title/author/creation date) as a starting point
    for the metadata_extractor module

This module does NOT clean or classify text — it only produces a
faithful, structured representation of the PDF so that downstream
modules (document_understanding, extraction) have real structure to
work with instead of one flattened string.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import pymupdf

log = logging.getLogger("sawvial.pdf_loader")


@dataclass
class TextBlock:
    text: str
    page: int
    font_size: float
    bold: bool
    bbox: tuple


@dataclass
class TableBlock:
    page: int
    rows: List[List[str]]
    bbox: tuple


@dataclass
class PageContent:
    page_number: int
    blocks: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)

    @property
    def plain_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


@dataclass
class RawDocument:
    source_path: str
    filename: str
    pdf_metadata: Dict[str, Any]
    pages: List[PageContent]
    n_pages: int

    @property
    def full_plain_text(self) -> str:
        return "\n\n".join(p.plain_text for p in self.pages)


def _extract_tables(page: "pymupdf.Page", page_number: int) -> List[TableBlock]:
    """Extract tables using PyMuPDF's table finder where available.
    Returns [] silently if unsupported, rather than crashing the pipeline."""
    tables: List[TableBlock] = []
    try:
        finder = page.find_tables()
        for tbl in finder.tables:
            try:
                rows = tbl.extract()
                # normalize None cells to ""
                rows = [["" if c is None else str(c).strip() for c in row] for row in rows]
                tables.append(TableBlock(page=page_number, rows=rows, bbox=tuple(tbl.bbox)))
            except Exception as e:  # pragma: no cover
                log.debug(f"Table extraction failed on page {page_number}: {e}")
    except Exception:
        # Older PyMuPDF without find_tables(), or a page with no tables.
        pass
    return tables


def _extract_blocks(page: "pymupdf.Page", page_number: int) -> List[TextBlock]:
    blocks: List[TextBlock] = []
    try:
        raw = page.get_text("dict")
    except Exception:
        raw = {"blocks": []}

    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:
            continue  # image block etc.
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            line_text = "".join(s.get("text", "") for s in spans).strip()
            if not line_text:
                continue
            sizes = [s.get("size", 0.0) for s in spans]
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0
            is_bold = any("bold" in (s.get("font", "").lower()) for s in spans)
            blocks.append(TextBlock(
                text=line_text,
                page=page_number,
                font_size=round(avg_size, 1),
                bold=is_bold,
                bbox=tuple(line.get("bbox", (0, 0, 0, 0))),
            ))
    return blocks


def load_pdf(pdf_path: Path) -> RawDocument:
    """
    Open a PDF and return a structure-aware RawDocument:
    per-page text blocks (with font-size / bold flags for heading
    detection) and per-page tables.
    """
    doc = pymupdf.open(str(pdf_path))
    meta = dict(doc.metadata or {})

    pages: List[PageContent] = []
    for i, page in enumerate(doc):
        page_number = i + 1
        blocks = _extract_blocks(page, page_number)
        tables = _extract_tables(page, page_number)
        pages.append(PageContent(page_number=page_number, blocks=blocks, tables=tables))

    n_pages = len(pages)
    doc.close()

    return RawDocument(
        source_path=str(pdf_path),
        filename=pdf_path.name,
        pdf_metadata=meta,
        pages=pages,
        n_pages=n_pages,
    )
