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
  - an ExtractionReport per document: pages without text (scanned PDFs), table
    failures, OCR usage and warnings - so problems are reported, not hidden

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
        """All of this page's text, including table content.

        Table rows are deliberately included: two documents that share the same narrative text but
        contain different data tables are different documents, and must not hash identically (see
        RawDocument.full_plain_text, used for document_hash()/document identity and deduplication).
        Table cells never duplicate the paragraph text produced by structure.py - that module reads
        page.blocks/page.tables directly and explicitly skips any block that falls inside a table's
        bounding box, so this property is the only place table content and paragraph text are
        combined into one flat string, and only for hashing/repeated-line-detection purposes.
        """
        parts = [b.text for b in self.blocks]
        for table in self.tables:
            for row in table.rows:
                row_text = " | ".join(c.strip() for c in row if c and c.strip())
                if row_text:
                    parts.append(row_text)
        return "\n".join(parts)


MIN_CHARS_PER_PAGE = 100          # fewer extracted characters than this -> "low-text" page
PARTIAL_FRACTION = 0.5            # >= this share of low-text pages -> document is incomplete


class PDFLoadError(RuntimeError):
    """The PDF could not be opened/read at all (encrypted, corrupt, ...)."""


@dataclass
class ExtractionReport:
    pages_total: int = 0
    pages_with_text: int = 0
    chars_total: int = 0
    low_text_pages: List[int] = field(default_factory=list)
    tables_found: int = 0
    tables_discarded: int = 0
    table_errors: List[Dict[str, Any]] = field(default_factory=list)
    ocr_requested: bool = False
    ocr_pages: List[int] = field(default_factory=list)
    ocr_error: Optional[str] = None
    truncated_at_page: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """ok | partial (incomplete text and/or table failures) | no_text (likely scanned)."""
        if self.pages_total == 0 or self.pages_with_text == 0 or self.chars_total < 50:
            return "no_text"
        if (len(self.low_text_pages) / self.pages_total >= PARTIAL_FRACTION
                or self.table_errors or self.truncated_at_page):
            return "partial"
        return "ok"

    @property
    def incomplete(self) -> bool:
        return self.status in ("partial", "no_text")

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "pages_total": self.pages_total,
                "pages_with_text": self.pages_with_text, "chars_total": self.chars_total,
                "low_text_pages": self.low_text_pages, "tables_found": self.tables_found,
                "tables_discarded": self.tables_discarded, "table_errors": self.table_errors,
                "ocr_requested": self.ocr_requested, "ocr_pages": self.ocr_pages,
                "ocr_error": self.ocr_error, "truncated_at_page": self.truncated_at_page,
                "warnings": self.warnings}


@dataclass
class RawDocument:
    source_path: str
    filename: str
    pdf_metadata: Dict[str, Any]
    pages: List[PageContent]
    n_pages: int
    extraction: ExtractionReport = field(default_factory=ExtractionReport)

    @property
    def full_plain_text(self) -> str:
        return "\n\n".join(p.plain_text for p in self.pages)


def _table_is_plausible(rows: List[List[str]]) -> bool:
    """>=2 rows, >=2 columns and at least 40% non-empty cells; otherwise it is layout noise."""
    if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
        return False
    cells = [c for r in rows for c in r]
    return bool(cells) and sum(bool(c) for c in cells) / len(cells) >= 0.4


def _extract_tables(page: "pymupdf.Page", page_number: int, report: ExtractionReport) -> List[TableBlock]:
    """Extract tables with PyMuPDF's finder. Failures are RECORDED in the report (not swallowed)
    so a document whose tables could not be read is not treated as fully processed."""
    tables: List[TableBlock] = []
    try:
        finder = page.find_tables()
    except AttributeError:
        if "table_extraction_unsupported" not in report.warnings:
            report.warnings.append("table_extraction_unsupported: this PyMuPDF version has no find_tables()")
        return tables
    except Exception as exc:
        report.table_errors.append({"page": page_number, "error": f"find_tables: {exc}"})
        return tables
    for tbl in finder.tables:
        try:
            rows = [["" if c is None else str(c).strip() for c in row] for row in tbl.extract()]
        except Exception as exc:
            report.table_errors.append({"page": page_number, "error": f"extract: {exc}"})
            continue
        if _table_is_plausible(rows):
            tables.append(TableBlock(page=page_number, rows=rows, bbox=tuple(tbl.bbox)))
            report.tables_found += 1
        else:
            report.tables_discarded += 1
    return tables


def _extract_blocks(page: "pymupdf.Page", page_number: int, textpage=None) -> List[TextBlock]:
    blocks: List[TextBlock] = []
    try:
        raw = page.get_text("dict", textpage=textpage) if textpage else page.get_text("dict")
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


def _ocr_blocks(page: "pymupdf.Page", page_number: int, language: str) -> List[TextBlock]:
    """OCR a page (needs the Tesseract program + language data installed)."""
    tp = page.get_textpage_ocr(language=language, dpi=200, full=True)
    return _extract_blocks(page, page_number, textpage=tp)


def load_pdf(pdf_path: Path, ocr: bool = False, ocr_language: str = "eng",
             max_pages: Optional[int] = None) -> RawDocument:
    """
    Open a PDF and return a structure-aware RawDocument plus an ExtractionReport.

    ocr=True runs OCR on pages with (almost) no extractable text. It is OPTIONAL because it
    needs Tesseract installed; if that is missing the failure is recorded in
    ``extraction.ocr_error`` and the document is flagged, never silently accepted.
    """
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise PDFLoadError(f"cannot open {Path(pdf_path).name}: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise PDFLoadError(f"{Path(pdf_path).name} is password protected")
    meta = dict(doc.metadata or {})
    report = ExtractionReport(ocr_requested=ocr)

    pages: List[PageContent] = []
    for i, page in enumerate(doc):
        page_number = i + 1
        if max_pages and page_number > max_pages:
            report.truncated_at_page = max_pages
            report.warnings.append(f"document truncated to the first {max_pages} pages")
            break
        blocks = _extract_blocks(page, page_number)
        if ocr and sum(len(b.text) for b in blocks) < MIN_CHARS_PER_PAGE and report.ocr_error is None:
            try:
                ocr_blocks = _ocr_blocks(page, page_number, ocr_language)
                if sum(len(b.text) for b in ocr_blocks) > sum(len(b.text) for b in blocks):
                    blocks = ocr_blocks
                    report.ocr_pages.append(page_number)
            except Exception as exc:                       # Tesseract missing / language data missing
                report.ocr_error = f"{exc.__class__.__name__}: {exc}"
                report.warnings.append(f"OCR failed ({report.ocr_error}); see README 'OCR (optional)'")
        tables = _extract_tables(page, page_number, report)
        pages.append(PageContent(page_number=page_number, blocks=blocks, tables=tables))
        chars = sum(len(b.text) for b in blocks)
        report.chars_total += chars
        if chars >= MIN_CHARS_PER_PAGE:
            report.pages_with_text += 1
        else:
            report.low_text_pages.append(page_number)
    report.pages_total = len(pages)
    n_pages = len(pages)
    doc.close()

    if report.status == "no_text":
        report.warnings.append("no extractable text - probably a scanned PDF. Re-run with --ocr "
                               "(requires Tesseract) or supply a text-based PDF")
    elif report.low_text_pages and report.status == "partial":
        report.warnings.append(f"{len(report.low_text_pages)}/{n_pages} pages have little or no text")
    if report.table_errors:
        report.warnings.append(f"table extraction failed on {len(report.table_errors)} table(s)")

    return RawDocument(source_path=str(pdf_path), filename=Path(pdf_path).name, pdf_metadata=meta,
                       pages=pages, n_pages=n_pages, extraction=report)
