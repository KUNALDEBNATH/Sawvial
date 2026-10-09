"""
sawvial.document_understanding.structure
==========================================
Detects document structure (sections → paragraphs) so that chunking
can operate on real scientific units instead of raw word windows.

Heuristics used (deterministic, no external ML model required):

  1. Section headings are detected from:
       - known scientific section names (Abstract, Introduction,
         Materials and Methods, Results, Discussion, Conclusion,
         Recommendations, References, ...)
       - short lines in a larger/bold font relative to the page's
         body-text font size (from pdf_loader's TextBlock metadata)
  2. Paragraphs are formed by grouping consecutive body-text lines
     until a heading, blank gap, or table boundary is hit.
  3. Table blocks are attached to the section they physically follow.

This is intentionally conservative: if no structure can be detected
(e.g. an extension leaflet with no headings), the whole document
becomes one "unknown" section containing all paragraphs, so nothing
is lost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from sawvial.ingestion.pdf_loader import RawDocument, TextBlock, TableBlock
from sawvial.preprocessing.cleaner import detect_repeated_lines, clean_page_text

_KNOWN_SECTIONS = {
    "abstract": "abstract",
    "summary": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "materials and methods": "materials_and_methods",
    "material and methods": "materials_and_methods",
    "methodology": "materials_and_methods",
    "methods": "materials_and_methods",
    "experimental procedure": "materials_and_methods",
    "results": "results",
    "results and discussion": "results",
    "observations": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "findings": "findings",
    "recommendation": "recommendation",
    "recommendations": "recommendation",
    "references": "references",
    "bibliography": "references",
    "acknowledgement": "acknowledgement",
    "acknowledgements": "acknowledgement",
}

_HEADING_LINE_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9 ,&/\-]{2,60}$")


def _classify_heading(line: str) -> Optional[str]:
    norm = re.sub(r"^\d+(\.\d+)*\.?\s*", "", line.strip().lower())
    norm = norm.strip(" :.-")
    return _KNOWN_SECTIONS.get(norm)


@dataclass
class Paragraph:
    text: str
    page: int
    section: str


@dataclass
class Section:
    name: str
    paragraphs: List[Paragraph] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)


@dataclass
class StructuredDocument:
    sections: List[Section] = field(default_factory=list)

    def iter_paragraphs(self):
        for section in self.sections:
            for para in section.paragraphs:
                yield section.name, para


def _is_heading_block(block: TextBlock, body_font_size: float) -> Optional[str]:
    known = _classify_heading(block.text)
    if known:
        return known
    is_larger_or_bold = block.font_size >= body_font_size + 1.0 or block.bold
    word_count = len(block.text.split())
    if is_larger_or_bold and 1 <= word_count <= 8 and _HEADING_LINE_RE.match(block.text.strip()):
        # Unrecognized but visually a heading -> keep as its own named section
        slug = re.sub(r"[^a-z0-9]+", "_", block.text.strip().lower()).strip("_")
        return slug or "unknown"
    return None


def _estimate_body_font_size(all_blocks: List[TextBlock]) -> float:
    if not all_blocks:
        return 10.0
    sizes = [b.font_size for b in all_blocks if b.font_size > 0]
    if not sizes:
        return 10.0
    sizes.sort()
    return sizes[len(sizes) // 2]  # median


def build_structured_document(raw_doc: RawDocument) -> StructuredDocument:
    all_blocks = [b for page in raw_doc.pages for b in page.blocks]
    body_font_size = _estimate_body_font_size(all_blocks)

    repeated_lines = detect_repeated_lines([p.plain_text for p in raw_doc.pages])

    sections: List[Section] = []
    current = Section(name="unknown")
    sections.append(current)

    paragraph_buffer: List[str] = []
    buffer_page = None
    prev_bbox_bottom = None
    prev_line_height = None

    def flush_buffer():
        nonlocal paragraph_buffer, buffer_page
        if paragraph_buffer:
            text = clean_page_text(" ".join(paragraph_buffer), repeated_lines)
            if text and len(text.split()) >= 4:
                current.paragraphs.append(Paragraph(text=text, page=buffer_page or 1, section=current.name))
        paragraph_buffer = []
        buffer_page = None

    for page in raw_doc.pages:
        prev_bbox_bottom = None
        prev_line_height = None
        for block in page.blocks:
            line = block.text.strip()
            if not line:
                continue
            heading = _is_heading_block(block, body_font_size)
            if heading:
                flush_buffer()
                current = Section(name=heading)
                sections.append(current)
                prev_bbox_bottom = None
                prev_line_height = None
                continue

            top = block.bbox[1]
            bottom = block.bbox[3]
            line_height = max(bottom - top, 1.0)

            # Paragraph break heuristic: a vertical gap noticeably larger
            # than the previous line's height indicates a new paragraph
            # (blank-line spacing) rather than a mere line wrap within the
            # same paragraph. This keeps genuinely related sentences
            # together while separating distinct paragraphs, instead of
            # merging an entire section into one giant paragraph.
            if (prev_bbox_bottom is not None and prev_line_height is not None
                    and (top - prev_bbox_bottom) > prev_line_height * 0.6):
                flush_buffer()

            if buffer_page is None:
                buffer_page = block.page
            paragraph_buffer.append(line)
            prev_bbox_bottom = bottom
            prev_line_height = line_height

            # Cap buffer growth so very long runs still get split into
            # digestible paragraphs even without a detected gap.
            if len(" ".join(paragraph_buffer)) > 1200:
                flush_buffer()
                prev_bbox_bottom = None
                prev_line_height = None
        # attach any tables found on this page to the current section
        if page.tables:
            current.tables.extend(page.tables)
    flush_buffer()

    # Drop entirely empty sections (can happen if a heading was immediately
    # followed by another heading).
    sections = [s for s in sections if s.paragraphs or s.tables]
    if not sections:
        sections = [Section(name="unknown")]

    return StructuredDocument(sections=sections)
