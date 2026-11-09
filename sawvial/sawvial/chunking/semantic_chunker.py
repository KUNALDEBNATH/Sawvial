"""
sawvial.chunking.semantic_chunker
====================================
Replaces the old fixed 400-word sliding-window chunker with adaptive,
paragraph/sentence-aware chunking.

Pipeline (per master-spec section 4/16):

    section -> paragraph -> sentence groups -> candidate chunk

Rules:
  - A paragraph that already forms a coherent scientific statement and
    fits within `max_chunk_chars` becomes ONE candidate chunk as-is
    (never sliced mid-reasoning).
  - A long paragraph is split at SENTENCE boundaries only, greedily
    packed up to `target_chunk_chars`, so an intervention/outcome pair
    inside a single sentence is never torn apart.
  - Candidate chunks below `min_chunk_chars` are merged with the next
    sentence group in the same paragraph rather than emitted as tiny,
    context-free fragments.
  - Table blocks become their own candidate chunk with a structured
    textual rendering, tagged separately so they are not lost the way
    the previous version discarded them entirely.

This module produces CandidateChunk objects only — no domain
classification, scoring or knowledge-unit construction happens here
(single responsibility; see extraction/classification/knowledge
modules for those steps).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from sawvial.document_understanding.structure import StructuredDocument, TableBlock
from sawvial.config import ChunkingConfig

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class CandidateChunk:
    text: str
    section: str
    page: int
    chunk_type: str = "text"       # "text" | "table"
    table_rows: List[List[str]] = field(default_factory=list)


def _split_sentences(paragraph_text: str) -> List[str]:
    sentences = _SENTENCE_SPLIT_RE.split(paragraph_text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _pack_sentences(sentences: List[str], cfg: ChunkingConfig) -> List[str]:
    packed: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)
        if current and current_len + s_len > cfg.target_chunk_chars:
            packed.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += s_len + 1

    if current:
        packed.append(" ".join(current))

    # Merge any too-small trailing fragment into the previous chunk so we
    # never emit a context-free one-liner.
    merged: List[str] = []
    for chunk in packed:
        if merged and len(chunk) < cfg.min_chunk_chars:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged


def _table_to_text(table: TableBlock) -> str:
    lines = []
    for row in table.rows:
        cells = [c for c in row if c]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def build_candidate_chunks(structured_doc: StructuredDocument, cfg: ChunkingConfig) -> List[CandidateChunk]:
    chunks: List[CandidateChunk] = []

    for section in structured_doc.sections:
        if section.name in ("references", "acknowledgement"):
            continue  # never scientifically meaningful knowledge units

        for para in section.paragraphs:
            if len(para.text) <= cfg.max_chunk_chars:
                if len(para.text) >= cfg.min_chunk_chars:
                    chunks.append(CandidateChunk(text=para.text, section=section.name, page=para.page))
                continue

            sentences = _split_sentences(para.text)
            for piece in _pack_sentences(sentences, cfg):
                if len(piece) >= cfg.min_chunk_chars:
                    chunks.append(CandidateChunk(text=piece, section=section.name, page=para.page))

        for table in section.tables:
            table_text = _table_to_text(table)
            if table_text.strip():
                chunks.append(CandidateChunk(
                    text=table_text, section=section.name, page=table.page,
                    chunk_type="table", table_rows=table.rows,
                ))

    return chunks
