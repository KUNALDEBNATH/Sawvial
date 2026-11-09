import shutil
from pathlib import Path

import pymupdf
import pytest

from sawvial.document_understanding.structure import build_structured_document
from sawvial.ingestion import pdf_loader
from sawvial.ingestion.pdf_loader import PDFLoadError, load_pdf
from tests.conftest import make_scanned_pdf, make_text_pdf


@pytest.fixture()
def pdfs(tmp_path):
    t, s = tmp_path / "text.pdf", tmp_path / "scanned.pdf"
    make_text_pdf(t); make_scanned_pdf(s, t)
    return {"text": t, "scanned": s, "dir": tmp_path}


def test_text_pdf_ok_and_table_goes_to_the_right_section(pdfs):
    raw = load_pdf(pdfs["text"])
    assert raw.extraction.status == "ok" and raw.extraction.tables_found == 1
    secs = {s.name: s for s in build_structured_document(raw).sections}
    assert len(secs["results"].tables) == 1 and not secs["conclusion"].tables and not secs["abstract"].tables
    assert raw.pages[0].page_number == 1


def test_scanned_pdf_is_detected_not_silently_accepted(pdfs):
    e = load_pdf(pdfs["scanned"]).extraction
    assert e.status == "no_text" and e.low_text_pages == [1] and any("--ocr" in w for w in e.warnings)


@pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract not installed")
def test_ocr_recovers_text_from_scanned_pdf(pdfs):
    r = load_pdf(pdfs["scanned"], ocr=True)
    text = " ".join(b.text for p in r.pages for b in p.blocks).lower()
    assert r.extraction.status == "ok" and r.extraction.ocr_pages == [1] and "groundnut" in text


def test_ocr_failure_is_recorded_not_hidden(pdfs, monkeypatch):
    def boom(*a, **k): raise RuntimeError("No OCR support: tesseract not found")
    monkeypatch.setattr(pdf_loader, "_ocr_blocks", boom)
    e = load_pdf(pdfs["scanned"], ocr=True).extraction
    assert e.status == "no_text" and "tesseract" in e.ocr_error and any("OCR failed" in w for w in e.warnings)


def test_table_extraction_failure_is_reported(pdfs, monkeypatch):
    monkeypatch.setattr(pymupdf.Page, "find_tables", lambda self: (_ for _ in ()).throw(RuntimeError("table boom")))
    e = load_pdf(pdfs["text"]).extraction
    assert e.table_errors and e.status == "partial" and any("table extraction failed" in w for w in e.warnings)


def test_encrypted_pdf_gives_clear_error(pdfs):
    out = pdfs["dir"] / "enc.pdf"
    d = pymupdf.open(str(pdfs["text"])); d.save(str(out), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="o")
    with pytest.raises(PDFLoadError, match="password"):
        load_pdf(out)


def test_corrupt_pdf_gives_clear_error(tmp_path):
    bad = tmp_path / "bad.pdf"; bad.write_bytes(b"not a pdf at all")
    with pytest.raises(PDFLoadError):
        load_pdf(bad)


def test_page_limit_is_reported(pdfs):
    d = pymupdf.open(str(pdfs["text"]))
    for _ in range(3): d.new_page()
    out = pdfs["dir"] / "long.pdf"; d.save(str(out))
    e = load_pdf(out, max_pages=2).extraction
    assert e.truncated_at_page == 2 and e.pages_total == 2 and e.status == "partial"


def test_add_pdf_end_to_end_with_index_and_duplicate_and_scanned(pdfs, tmp_path):
    from sawvial.config import BuildConfig, ChunkingConfig, EmbeddingConfig, ExtractionConfig, Taxonomy
    from sawvial.index.passages import retrieve_passages
    from sawvial.index.retriever import load_index_bundle
    from sawvial.pipeline import run_add_pdfs, run_embed_and_index
    root = Path(__file__).resolve().parent.parent
    cfg = BuildConfig(taxonomy=Taxonomy(root / "taxonomy.json"), chunking=ChunkingConfig(), extraction=ExtractionConfig(),
                      embedding=EmbeddingConfig(offline=True), input_dir="", output_dir=str(tmp_path / "ds"), index_dir=str(tmp_path / "ix"))
    (tmp_path / "ds").mkdir()
    res = run_add_pdfs(cfg, [pdfs["text"], pdfs["scanned"]])
    assert res["documents_added"] == 1 and res["units_added"] >= 1
    assert {f["status"] for f in res["files"]} == {"ok", "no_text"}          # scanned PDF reported, not added
    run_embed_and_index(cfg)
    r = retrieve_passages(load_index_bundle(cfg.index_dir, cfg.output_dir), "Is drip irrigation recommended for groundnut?", 3)
    assert r["passages"] and r["passages"][0]["page"] == 1 and "groundnut" in r["passages"][0]["text"].lower()
    again = run_add_pdfs(cfg, [pdfs["text"]])
    assert again["documents_skipped_duplicate"] == 1 and again["units_added"] == 0


def test_mostly_blank_pdf_is_never_auto_validated(pdfs, tmp_path):
    from sawvial.config import BuildConfig, ChunkingConfig, EmbeddingConfig, ExtractionConfig, Taxonomy
    from sawvial.pipeline import run_add_pdfs
    import json
    d = pymupdf.open(str(pdfs["text"]))
    for _ in range(4): d.new_page()
    p = tmp_path / "mostly_blank.pdf"; d.save(str(p))
    root = Path(__file__).resolve().parent.parent
    cfg = BuildConfig(taxonomy=Taxonomy(root / "taxonomy.json"), chunking=ChunkingConfig(), extraction=ExtractionConfig(),
                      embedding=EmbeddingConfig(offline=True), input_dir="", output_dir=str(tmp_path / "ds2"), index_dir=str(tmp_path / "ix2"))
    (tmp_path / "ds2").mkdir()
    res = run_add_pdfs(cfg, [p])
    assert res["files"][0]["status"] == "partial"
    statuses = {json.loads(l)["quality"]["validation_status"] for l in open(tmp_path / "ds2" / "knowledge_units.jsonl")}
    assert "validated" not in statuses and "needs_review" in statuses
