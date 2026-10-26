import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "llm"))


def make_unit(uid, text, crops=(), status="validated", doc="doc1", title="A study", page=3, doi=None,
              url=None, authors=("A. Rao",), year=2022, journal="Irrig. Sci.", flags=None):
    q = {"validation_status": status}
    if flags:
        q["flags"] = flags
    return {"id": uid, "knowledge": {"evidence": text, "finding": None, "recommendation": None},
            "retrieval": {"semantic_text": text}, "quality": q,
            "source": {"title": title, "authors": list(authors), "year": year, "journal": journal, "doi": doi,
                       "url": url, "section": "results", "document_id": doc},
            "provenance": {"document_id": doc, "page": page, "section": "results",
                           "evidence_hash": hashlib.md5(text.encode()).hexdigest()},
            "classification": {"primary_domain": "water_management", "subdomains": []},
            "context": {"crop": list(crops)}, "structured_facts": []}


CORPUS = [
    ("SWM_000001", "Drip irrigation increased pod yield of groundnut by 18% over flood irrigation in the summer season at Coimbatore on loamy sand soil.", ["groundnut"], "validated"),
    ("SWM_000002", "Groundnut water use efficiency improved under drip irrigation scheduling at 0.8 pan evaporation in sandy loam soils during kharif.", ["groundnut"], "validated"),
    ("SWM_000003", "Nitrogen fertilizer application at 120 kg per hectare increased maize grain yield significantly in the rainfed alluvial soils of the region.", ["maize"], "validated"),
    ("SWM_000004", "Wheat irrigation at crown root initiation and flowering stages gave the highest grain yield and water productivity in semi-arid conditions.", ["wheat"], "validated"),
    ("SWM_000005", "Mulching with crop residue reduced soil evaporation and conserved moisture in rainfed farming systems across several seasons of study.", [], "validated"),
    ("SWM_000006", "Drip irrigation of cotton improved water use efficiency by 24% compared with furrow irrigation in the black soils of the deccan plateau.", ["cotton"], "validated"),
    ("SWM_000007", "Garlic and banana and rice and wheat and maize and cotton coefficients table lists many crops water requirement values for scheduling.", [], "needs_review"),
    ("SWM_000101", "Mango orchards drip irrigation scheduling needed substantial water in summer according to an unreviewed borderline extraction record here.", ["mango"], "needs_review"),
    ("SWM_000102", "Evidence only record about paddy rice water saving alternate wetting and drying practice in lowland irrigated rice systems of the delta.", ["rice"], "evidence_only"),
    ("SWM_000103", "Rejected record about wheat nitrogen where the extraction was broken and the evidence text is not reliable for any recommendation.", ["wheat"], "rejected"),
    ("SWR_000001", "Pending review record from online research about groundnut irrigation that no human has approved and so must never be used.", ["groundnut"], "pending_review"),
]


@pytest.fixture()
def units():
    return [make_unit(u, t, c, s, doc=f"doc_{u}") for u, t, c, s in CORPUS]


@pytest.fixture()
def built(tmp_path, units):
    """A real index built from CORPUS (+ dataset file) in tmp_path."""
    import json
    from sawvial.index.faiss_builder import build_index
    ds, idx = tmp_path / "dataset", tmp_path / "index"
    ds.mkdir()
    with open(ds / "knowledge_units.jsonl", "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")
    meta = build_index(units, idx, dataset_file=ds / "knowledge_units.jsonl")
    return {"dataset": ds, "index": idx, "meta": meta, "units": units}


def make_text_pdf(path, crop="groundnut"):
    import pymupdf
    doc = pymupdf.open(); pg = doc.new_page()
    pg.insert_text((72, 72), f"Effect of drip irrigation on {crop}", fontsize=18)
    pg.insert_text((72, 110), "Abstract", fontsize=13)
    y = 130
    for ln in [f"Drip irrigation increased pod yield of {crop} by 18% over flood irrigation in the summer season.",
               "Water use efficiency also improved in loamy sand soils under drip irrigation at Coimbatore."]:
        pg.insert_text((72, y), ln, fontsize=10); y += 14
    pg.insert_text((72, 200), "Results", fontsize=13)
    cells = ["Treatment", "Yield", "WUE", "Drip", "2.4", "5.1", "Flood", "2.0", "3.9"]
    for r in range(3):
        for c in range(3):
            rect = pymupdf.Rect(72 + c * 90, 225 + r * 22, 72 + (c + 1) * 90, 225 + (r + 1) * 22)
            pg.draw_rect(rect); pg.insert_text((rect.x0 + 4, rect.y0 + 15), cells[r * 3 + c], fontsize=9)
    pg.insert_text((72, 320), "Conclusion", fontsize=13)
    pg.insert_text((72, 338), f"Drip irrigation is recommended for {crop} in water scarce regions of India.", fontsize=10)
    doc.save(str(path))
    return pg


def make_scanned_pdf(path, text_pdf):
    import pymupdf
    src = pymupdf.open(str(text_pdf))
    pix = src[0].get_pixmap(dpi=200)
    sc = pymupdf.open(); p = sc.new_page(); p.insert_image(p.rect, pixmap=pix); sc.save(str(path))
