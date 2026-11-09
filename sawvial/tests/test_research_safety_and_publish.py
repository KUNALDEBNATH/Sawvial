import json
import shutil
from pathlib import Path

import pytest

from sawvial import settings
from sawvial.research import safe_http as H
from sawvial.research import search as S
from sawvial.research import workflow as W
from sawvial.research.store import ResearchStore
from tests.conftest import make_text_pdf
from tests.netfixture import SEEN, web  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
ARTICLE = ("<html><head><title>Tomato notes</title></head><body><article><h1>Tomato</h1><p>{}</p></article></body></html>")
LONG = ("Tomato grows well in well drained loamy soils that are rich in organic matter and have a balanced soil pH value. "
        "Heavy clay soils with poor drainage reduce root growth and fruit set in tomato crops during the rainy season. ") * 3


@pytest.fixture(scope="module")
def pipe():
    return W.get_pipeline()


# ---------------------------------------------------------------- trust: permission to fetch != trust to answer
@pytest.mark.parametrize("url,tier", [
    ("https://agritech.tnau.ac.in/x", "official"), ("https://www.icar.gov.in/a", "official"), ("https://epubs.icar.org.in/a.pdf", "official"),
    ("https://vikaspedia.in/p", "reputable"), ("https://dept.somecollege.edu/a", "reputable"), ("https://x.ac.in/a", "reputable"),
    ("https://doi.org/10.1000/x", "unknown"), ("https://random-blog.com/p", "unknown"),
    ("https://icar.gov.in.attacker.com/a", "unknown"), ("https://notfao.org/a", "unknown"), ("https://kvkfake.org/a", "unknown")])
def test_trust_tier_is_decided_from_the_host_only(url, tier):
    assert S.trust_tier(url) == tier


def test_min_trust_levels():
    assert S.tier_allowed("official", "official") and not S.tier_allowed("reputable", "official")
    assert S.tier_allowed("reputable", "reputable") and not S.tier_allowed("unknown", "reputable")
    assert S.tier_allowed("unknown", "any")


def test_source_type_classification():
    assert S.classify_source("https://agritech.tnau.ac.in/x", None, "direct_url", "text/html") == "official_guidance"
    assert S.classify_source("https://epubs.icar.org.in/a.pdf", "10.1/x", "openalex", "application/pdf") == "research_paper"
    assert S.classify_source("https://vikaspedia.in/x", None, "direct_url", "text/html") == "informational_article"


# ---------------------------------------------------------------- robots.txt and politeness
def test_robots_disallow_is_respected(web, tmp_path, pipe):
    W._ROBOTS.clear()
    web["/robots.txt"] = (200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /private/\n")
    web["/private/page.html"] = (200, {"Content-Type": "text/html"}, ARTICLE.format(LONG).encode())
    web["/open/page.html"] = (200, {"Content-Type": "text/html"}, ARTICLE.format(LONG).encode())
    store = ResearchStore(tmp_path)
    blocked = W.ingest_url("http://agritech.tnau.ac.in/private/page.html", store, pipe)
    assert blocked["status"] == "failed" and "robots.txt" in blocked["reason"] and "/private/page.html" not in SEEN
    assert W.ingest_url("http://agritech.tnau.ac.in/open/page.html", store, pipe)["status"] == "ok"


def test_robots_5xx_means_do_not_crawl_but_404_means_allowed(web, tmp_path, pipe):
    W._ROBOTS.clear()
    web["/robots.txt"] = (503, {}, b"down")
    web["/a.html"] = (200, {"Content-Type": "text/html"}, ARTICLE.format(LONG).encode())
    assert "robots.txt unavailable" in W.ingest_url("http://agritech.tnau.ac.in/a.html", ResearchStore(tmp_path), pipe)["reason"]
    W._ROBOTS.clear()
    web.pop("/robots.txt")                                       # 404 -> no rules -> allowed
    assert W.ingest_url("http://agritech.tnau.ac.in/a.html", ResearchStore(tmp_path / "2"), pipe)["status"] == "ok"


def test_unchanged_documents_are_not_downloaded_twice(web, tmp_path, pipe):
    W._ROBOTS.clear()
    web["/c.html"] = (200, {"Content-Type": "text/html"}, ARTICLE.format(LONG).encode())
    store = ResearchStore(tmp_path)
    assert W.ingest_url("http://agritech.tnau.ac.in/c.html", store, pipe)["status"] == "ok"
    SEEN.clear()
    again = W.ingest_url("http://agritech.tnau.ac.in/c.html", store, pipe)
    assert again["status"] == "cached" and "/c.html" not in SEEN and len(store.sources()) == 1


# ---------------------------------------------------------------- HTML extraction quality
def test_boilerplate_is_removed_and_main_content_kept():
    raw, meta = W.html_to_document((
        "<html><head><title>T</title></head><body><nav><a>Home</a><a>Menu item that is long enough to look like text</a></nav>"
        "<div class='cookie-banner'>We use cookies on this site, please accept all cookies to continue browsing</div>"
        "<div id='sidebar-ads' class='advertisement'>Buy fertilizer now with a huge discount today only for you</div>"
        "<article><h1>Soil</h1><p>" + LONG + "</p></article><footer>Privacy policy and terms of use footer text here</footer></body></html>").encode(), "http://x")
    text = " ".join(b.text for b in raw.pages[0].blocks)
    assert "Tomato grows well" in text and all(w not in text for w in ("cookies", "discount", "Privacy policy", "Menu item"))


def test_encoding_is_decoded_correctly():
    latin = "<html><head><meta charset='iso-8859-1'><title>Café</title></head><body><article><p>Crème soils of São Paulo " + "x" * 5 + "</p></article></body></html>".encode("latin-1").decode("latin-1")
    assert "Café" in W.decode_html(latin.encode("latin-1"), None) and "São Paulo" in W.decode_html(latin.encode("latin-1"), "ISO-8859-1")
    ta = "<html><head><meta charset='utf-8'></head><body>தக்காளி மண்</body></html>".encode("utf-8")
    assert "தக்காளி" in W.decode_html(ta, None)
    assert "தக்காளி" in W.decode_html(ta, "utf-8")


def test_non_english_and_too_short_pages_are_rejected(web, tmp_path, pipe):
    W._ROBOTS.clear()
    web["/ta.html"] = (200, {"Content-Type": "text/html; charset=utf-8"}, ARTICLE.format("தக்காளி நன்கு வடிகால் உள்ள மண்ணில் நன்றாக வளரும். " * 20).encode())
    web["/short.html"] = (200, {"Content-Type": "text/html"}, ARTICLE.format("Tomato likes loam soil.").encode())
    store = ResearchStore(tmp_path)
    assert "not in English" in W.ingest_url("http://agritech.tnau.ac.in/ta.html", store, pipe)["reason"]
    assert "too little readable text" in W.ingest_url("http://agritech.tnau.ac.in/short.html", store, pipe)["reason"]
    assert store.records() == []


def test_html_is_processed_as_html_not_as_pdf(web, tmp_path, pipe):
    W._ROBOTS.clear()
    web["/h.html"] = (200, {"Content-Type": "text/html"}, ARTICLE.format(LONG).encode())
    o = W.ingest_url("http://agritech.tnau.ac.in/h.html", ResearchStore(tmp_path), pipe)
    assert o["status"] == "ok" and o["source_type"] == "official_guidance" and not list((tmp_path / "downloads").glob("*")) if (tmp_path / "downloads").exists() else o["status"] == "ok"


# ---------------------------------------------------------------- query generation + provider orchestration
def test_queries_come_from_the_question_not_from_a_crop_table():
    q = S.build_queries("Which type of soil is good for growing tomatoes?", ["tomato"])
    assert q[0] == "soil tomatoes" and any("pH drainage texture organic matter" in x for x in q)
    w = S.build_queries("How much irrigation does wheat need?", ["wheat"])
    assert any("irrigation water requirement" in x for x in w) and all("tomato" not in x for x in w)


def test_discover_merges_dedupes_ranks_and_tolerates_a_failing_provider(monkeypatch):
    oa = {"results": [{"title": "TNAU-hosted tomato paper", "publication_year": 2020, "doi": "https://doi.org/10.1/p", "authorships": [],
                       "primary_location": {"landing_page_url": "https://agritech.tnau.ac.in/t.html"}, "best_oa_location": {}},
                      {"title": "Blog post", "publication_year": 2021, "authorships": [], "primary_location": {"landing_page_url": "https://blog.example.com/t"}}]}
    seen = []
    def fake(url, host):
        seen.append(url)
        if "openalex" in url: return oa
        raise S.SearchUnavailable("api.crossref.org: HTTP 503")
    monkeypatch.setattr(S, "_get_json", fake)
    cands, report = S.discover("Which type of soil is good for growing tomatoes?", ["tomato"])
    assert cands[0].landing_url == "https://agritech.tnau.ac.in/t.html"                      # official source ranked first
    assert len({c.landing_url for c in cands}) == len(cands)
    assert {r["provider"]: r["status"] for r in report} == {"openalex": "ok", "crossref": "failed"}
    assert all("open_access.is_oa:true" in u for u in seen if "openalex" in u)               # only readable (open-access) works
    assert not any("key" in u.lower() or "token" in u.lower() for u in seen)                  # no credentials anywhere


def test_discover_raises_when_every_provider_fails(monkeypatch):
    monkeypatch.setattr(S, "_get_json", lambda *a, **k: (_ for _ in ()).throw(S.SearchUnavailable("offline")))
    with pytest.raises(S.SearchUnavailable, match="offline"):
        S.discover("soil for tomato", ["tomato"])


# ---------------------------------------------------------------- Test G: publishing is a transaction
@pytest.fixture()
def publish_env(tmp_path, pipe, web):
    from sawvial.pipeline import run_add_pdfs, run_embed_and_index
    cfg = pipe.cfg; cfg.output_dir, cfg.index_dir = str(tmp_path / "ds"), str(tmp_path / "ix"); (tmp_path / "ds").mkdir()
    seed = tmp_path / "seed.pdf"; make_text_pdf(seed, crop="maize")
    run_add_pdfs(cfg, [seed]); run_embed_and_index(cfg)
    W._ROBOTS.clear()
    pdf = tmp_path / "gn.pdf"; make_text_pdf(pdf, crop="groundnut")
    web["/gn.pdf"] = (200, {"Content-Type": "application/pdf"}, pdf.read_bytes())
    store = ResearchStore(tmp_path / "rs")
    assert W.ingest_url("http://epubs.icar.org.in/gn.pdf", store, pipe)["status"] == "ok"
    from sawvial.eligibility import is_usable_evidence
    ids = [r["id"] for r in store.records() if is_usable_evidence(r["knowledge"]["evidence"], r["source"].get("title"))]
    store.set_status(ids, "verified", "Dr. Reviewer")
    return {"store": store, "ds": tmp_path / "ds", "ix": tmp_path / "ix", "ids": ids}


def snapshot(d: Path):
    return {n: (d / n).read_bytes() for n in ("documents.jsonl", "knowledge_units.jsonl", "qa.jsonl", "dataset_statistics.json")}


def test_G_failed_index_rebuild_rolls_everything_back(publish_env, pipe, monkeypatch):
    import sawvial.pipeline as P
    from sawvial.index.retriever import load_index_bundle
    before, idx_before = snapshot(publish_env["ds"]), (publish_env["ix"] / "index_metadata.json").read_text()
    monkeypatch.setattr(P, "run_embed_and_index", lambda cfg: (_ for _ in ()).throw(RuntimeError("disk full")))
    res = W.publish_verified(publish_env["store"], publish_env["ds"], publish_env["ix"], pipe)
    assert res["published"] == 0 and res["rolled_back"] and "disk full" in res["error"]
    assert snapshot(publish_env["ds"]) == before                                           # dataset restored byte-for-byte
    assert (publish_env["ix"] / "index_metadata.json").read_text() == idx_before          # previous index untouched
    assert not load_index_bundle(publish_env["ix"], publish_env["ds"])["stale"]
    recs = publish_env["store"].records()
    assert all(not r["quality"].get("published_as") for r in recs)                        # NOT marked published
    assert {r["quality"]["validation_status"] for r in recs if r["id"] in publish_env["ids"]} == {"verified"}
    assert list((publish_env["store"].downloads).glob("*.pdf"))                            # downloaded research kept


def test_G_retry_after_failure_succeeds_and_is_searchable(publish_env, pipe):
    from sawvial.index.passages import retrieve_passages
    from sawvial.index.retriever import load_index_bundle
    res = W.publish_verified(publish_env["store"], publish_env["ds"], publish_env["ix"], pipe)
    assert res["published"] >= 1 and res["index_rebuilt"]
    b = load_index_bundle(publish_env["ix"], publish_env["ds"])
    assert not b["stale"]
    r = retrieve_passages(b, "Is drip irrigation recommended for groundnut?", 3)
    assert r["passages"][0]["id"].startswith("SWR_") and r["passages"][0]["status"] == "verified"
    assert all(x["quality"].get("published_as") for x in publish_env["store"].records() if x["id"] in publish_env["ids"])
    assert W.publish_verified(publish_env["store"], publish_env["ds"], publish_env["ix"], pipe)["published"] == 0


def test_G_pending_records_never_enter_the_answerable_index(publish_env, pipe):
    from sawvial.index.retriever import load_index_bundle
    pending_before = [r for r in publish_env["store"].records() if r["quality"]["validation_status"] == "pending_review"]
    W.publish_verified(publish_env["store"], publish_env["ds"], publish_env["ix"], pipe)
    ids = set(load_index_bundle(publish_env["ix"], publish_env["ds"])["ids"])
    assert not ({r["id"] for r in pending_before} & ids)
