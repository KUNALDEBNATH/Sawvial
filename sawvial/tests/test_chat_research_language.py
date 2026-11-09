"""Scenarios A-G from the task: language handling, online research inside /api/ask, citations, speech, publishing.
Faked: the LLM, the translation model and the search-provider JSON. REAL: fetching (SSRF guard, redirects, robots),
HTML/PDF extraction, trust tiers, retrieval/crop gating, grounding, storage, publishing."""
import json
import types

import pytest
from fastapi.testclient import TestClient

import api
from sawvial import settings
from sawvial.index.retriever import load_index_bundle
from sawvial.research import live as L
from sawvial.research import search as S
from sawvial.research.store import ResearchStore
from tests.netfixture import web  # noqa: F401

EN_Q = "Which type of soil is good for growing tomatoes?"
TA_Q = "தக்காளி வளர்க்க எந்த மண் நல்லது?"

# Fixture pages written for these tests (NOT copies of any real website).
TNAU_PAGE = ("<html><head><title>Tomato - soil and climate</title><meta name='og:site_name' content='Agri Portal'></head><body>"
             "<nav><ul><li>Home</li><li>Crops</li><li>Contact the university office</li></ul></nav>"
             "<article><h1>Tomato cultivation</h1><h2>Soil</h2>"
             "<p>Tomato grows well in a wide range of soils, but well drained sandy loam and red loam soils rich in organic matter "
             "are best, with a soil pH of 6.0 to 7.0 for good yield of the tomato crop.</p>"
             "<p>Waterlogged or heavy clay soils with poor drainage should be avoided because they favour root diseases and "
             "reduce fruit set in tomato crops grown in the field during the rainy season.</p>"
             "<p>Add well decomposed farmyard manure before planting tomato seedlings to improve soil structure and moisture.</p>"
             "</article><footer>Copyright notice that must never reach the evidence</footer></body></html>").encode()
ICAR_PAGE = ("<html><head><title>Soil requirement of tomato under open field conditions</title>"
             "<meta name='citation_title' content='Soil requirement of tomato under open field conditions'>"
             "<meta name='citation_author' content='S. Devi'><meta name='citation_publication_date' content='2019/03/01'>"
             "<meta name='citation_journal_title' content='Indian Journal of Horticulture'></head><body><article>"
             "<h1>Soil requirement of tomato</h1><p>In this study tomato performed best in deep fertile loam soils with a soil pH of "
             "5.5 to 6.8 and good drainage, and yield declined on saline and waterlogged soils in the tomato trials.</p>"
             "<p>The tomato plots were irrigated by furrows and organic matter content of the loam soil was kept above one percent "
             "throughout the experiment which was repeated over two seasons at the research station.</p></article></body></html>").encode()
SHADY_PAGE = TNAU_PAGE.replace(b"Tomato cultivation", b"Tomato miracle soil").replace(b"6.0 to 7.0", b"1.0 to 2.0")


def cand(title, url, provider="openalex", **kw):
    return S.Candidate(title=title, landing_url=url, provider=provider, **kw)


class FakeTranslator:
    """Records every call; Tamil question -> English via a table; English -> 'Tamil' by prefixing Tamil text."""
    def __init__(self):
        self.calls = []

    def translate(self, text, src, tgt):
        self.calls.append((src, tgt, text))
        if src == "tam_Taml" and tgt == "eng_Latn":
            return {TA_Q: EN_Q}.get(text, "translated: " + text)
        if tgt == "tam_Taml":
            return "தமிழ்: " + text
        return text


def fake_llm(tok, model, q, passages, mx, history=None, cancel_event=None):
    fake_llm.seen = {"q": q, "ids": [p["id"] for p in passages]}
    p = passages[0]
    return f"{p['text'].split('. ')[0].strip().rstrip('.')} [{p['id']}]."


@pytest.fixture()
def app_env(built, tmp_path, monkeypatch, web):
    monkeypatch.setattr(settings, "ONLINE_RESEARCH", True)
    monkeypatch.setattr(settings, "RESEARCH_DIR", tmp_path / "research")
    monkeypatch.setattr(settings, "RESEARCH_MIN_TRUST", "reputable")
    monkeypatch.setitem(api.CFG, "mock", False)
    api.STATE.clear()
    tr = FakeTranslator()
    api.STATE.update(bundle=load_index_bundle(built["index"], built["dataset"]), embedder=None, tok=None, model=None,
                     translator=tr, loaded=True, llm_status={"mode": "adapter", "adapter_status": "ok"},
                     rag=types.SimpleNamespace(answer=fake_llm))
    api.LIMITER.hits.clear()
    web["/tomato-soil.html"] = (200, {"Content-Type": "text/html; charset=utf-8"}, TNAU_PAGE)
    web["/soil-requirement.html"] = (200, {"Content-Type": "text/html"}, ICAR_PAGE)
    web["/miracle.html"] = (200, {"Content-Type": "text/html"}, SHADY_PAGE)
    log = []
    monkeypatch.setattr(L, "discover", lambda q, crops, max_results=12, provider=None: (
        log.append(("discover", q)) or ([cand("Tomato - soil and climate", "http://agritech.tnau.ac.in/tomato-soil.html", provider="direct_url"),
                                         cand("Soil requirement of tomato", "http://epubs.icar.org.in/soil-requirement.html",
                                              provider="openalex", doi="10.1000/tom.1", year=2019, authors=["S. Devi"],
                                              venue="Indian Journal of Horticulture")],
                                        [{"provider": "openalex", "status": "ok", "results": 2, "error": None}])))
    real = api.retrieve_passages
    monkeypatch.setattr(api, "retrieve_passages", lambda *a, **k: (log.append(("local", a[1])) or real(*a, **k)))
    yield types.SimpleNamespace(client=TestClient(api.app), tr=tr, log=log, web=web, tmp=tmp_path, mp=monkeypatch)
    api.STATE.clear()


def ask(env, q, **kw):
    r = env.client.post("/api/ask", json={"question": q, **kw})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- Test A: the exact tomato question (English in, Tamil out)
def test_A_tomato_english_question_tamil_answer(app_env):
    j = ask(app_env, EN_Q, answer_language="ta", input_language="en")
    # language handling: an ENGLISH question must never be translated "from Tamil"
    assert j["input_language"] == "en" and j["answer_language"] == "ta" and j["text_language"] == "ta"
    assert [(s, t) for s, t, _ in app_env.tr.calls] == [("eng_Latn", "tam_Taml")]        # only the answer is translated
    assert fake_llm.seen["q"] == EN_Q                                                      # retrieval/LLM got the original text
    # local first, then online research
    assert [k for k, _ in app_env.log] == ["local", "discover"]
    r = j["research"]
    assert r["status"] == "used" and r["attempted"] and {d["status"] for d in r["documents"]} == {"ok"}
    # a Tamil answer for real, built from web evidence
    assert j["answer"].startswith("தமிழ்: ") and j["grounded"] and j["evidence_origin"] == "web"
    assert j["grounding"]["status"] == "supported" and "web_unreviewed" in j["grounding"]["notes"]
    assert all(i.startswith("SWR_") for i in fake_llm.seen["ids"])
    # sources: real metadata + clickable original links
    by_url = {s["url"]: s for s in j["sources"]}
    tnau = by_url["http://agritech.tnau.ac.in/tomato-soil.html"]
    assert tnau["origin"] == "web" and tnau["trust_tier"] == "official" and tnau["source_type"] == "official_guidance"
    assert tnau["title"] == "Tomato - soil and climate" and tnau["doi_url"] is None and tnau["retrieved_at"]
    icar = by_url["http://epubs.icar.org.in/soil-requirement.html"]
    assert icar["source_type"] == "research_paper" and icar["authors"] == ["S. Devi"] and icar["year"] == 2019
    assert icar["doi_url"] == "https://doi.org/10.1000/tom.1"
    # nothing was hard-coded: the answer sentence is verbatim from the fetched page, boilerplate is absent
    assert "Copyright" not in json.dumps(j) and "Contact the university" not in json.dumps(j)
    # collected evidence is stored for review, NEVER verified automatically
    recs = ResearchStore(settings.RESEARCH_DIR).records()
    assert recs and {r["quality"]["validation_status"] for r in recs} == {"pending_review"}


# ---------------------------------------------------------------- Test B: Tamil question, Tamil answer
def test_B_tamil_question_tamil_answer(app_env):
    j = ask(app_env, TA_Q, answer_language="ta")                       # input language detected from the script
    assert j["input_language"] == "ta" and j["text_language"] == "ta"
    assert [(s, t) for s, t, _ in app_env.tr.calls] == [("tam_Taml", "eng_Latn"), ("eng_Latn", "tam_Taml")]
    assert fake_llm.seen["q"] == EN_Q and j["question_en"] == EN_Q
    assert j["grounded"] and any(s["cited"] and s["url"] for s in j["sources"])
    assert j["answer"].startswith("தமிழ்: ")


def test_B2_selected_language_is_not_assumed_to_be_the_input_language(app_env):
    ask(app_env, EN_Q, language="ta")                                  # legacy field = answer language only
    assert all(s != "tam_Taml" for s, _, _ in app_env.tr.calls)


def test_B3_other_input_language_translated_from_its_own_script(app_env):
    ask(app_env, "टमाटर के लिए कौन सी मिट्टी अच्छी है?", answer_language="ta")
    assert app_env.tr.calls[0][:2] == ("hin_Deva", "eng_Latn") and app_env.tr.calls[-1][:2] == ("eng_Latn", "tam_Taml")


# ---------------------------------------------------------------- Test C: local has nothing -> mocked provider article used
def test_C_research_fallback_with_english_answer_and_evidence_compared(app_env):
    j = ask(app_env, EN_Q, answer_language="en")
    assert j["text_language"] == "en" and not app_env.tr.calls and j["research"]["status"] == "used"
    cmp = j["research"]["comparison"]
    assert cmp["n_sources"] == 2 and set(cmp["source_types"]) == {"official_guidance", "research_paper"}
    ph = [d for d in cmp["differences"] if d["quantity"].lower() == "ph"]
    assert ph and {(v["low"], v["high"]) for v in ph[0]["values"]} == {(6.0, 7.0), (5.5, 6.8)}      # sources differ: reported
    # both stages of a research-backed answer are visible to the client
    assert "differing_values" in j["grounding"]["notes"] or j["grounding"]["evidence_level"] == "single_source"


def test_C2_local_evidence_wins_and_no_research_when_sufficient(app_env):
    j = ask(app_env, "mulching moisture conservation in rainfed farming", answer_language="en")
    assert [k for k, _ in app_env.log] == ["local"] and j["research"]["status"] == "not_needed" and j["evidence_origin"] == "local"


def test_C3_research_always_and_never_flags(app_env):
    ask(app_env, "mulching moisture conservation in rainfed farming", research="always")
    assert "discover" in [k for k, _ in app_env.log]
    app_env.log.clear()
    j = ask(app_env, EN_Q, research="never")
    assert [k for k, _ in app_env.log] == ["local"] and not j["grounded"] and j["research"]["attempted"] is False


# ---------------------------------------------------------------- Test D: no evidence / no internet / translation unavailable
def test_D_search_failure_is_reported_honestly(app_env):
    app_env.mp.setattr(L, "discover", lambda *a, **k: (_ for _ in ()).throw(S.SearchUnavailable("openalex: cannot resolve host")))
    j = ask(app_env, EN_Q, answer_language="en")
    assert not j["grounded"] and j["sources"] == [] and j["research"]["status"] == "search_failed"
    assert "online search could not be completed" in j["answer"] and "cannot resolve host" in j["answer"]
    assert j["grounding"]["status"] == "insufficient_evidence"


def test_D2_untrusted_pages_and_doi_redirects_are_not_used(app_env):
    app_env.web["/10.1000/tomato"] = (302, {"Location": "http://shady.test/miracle.html"}, b"")
    app_env.mp.setattr(L, "discover", lambda *a, **k: ([cand("Via DOI", "http://doi.org/10.1000/tomato", provider="crossref", doi="10.1000/tomato"),
                                                        cand("Blog", "http://blog.example.com/miracle.html")],
                                                       [{"provider": "crossref", "status": "ok", "results": 2, "error": None}]))
    j = ask(app_env, EN_Q, answer_language="en")
    assert not j["grounded"] and j["research"]["status"] == "no_usable_documents"
    reasons = " | ".join(d["reason"] or "" for d in j["research"]["documents"])
    assert "not a trusted source" in reasons and "redirected from doi.org" in reasons
    assert ResearchStore(settings.RESEARCH_DIR).records() == []              # untrusted content is not even stored


def test_D3_translation_unavailable_is_visible_not_silent(app_env):
    api.STATE["translator"] = None
    j = ask(app_env, EN_Q, answer_language="ta")
    assert j["answer_language"] == "ta" and j["text_language"] == "en"           # English is NOT labelled Tamil
    assert "translation_unavailable" in j["warnings"] and j["translation"]["available"] is False
    assert j["grounded"] and j["answer"]                                          # the evidence-based answer is still returned
    t = ask(app_env, TA_Q, answer_language="ta")                                  # cannot understand Tamil without the model
    assert not t["grounded"] and t["text_language"] == "en" and "input_translation_unavailable" in t["warnings"]


def test_D4_every_early_return_has_consistent_language_metadata(app_env):
    for q in ("hello", "What is the capital of France?"):
        j = ask(app_env, q, answer_language="ta")
        assert j["text_language"] == "ta" and j["answer"].startswith("தமிழ்: ") and j["input_language"] == "en"
    api.STATE["translator"] = None
    j = ask(app_env, "hello", answer_language="ta")
    assert j["text_language"] == "en" and "translation_unavailable" in j["warnings"]
    app_env.mp.setitem(api.CFG, "mock", True)
    j = ask(app_env, "anything", answer_language="ta")
    assert j["text_language"] == "en" and "translation_unavailable" in j["warnings"]
    assert ask(app_env, "anything", answer_language="bn")["text_language"] == "bn"


def test_D5_cancelled_request_is_translated_consistently(app_env):
    import threading
    ev = threading.Event(); ev.set()
    res = api.run_ask(EN_Q, "ta", 4, cancel_event=ev)
    assert res["grounding"]["status"] == "cancelled" and res["text_language"] == "ta"


# ---------------------------------------------------------------- Test E: citations
def test_E_unsupported_claims_and_fake_citations_never_marked_grounded(app_env):
    def liar(tok, model, q, passages, mx, history=None, cancel_event=None):
        p = passages[0]
        return (f"{p['text'].split('. ')[0].rstrip('.')} [{p['id']}]. Tomato yield doubles with miracle soil [{p['id']}]. "
                "Always use unlimited pesticide [SWR_999999].")
    api.STATE["rag"] = types.SimpleNamespace(answer=liar)
    j = ask(app_env, EN_Q, answer_language="en")
    assert j["grounding"]["status"] == "partially_supported" and j["grounding"]["invalid_citations"] == ["SWR_999999"]
    assert "miracle" not in j["answer"] and "pesticide" not in j["answer"]
    urls = {s["url"] for s in j["sources"]}
    assert urls <= {"http://agritech.tnau.ac.in/tomato-soil.html", "http://epubs.icar.org.in/soil-requirement.html"}   # only real fetches
    for s in j["sources"]:
        assert s["snippet"] and s["id"].startswith("SWR_")


def test_E2_useful_excerpt_is_returned_instead_of_a_generic_refusal(app_env):
    api.STATE["rag"] = types.SimpleNamespace(answer=lambda *a, **k: "Completely made up text without any citation.")
    j = ask(app_env, "tomato", answer_language="en")
    assert j["grounded"] and j["grounding"]["status"] in ("extractive_fallback", "excerpt_only")
    assert "tomato" in j["answer"].lower() and "could not find" not in j["answer"].lower()


# ---------------------------------------------------------------- stages are streamed to the UI
def test_stream_endpoint_reports_real_stages_then_final(app_env):
    r = app_env.client.post("/api/ask/stream", json={"question": EN_Q, "answer_language": "ta"})
    events = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    stages = [e["stage"] for e in events if "stage" in e]
    assert stages[:3] == ["local_search", "online_search", "reading"] and stages[-2:] == ["generating", "translating"]
    assert "final" in events[-1] and events[-1]["final"]["text_language"] == "ta"


def test_stream_validation_errors_are_normal_http_errors(app_env):
    assert app_env.client.post("/api/ask/stream", json={"question": " "}).status_code == 400


# ---------------------------------------------------------------- Test F: Tamil speech request (API side)
def test_F_speech_endpoint_gets_tamil_code_and_actual_text(app_env):
    import speech
    got = {}
    app_env.mp.setattr(speech, "is_cached", lambda lang: True)
    app_env.mp.setattr(speech, "stream_audio", lambda text, language, ev=None: (got.update(text=text, language=language) or iter([b"\x00\x00\x00\x00"])))
    ta_text = "தக்காளிக்கு வடிகால் வசதியுள்ள மணல் கலந்த களிமண் நல்லது."
    r = app_env.client.post("/api/speech/stream", json={"text": ta_text, "language": "ta"})
    assert r.status_code == 200 and got == {"text": ta_text, "language": "ta"} and r.content
    app_env.mp.setattr(speech, "is_cached", lambda lang: False)
    r = app_env.client.post("/api/speech/stream", json={"text": ta_text, "language": "ta"})
    assert r.status_code == 503 and "download_models.py --speech" in r.json()["detail"]


def test_F2_health_reports_translator_speech_and_research(app_env):
    h = app_env.client.get("/api/health").json()
    assert h["translator"] is True and "translator_model_cached" in h and isinstance(h["speech_local"], list)
    assert h["research"]["enabled"] is True and h["research"]["min_trust"] == "reputable" and "device" in h


# ---------------------------------------------------------------- the real-world trap: local KB has the crop, but not the topic
def test_local_tomato_irrigation_evidence_does_not_block_research_for_a_soil_question(app_env):
    from sawvial.index.retriever import bundle_from_units
    from tests.conftest import make_unit
    from tests.test_retrieval import IRRIGATION_ONLY
    api.STATE["bundle"] = bundle_from_units([make_unit("SWM_000300", IRRIGATION_ONLY, ["tomato"])], frozenset({"validated"}), origin="local")
    j = ask(app_env, EN_Q, answer_language="en")
    assert [k for k, _ in app_env.log] == ["local", "discover"]            # local was tried first, found nothing ON TOPIC
    assert j["research"]["status"] == "used" and j["evidence_origin"] == "web" and j["grounded"]
    assert all(i.startswith("SWR_") for i in fake_llm.seen["ids"])           # the irrigation record was not used for a soil answer
    # ... while a question the local evidence DOES cover is answered locally, without going online
    app_env.log.clear()
    j2 = ask(app_env, "How much water does tomato need?", answer_language="en")
    assert [k for k, _ in app_env.log] == ["local"] and j2["evidence_origin"] == "local" and j2["research"]["status"] == "not_needed"


def test_one_source_entry_per_document_with_all_pages_and_ids(app_env):
    j = ask(app_env, EN_Q, answer_language="en")
    urls = [s["url"] for s in j["sources"]]
    assert len(urls) == len(set(urls)) == 2                                     # two documents, however many passages were used
    assert j["sources"][0]["cited"] and all(len(s["ids"]) >= 1 for s in j["sources"])
    assert set(fake_llm.seen["ids"]) == {i for s in j["sources"] for i in s["ids"]}
