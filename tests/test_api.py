import types

import pytest
from fastapi.testclient import TestClient

import api
from sawvial.index.retriever import load_index_bundle


@pytest.fixture()
def client(built, monkeypatch):
    monkeypatch.setitem(api.CFG, "mock", False)
    api.STATE.clear()
    api.STATE.update(bundle=load_index_bundle(built["index"], built["dataset"]), embedder=None, tok=None,
                     model=None, translator=None, loaded=True, llm_status={"mode": "adapter", "adapter_status": "ok"})
    api.LIMITER.hits.clear()
    yield TestClient(api.app)
    api.STATE.clear()


def set_llm(fn):
    api.STATE["rag"] = types.SimpleNamespace(answer=fn)


def good_llm(tok, model, q, passages, mx, history=None, cancel_event=None):
    p = passages[0]
    return f"{p['text'].split('. ')[0].rstrip('.')} [{p['id']}]."


def test_normal_question_is_grounded_with_real_sources(client):
    set_llm(good_llm)
    j = client.post("/api/ask", json={"question": "mulching moisture conservation in rainfed farming"}).json()
    assert j["grounded"] and j["grounding"]["status"] == "supported" and "[SWM" not in j["answer"]
    assert [s["id"] for s in j["sources"] if s["cited"]] == ["SWM_000005"]
    s = j["sources"][0]
    assert s["title"] == "A study" and s["authors"] == ["A. Rao"] and s["year"] == 2022 and s["page"] == 3
    assert set(j["timings_ms"]) >= {"retrieval", "generation", "grounding", "total"}


def test_fabricated_citation_and_unsupported_sentences_are_removed(client):
    def bad(tok, model, q, passages, mx, history=None, cancel_event=None):
        p = passages[0]
        return (f"{p['text'].split('. ')[0].rstrip('.')} [{p['id']}]. Yields doubled everywhere [{p['id']}]. "
                "Biochar fixes everything [ZZZ_000001].")
    set_llm(bad)
    j = client.post("/api/ask", json={"question": "mulching moisture conservation in rainfed farming"}).json()
    assert j["grounding"]["status"] == "partially_supported" and j["grounding"]["invalid_citations"] == ["ZZZ_000001"]
    assert "doubled" not in j["answer"] and "Biochar" not in j["answer"]
    assert "partially_supported" in j["grounding"]["notes"]
    assert all(s["id"] in {"SWM_000005", "SWM_000004", "SWM_000001", "SWM_000002", "SWM_000003", "SWM_000006"} for s in j["sources"])


def test_model_output_with_nothing_supported_falls_back_to_quoted_evidence(client):
    set_llm(lambda *a, **k: "Totally invented advice with no citations at all.")
    j = client.post("/api/ask", json={"question": "mulching moisture conservation in rainfed farming"}).json()
    assert j["grounding"]["status"] == "extractive_fallback" and j["answer"].startswith("According to the retrieved")
    assert "extractive_fallback" in j["grounding"]["notes"]


def test_crop_without_evidence_says_so_and_skips_the_llm(client):
    def must_not_run(*a, **k): raise AssertionError("LLM must not run without evidence")
    set_llm(must_not_run)
    j = client.post("/api/ask", json={"question": "How much water do mango orchards need?"}).json()
    assert not j["grounded"] and j["grounding"]["status"] == "insufficient_evidence" and j["sources"] == []
    assert "mango" in j["answer"] and "does not currently contain enough evidence" in j["answer"]


def test_out_of_scope_and_greeting(client):
    set_llm(lambda *a, **k: 1 / 0)
    assert client.post("/api/ask", json={"question": "What is the capital of France?"}).json()["grounding"]["reason"] == "out_of_scope"
    assert not client.post("/api/ask", json={"question": "hello"}).json()["grounded"]


def test_source_links_are_sanitised(client, built):
    from tests.conftest import make_unit
    p = {"id": "SWM_000001", "doi": "doi: 10.1000/xyz.9.", "url": "javascript:alert(1)", "score": 0.5, "text": "t"}
    e = api.source_entry(p, True)
    assert e["doi"] == "10.1000/xyz.9" and e["doi_url"] == "https://doi.org/10.1000/xyz.9" and e["url"] is None
    assert api.source_entry({**p, "url": "https://x.org/a.pdf", "doi": None}, False)["url"] == "https://x.org/a.pdf"
    assert api.source_entry({"id": "U", "score": 0, "text": ""}, False)["doi_url"] is None      # nothing invented


def test_request_validation_and_limits(client):
    set_llm(good_llm)
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400
    assert client.post("/api/ask", json={"question": "x" * 1100}).status_code == 400
    assert client.post("/api/ask", json={"question": "soil", "top_k": 99}).status_code == 422
    assert client.post("/api/ask", json={"question": "soil", "history": [{"role": "system", "content": "x"}]}).status_code == 422


def test_rate_limit(client, monkeypatch):
    set_llm(good_llm)
    monkeypatch.setattr(api.LIMITER, "per_minute", 3)
    codes = [client.post("/api/ask", json={"question": "mulching moisture"}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]


def test_not_ready_returns_503(client):
    api.STATE["loaded"] = False
    assert client.post("/api/ask", json={"question": "soil"}).status_code == 503


def test_lock_not_held_during_retrieval_or_grounding(client, monkeypatch):
    seen = {}
    real = api.retrieve_passages
    def spy(*a, **k):
        seen["retrieval_locked"] = api.INFER_LOCK.locked(); return real(*a, **k)
    monkeypatch.setattr(api, "retrieve_passages", spy)
    def llm(*a, **k):
        seen["generation_locked"] = api.INFER_LOCK.locked(); return good_llm(*a, **k)
    set_llm(llm)
    client.post("/api/ask", json={"question": "mulching moisture conservation"})
    assert seen == {"retrieval_locked": False, "generation_locked": True}


def test_missing_translator_degrades_to_english(client):
    set_llm(good_llm)
    j = client.post("/api/ask", json={"question": "mulching moisture conservation", "language": "hi"}).json()
    assert j["language"] == "hi" and j["answer_language"] == "en" and j["note"] == "translation_unavailable"


def test_health_reports_adapter_and_index(client):
    h = client.get("/api/health").json()
    assert h["llm"]["mode"] == "adapter" and h["index"]["n_units"] == 6 and h["answerable_statuses"] == ["validated", "verified"]
    assert isinstance(h["speech_local"], list)


def test_cors_is_not_wildcard(client):
    r = client.options("/api/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") != "*" and "evil.example" not in r.headers.get("access-control-allow-origin", "")


def test_speech_unavailable_language_returns_clear_503(client, monkeypatch):
    import speech
    monkeypatch.setattr(speech, "is_cached", lambda lang: False)
    r = client.post("/api/speech/stream", json={"text": "hello", "language": "ta"})
    assert r.status_code == 503 and "download_models" in r.json()["detail"]
