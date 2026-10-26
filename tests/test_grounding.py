from sawvial import grounding as g

P = [{"id": "SWM_000001", "document_id": "d1", "text": "Drip irrigation increased water-use efficiency by 24% compared with flood irrigation in cotton during the kharif season."},
     {"id": "SWM_000002", "document_id": "d2", "text": "Mulching with straw reduced soil evaporation and conserved moisture in wheat fields."}]


def test_supported_sentence_passes():
    a = g.assess("Drip irrigation raised water use efficiency by 24% over flood irrigation in cotton [SWM_000001].", P)
    assert a["n_claims"] == 1 and a["n_supported"] == 1 and not a["invalid_citations"]


def test_fabricated_id_is_rejected_and_reported():
    a = g.assess("Biochar is great for every crop [SWM_999999].", P)
    assert a["invalid_citations"] == ["SWM_999999"] and a["n_supported"] == 0
    assert g.compose_supported_answer(a) == ("", [])


def test_uncited_claim_not_supported():
    a = g.assess("Cotton needs a lot of water in summer.", P)
    assert a["sentences"][0]["reason"] == "no_citation" and a["n_supported"] == 0


def test_invented_number_is_rejected():
    a = g.assess("Mulching cut evaporation by 60% in wheat [SWM_000002].", P)
    assert not a["sentences"][0]["supported"] and a["sentences"][0]["reason"].startswith("number_not_in_evidence")


def test_citation_to_wrong_passage_is_rejected():
    a = g.assess("Drip irrigation raised water use efficiency by 24% in cotton [SWM_000002].", P)
    assert not a["sentences"][0]["supported"]


def test_only_supported_sentences_survive_and_ids_are_real():
    raw = ("Drip irrigation raised water use efficiency by 24% over flood irrigation in cotton [SWM_000001]. "
           "Plant trees for shade [SWM_000001]. Unrelated [SWM_999999].")
    text, ids = g.compose_supported_answer(g.assess(raw, P))
    assert "trees" not in text and "[SWM" not in text and ids == ["SWM_000001"]


def test_refusal_is_not_a_claim():
    a = g.assess("The provided sources do not contain enough information to answer this question.", P)
    assert a["is_refusal"] and a["n_claims"] == 0


def test_extractive_answer_is_verbatim_and_none_when_unrelated():
    out = g.extractive_answer("effect of drip irrigation on water use efficiency in cotton", P)
    assert out and "24%" in out[0] and out[1]["id"] == "SWM_000001"
    assert g.extractive_answer("tractor engine maintenance schedule", P) is None


def test_evidence_level():
    assert g.evidence_level(["SWM_000001"], P) == "single_source"
    assert g.evidence_level(["SWM_000001", "SWM_000002"], P) == "multiple_sources"
