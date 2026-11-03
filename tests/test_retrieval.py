from sawvial.index.passages import retrieve_passages
from sawvial.index.retriever import detect_crops, expand_query, load_index_bundle


def bundle(built):
    return load_index_bundle(built["index"], built["dataset"])


def test_crop_question_returns_only_that_crop(built):
    r = retrieve_passages(bundle(built), "How does drip irrigation affect groundnut?", 4)
    assert r["crops"] == ["groundnut"] and not r["crop_missing"]
    assert r["passages"] and all(p["crop_labeled"] for p in r["passages"])
    assert {p["id"] for p in r["passages"]} <= {"SWM_000001", "SWM_000002"}


def test_no_generic_substitution_when_crop_has_no_evidence(built):
    # mango exists only as needs_review (not answerable) -> must say "no evidence", not return generic irrigation text
    r = retrieve_passages(bundle(built), "How much drip irrigation water do mango orchards need?", 4)
    assert r["crops"] == ["mango"] and r["crop_missing"] and r["passages"] == []


def test_synonyms_paddy_corn_peanut():
    assert detect_crops("paddy water saving") == ["rice"]
    assert detect_crops("corn nitrogen") == ["maize"]
    assert detect_crops("peanut irrigation") == ["groundnut"]
    assert detect_crops("Which crops need less water?") == []          # multi-crop question: no gating
    assert "water use efficiency" in expand_query("how to raise WUE")


def test_listing_a_crop_among_many_is_not_crop_evidence(built):
    # SWM_000007 (needs_review anyway) lists banana among many crops; even if it were indexed it must not count
    r = retrieve_passages(bundle(built), "banana water requirement", 4)
    assert r["crop_missing"]


def test_generic_question_not_crop_gated(built):
    r = retrieve_passages(bundle(built), "mulching moisture conservation", 3)
    assert r["crops"] == [] and r["passages"][0]["id"] == "SWM_000005"


def test_passages_carry_traceable_metadata(built):
    p = retrieve_passages(bundle(built), "cotton drip water use efficiency", 2)["passages"][0]
    for k in ("id", "document_id", "title", "authors", "year", "journal", "doi", "url", "page", "text", "score"):
        assert k in p
    assert p["id"] == "SWM_000006" and p["page"] == 3 and p["authors"] == ["A. Rao"]


# ---- relevance: the passage must be about what the QUESTION asks, not just mention the crop ------------------
def _local_bundle(*texts):
    from sawvial.index.retriever import bundle_from_units
    from tests.conftest import make_unit
    us = [make_unit(f"SWM_0002{i:02d}", t, ["tomato"], doc=f"d{i}") for i, t in enumerate(texts)]
    return bundle_from_units(us, frozenset({"validated"}), origin="local")


IRRIGATION_ONLY = ("The crop coefficient of tomato starts low and rises during fruit development, so the irrigation water requirement "
                   "of tomato in the field was estimated at several hundred mm over the whole growing season of the crop.")
SOIL_TEXT = ("Tomato grows best in well drained sandy loam soils rich in organic matter with a soil pH between six and seven, "
             "while heavy clay soils with poor drainage are unsuitable for tomato.")


def test_crop_matched_but_wrong_aspect_passages_do_not_count_as_evidence():
    r = retrieve_passages(_local_bundle(IRRIGATION_ONLY), "Which type of soil is good for growing tomatoes?", 4)
    assert r["passages"] == [] and r["aspect"] == "soil suitability" and r["aspect_unmet"] and not r["crop_missing"]


def test_passage_about_the_asked_aspect_is_kept_and_wrong_ones_dropped():
    r = retrieve_passages(_local_bundle(IRRIGATION_ONLY, SOIL_TEXT), "Which type of soil is good for growing tomatoes?", 4)
    assert [p["text"] for p in r["passages"]] == [SOIL_TEXT] and r["passages"][0]["rank"] == 1 and not r["aspect_unmet"]


def test_other_questions_about_the_same_crop_still_use_the_irrigation_passage():
    r = retrieve_passages(_local_bundle(IRRIGATION_ONLY), "How much water does tomato need?", 4)
    assert len(r["passages"]) == 1 and r["aspect"] == "water requirement"
