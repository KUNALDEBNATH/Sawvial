import json
from pathlib import Path

import pytest

from sawvial.validation.qa_checks import conflicting_questions, is_hard, qa_issues, repair_answer, sft_usable
from tests.conftest import make_unit

GOOD = "Drip irrigation increased pod yield of groundnut by 18% over flood irrigation in the summer season at Coimbatore."


def ku(title="Paper title"):
    return make_unit("SWM_000001", GOOD, ["groundnut"], title=title)


def qa(q, a, t="factual"):
    return {"question": q, "answer": a, "question_type": t}


def test_title_only_answer_rejected():
    assert "answer_is_title_or_metadata" in qa_issues(ku("Effect of drip irrigation on groundnut yield and water use"),
                                                      qa("What does the source report about irrigation?", "Effect of drip irrigation on groundnut yield and water use"))


def test_reference_list_and_empty_and_short_rejected():
    ref = "Rao, K. S., Kumar, P. R., & Das, M. 2019. Irrigation of groundnut. Agricultural Water Management 120, 45-52."
    assert is_hard(qa_issues(ku(), qa("What does the source report about irrigation?", ref)))
    assert "empty_answer" in qa_issues(ku(), qa("What does the source report about irrigation?", ""))
    assert is_hard(qa_issues(ku(), qa("What does the source report about irrigation?", "Yes.")))


def test_off_topic_answer_rejected():
    i = qa_issues(ku(), qa("What does the source report about mulching?", GOOD))
    assert "answer_off_topic" in i and is_hard(i)


def test_good_pair_passes_and_unverified_label_is_applied_by_generator():
    assert not is_hard(qa_issues(ku(), qa("What did the source report about drip irrigation of groundnut?", GOOD)))


def test_repair_trims_fragments_without_inventing_text():
    frag = "ing in loamy soils. Drip irrigation increased pod yield by 18% in summer. The next sentence was cut off by"
    fixed = repair_answer(frag)
    assert fixed == "Drip irrigation increased pod yield by 18% in summer." and fixed in frag
    assert repair_answer("no terminal sentence boundary here at all") == ""


def test_sft_usable_returns_repaired_answer():
    a, _ = sft_usable(ku(), qa("What did the source report about drip irrigation of groundnut?",
                              "ing in loamy soils of the region. " + GOOD + " The following sentence is truncated mid"))
    assert a == GOOD


def test_conflicting_questions_detected():
    rows = [qa("Q?", "a1"), qa("Q?", "a2"), qa("Other?", "a3")]
    assert conflicting_questions(rows) == {"q?"}


REPO = Path(__file__).resolve().parent.parent


def test_heldout_documents_never_in_sft_training_data():
    """The evaluation hold-out is exactly the documents excluded from SFT training."""
    import sys; sys.path.insert(0, str(REPO / "llm"))
    from common import is_heldout_doc
    train = REPO / "llm" / "sft" / "train.jsonl"
    if not train.exists():
        pytest.skip("llm/sft/train.jsonl not present")
    docs = {json.loads(l)["meta"]["doc"] for l in open(train, encoding="utf-8") if l.strip()}
    assert docs and not any(is_heldout_doc(d, 5) for d in docs)
