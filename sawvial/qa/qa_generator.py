"""
sawvial.qa.qa_generator
=========================
Generates grounded question/answer pairs for a knowledge unit.

Every answer is built directly from fields already present on the
knowledge unit (evidence, finding, structured_facts, context) — never
generated freely — so nothing can be hallucinated. Questions are
template-based per the question types listed in the master spec
(factual, quantitative, comparison, condition-specific, cause-effect,
practice/recommendation, definition).

Q&A is only generated for knowledge units whose scientific-importance
score clears a minimum bar (checked by the caller), keeping the QA set
useful rather than exhaustive.
"""
from __future__ import annotations

from typing import List, Dict, Any


def _topic_phrase(ku: Dict[str, Any]) -> str:
    topics = ku["classification"].get("topics", [])
    if topics:
        return topics[0].replace("_", " ")
    return ku["classification"].get("primary_domain", "this topic").replace("_", " ")


def _context_phrase(ku: Dict[str, Any]) -> str:
    ctx = ku["context"]
    bits = []
    if ctx.get("crop"):
        bits.append(f"in {', '.join(ctx['crop'])}")
    if ctx.get("season"):
        bits.append(f"during {', '.join(ctx['season'])}")
    if ctx.get("agro_climatic_zone"):
        bits.append(f"under {', '.join(ctx['agro_climatic_zone'])} conditions")
    return " ".join(bits)


def generate_qa(ku: Dict[str, Any], max_pairs: int = 3) -> List[Dict[str, str]]:
    qa_pairs: List[Dict[str, str]] = []
    evidence = ku["knowledge"]["evidence"]
    finding = ku["knowledge"]["finding"]
    topic_phrase = _topic_phrase(ku)
    context_phrase = _context_phrase(ku)

    facts = ku.get("structured_facts", [])
    effect_facts = [f for f in facts if f.get("fact_type") == "effect"]

    # Quantitative — grounded directly in a detected effect fact.
    for fact in effect_facts[:2]:
        param = fact.get("parameter", topic_phrase)
        question = f"What effect was observed on {param} related to {topic_phrase}" \
                   f"{(' ' + context_phrase) if context_phrase else ''}?"
        answer = f"{param.capitalize()} showed a {fact.get('effect')} of " \
                 f"{fact.get('magnitude')}{fact.get('unit', '')}, based on: {evidence}"
        qa_pairs.append({"question": question.strip(), "answer": answer.strip(),
                          "question_type": "quantitative"})

    # Condition-specific / factual — grounded in evidence + context.
    if context_phrase and len(qa_pairs) < max_pairs:
        question = f"What was found regarding {topic_phrase} {context_phrase}?"
        qa_pairs.append({"question": question.strip(), "answer": evidence,
                          "question_type": "condition_specific"})

    # Practice/recommendation — only if the source contains an explicit one.
    if ku["knowledge"].get("recommendation") and len(qa_pairs) < max_pairs:
        question = f"What is recommended regarding {topic_phrase}?"
        qa_pairs.append({"question": question.strip(), "answer": ku["knowledge"]["recommendation"],
                          "question_type": "practice_recommendation"})

    # Generic factual fallback — always grounded in the evidence verbatim.
    if not qa_pairs:
        question = f"What does the source report about {topic_phrase}?"
        qa_pairs.append({"question": question.strip(), "answer": evidence,
                          "question_type": "factual"})

    return qa_pairs[:max_pairs]
