"""Shared helpers for SAWVIAL LLM fine-tuning + RAG."""
from __future__ import annotations
import json
import re

SYSTEM_PROMPT = (
    "You are SAWVIAL, an agricultural research assistant for soil management, "
    "water management, and soil & water conservation. Read the retrieved study "
    "records and the conversation, identify which evidence answers the user's "
    "current question, and explain the result in your own clear, natural wording. "
    "Answer the specific question; do not repeat a generic answer or copy a "
    "previous answer unless it is still relevant. A shared word is not enough to "
    "make a passage relevant: ignore evidence that does not discuss the crop, "
    "practice, or condition asked about. Use only claims directly supported by "
    "relevant records. If the user asks for a variety, method, or other detail "
    "that the records do not name, state plainly that it is not specified; never "
    "fill the gap with general advice. Cite each factual claim with the relevant "
    "passage ID in square brackets, e.g. [SWM_000123]. Preserve numbers, units, "
    "and experimental conditions exactly as reported. Distinguish a study finding "
    "from a source-stated recommendation. Every factual paragraph or bullet must "
    "include at least one citation from the supplied records; if you cannot cite "
    "a supported answer, say the records do not provide enough information. "
    "Do not turn a within-crop treatment "
    "comparison into a comparison between crops or studies; make a comparison "
    "only when the same record explicitly reports comparable results. If sources "
    "disagree, describe the difference. Treat follow-up messages as instructions "
    "about the active conversation, not standalone questions. Resolve words like "
    "'it' using the most recent relevant user request. If the user asks to shorten, "
    "reformat, or rewrite the previous answer, preserve its supported meaning and "
    "citations while applying that exact style; a one-line request means one line. "
    "When one record mentions multiple crops, never apply another crop's irrigation "
    "method, timing, or amount to the crop the user asked about. When the supplied "
    "evidence is already narrowed to a crop-labeled result, state only that result "
    "and clearly name any requested details it does not contain. "
    "Write a concise direct answer in fresh wording: 2 to 4 sentences, no more. "
    "Start with the answer itself. Never begin a sentence with a bare verb such as "
    "'found' or 'noted'; say 'the study found' or 'one study reported'. Do not mention "
    "author names. Numbers without a stated unit in the records must not be quoted. "
    "Do not list what the records fail to say unless the question asks for it."
)
NO_ANSWER = "The provided sources do not contain enough information to answer this question."

BAD_SECTIONS = {"references", "reference", "bibliography", "literature cited",
                "acknowledgements", "acknowledgments", "appendix"}

# ---- reference-list / citation detection (fixes the noise seen in retrieval) ----
_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_INITIALS = re.compile(r"\b[A-Z]\.(?:\s?[A-Z]\.)*")
_PAGES_END = re.compile(r"\d{1,4}\s*[,:]\s*\d{1,5}\s*[-–]\s*\d{1,5}\.?\s*$")
_VOL_PAGES = re.compile(r"\b\d{1,3}\s*\(\d{1,3}\)\s*[,:]\s*\d{1,5}\s*[-–]\s*\d{1,5}")
_DOI = re.compile(r"doi\.org|doi:\s*10\.", re.I)


def looks_like_reference(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    has_year = bool(_YEAR.search(t))
    initials = len(_INITIALS.findall(t))
    pages = bool(_PAGES_END.search(t) or _VOL_PAGES.search(t))
    if has_year and pages:
        return True
    if has_year and initials >= 4:
        return True
    if _DOI.search(t) and initials >= 2:
        return True
    return False


def is_usable_evidence(evidence: str, title: str | None = None) -> bool:
    ev = (evidence or "").strip()
    if len(ev) < 80 or len(ev.split()) < 12:
        return False
    if (re.match(r"^(?:table|figure)\s+\d+\s*[.:]", ev, re.I)
            and len(ev.split()) < 24):
        return False
    if re.match(r"^(?:keywords?|index terms?)\b", ev, re.I) and len(ev.split()) < 35:
        return False
    if looks_like_reference(ev):
        return False
    if title:
        t = title.strip()
        if ev == t or (t in ev and len(ev) < len(t) + 20):
            return False
    return True


def usable_unit(ku: dict, allow_review: bool = False) -> bool:
    ok = {"validated"} | ({"needs_review"} if allow_review else set())
    if ku.get("quality", {}).get("validation_status") not in ok:
        return False
    src = ku.get("source", {})
    if (src.get("section") or "").lower() in BAD_SECTIONS:
        return False
    return is_usable_evidence(ku.get("knowledge", {}).get("evidence"), src.get("title"))


def fix_grammar(s: str) -> str:
    # the pipeline's templated answers produce "a increase" etc.
    return re.sub(r"\ba (?=[aeiouAEIOU])", "an ", s)


def format_context(passages: list[dict], max_chars: int = 1200) -> str:
    """Format retrieved evidence with its extracted findings and study context."""
    blocks = []
    for p in passages:
        head = f"[{p['id']}] {p.get('title') or 'Unknown source'}"
        loc = ", ".join(str(x) for x in (f"p.{p['page']}" if p.get("page") else None,
                                         p.get("section")) if x and x != "unknown")
        if loc:
            head += f" ({loc})"
        details = [head]
        if p.get("finding"):
            details.append(f"Extracted finding: {p['finding']}")
        if p.get("recommendation"):
            details.append(f"Source-stated recommendation: {p['recommendation']}")
        context = p.get("context") or {}
        context_items = [
            f"{key.replace('_', ' ')}: {', '.join(map(str, values))}"
            for key, values in context.items()
            if isinstance(values, list) and values
        ]
        if context_items:
            details.append("Study context: " + "; ".join(context_items))
        facts = p.get("structured_facts") or []
        fact_text = [str(fact.get("raw_text") or "") for fact in facts
                     if isinstance(fact, dict) and fact.get("raw_text")]
        if fact_text:
            details.append("Structured facts: " + "; ".join(fact_text))
        details.append(f"Source evidence: {p.get('text', '')[:max_chars]}")
        blocks.append("\n".join(details))
    return "\n\n".join(blocks)


def make_user_prompt(context: str, question: str, history: list[dict] | None = None) -> str:
    conversation = ""
    if history:
        turns = []
        for turn in history[-24:]:
            role = "Assistant" if turn.get("role") == "assistant" else "User"
            content = (turn.get("content") or "").strip()[:700]
            if content:
                turns.append(f"{role}: {content[:500]}")
        if turns:
            conversation = "Recent conversation:\n" + "\n".join(turns) + "\n\n"
    return (
        f"{conversation}Retrieved study records (use only records relevant to "
        f"the current question):\n{context}\n\n"
        f"Current question: {question}\n"
        "Answer this question directly. If the records do not provide the requested "
        "detail, say that it is not specified in these records."
    )


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
