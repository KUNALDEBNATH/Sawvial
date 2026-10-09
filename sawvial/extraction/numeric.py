"""
sawvial.extraction.numeric
============================
Extracts and preserves numerical scientific information so a sentence
like "Water use decreased by 24%" is never flattened to "Water use
decreased." — the old pipeline threw this precision away entirely by
treating text as an opaque bag of words for embedding only.

Produces structured facts of the shape:

    {
        "raw_text": "decreased by 24%",
        "effect": "decrease",          # increase / decrease / none
        "magnitude": 24.0,
        "unit": "%",
        "parameter_hint": "water use"  # nearby noun phrase, best-effort
    }

as well as flat lists of detected units, percentages, ranges and
statistical values (p-values, confidence intervals) for the
`structured_facts` and `quality` sections of a knowledge unit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

_UNIT_PATTERN = (
    r"(?:%|kg\s*ha-?1|kg/ha|t\s*ha-?1|t/ha|mm|cm|m3|m³|ha|mg\s*kg-?1|mg/kg|"
    r"dS\s*m-?1|dS/m|µS\s*cm-?1|°C|degC|ppm|mmol|cmol|g\s*kg-?1|g/kg|"
    r"l\s*ha-?1|l/ha|litres?|liters?|days?|hours?|years?)"
)

_NUMBER = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?"

_PERCENT_EFFECT_RE = re.compile(
    rf"(?P<param>[A-Za-z][A-Za-z \-]{{2,40}}?)\s+"
    rf"(?P<effect>increased|decreased|reduced|improved|enhanced|declined|rose|fell)\s+"
    rf"(?:by\s+)?(?P<mag>{_NUMBER})\s*(?P<unit>%|percent)",
    re.IGNORECASE,
)

_RANGE_RE = re.compile(
    rf"(?P<lo>{_NUMBER})\s*(?:-|to|–|—)\s*(?P<hi>{_NUMBER})\s*(?P<unit>{_UNIT_PATTERN})?",
    re.IGNORECASE,
)

_MEASUREMENT_RE = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT_PATTERN})\b",
    re.IGNORECASE,
)

_PVALUE_RE = re.compile(
    rf"p\s*(?:[<=>]|less than|greater than)\s*(?P<pval>{_NUMBER})",
    re.IGNORECASE,
)

_CI_RE = re.compile(
    rf"(?:95\s*%\s*CI|confidence interval)[:\s]*\(?\s*(?P<lo>{_NUMBER})\s*(?:-|,|to)\s*(?P<hi>{_NUMBER})\)?",
    re.IGNORECASE,
)


@dataclass
class StructuredFact:
    fact_type: str            # "effect" | "range" | "measurement" | "p_value" | "confidence_interval"
    raw_text: str
    parameter: Optional[str] = None
    effect: Optional[str] = None
    magnitude: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


_EFFECT_NORMALIZATION = {
    "increased": "increase", "improved": "increase", "enhanced": "increase", "rose": "increase",
    "decreased": "decrease", "reduced": "decrease", "declined": "decrease", "fell": "decrease",
}


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def extract_structured_facts(text: str) -> List[StructuredFact]:
    facts: List[StructuredFact] = []
    consumed_spans = []

    def overlaps(span):
        return any(a <= span[0] < b or a < span[1] <= b for a, b in consumed_spans)

    for m in _PERCENT_EFFECT_RE.finditer(text):
        span = m.span()
        consumed_spans.append(span)
        facts.append(StructuredFact(
            fact_type="effect",
            raw_text=m.group(0).strip(),
            parameter=m.group("param").strip(),
            effect=_EFFECT_NORMALIZATION.get(m.group("effect").lower(), m.group("effect").lower()),
            magnitude=_to_float(m.group("mag")),
            unit="%",
        ))

    for m in _PVALUE_RE.finditer(text):
        facts.append(StructuredFact(fact_type="p_value", raw_text=m.group(0).strip(),
                                     magnitude=_to_float(m.group("pval"))))

    for m in _CI_RE.finditer(text):
        facts.append(StructuredFact(fact_type="confidence_interval", raw_text=m.group(0).strip(),
                                     low=_to_float(m.group("lo")), high=_to_float(m.group("hi"))))

    for m in _RANGE_RE.finditer(text):
        span = m.span()
        if overlaps(span):
            continue
        facts.append(StructuredFact(fact_type="range", raw_text=m.group(0).strip(),
                                     low=_to_float(m.group("lo")), high=_to_float(m.group("hi")),
                                     unit=(m.group("unit") or "").strip() or None))

    for m in _MEASUREMENT_RE.finditer(text):
        span = m.span()
        if overlaps(span):
            continue
        facts.append(StructuredFact(fact_type="measurement", raw_text=m.group(0).strip(),
                                     magnitude=_to_float(m.group("value")), unit=m.group("unit")))

    return facts


def has_numeric_signal(text: str) -> bool:
    return bool(_MEASUREMENT_RE.search(text) or _PERCENT_EFFECT_RE.search(text) or _RANGE_RE.search(text))
