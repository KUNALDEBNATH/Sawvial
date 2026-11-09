"""
Input-language vs answer-language handling (torch-free, no extra dependencies).

* ``answer_language``  - the language the user WANTS the reply in (normally the UI language).
* ``input_language``   - the language the question is actually written in.

They are independent. The input language is taken from an explicit hint when the client gives one,
otherwise detected from the Unicode scripts in the text. Detection is deliberately simple:
  - Tamil / Telugu / Kannada / Bengali / Gujarati / Punjabi / Malayalam / Odia scripts are unambiguous;
  - Devanagari is Hindi or Marathi (Marathi marker words, or the user's own selection, decide);
  - Latin text is English.
Limitation: romanised text ("thakkali mann") is treated as English.
"""
from __future__ import annotations

import re
from typing import Optional

NLLB = {"en": "eng_Latn", "bn": "ben_Beng", "hi": "hin_Deva", "as": "asm_Beng",
        "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva", "ml": "mal_Mlym",
        "kn": "kan_Knda", "gu": "guj_Gujr", "pa": "pan_Guru", "or": "ory_Orya",
        "ur": "urd_Arab", "ne": "npi_Deva"}
NAMES = {"english": "en", "bengali": "bn", "bangla": "bn", "hindi": "hi", "assamese": "as",
         "tamil": "ta", "telugu": "te", "marathi": "mr", "malayalam": "ml", "kannada": "kn",
         "gujarati": "gu", "punjabi": "pa", "odia": "or", "oriya": "or", "urdu": "ur", "nepali": "ne"}

# (first, last) code points of each script
_SCRIPTS = {"deva": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "pa": (0x0A00, 0x0A7F), "gu": (0x0A80, 0x0AFF),
            "or": (0x0B00, 0x0B7F), "ta": (0x0B80, 0x0BFF), "te": (0x0C00, 0x0C7F), "kn": (0x0C80, 0x0CFF),
            "ml": (0x0D00, 0x0D7F), "ur": (0x0600, 0x06FF)}
_MARATHI_WORDS = re.compile(r"(?:आहे|आहेत|काय|कसे|कोणती|कोणता|मला|तुम्ही|करावा|करावी|पाणी\s+द्यावे|ळ)")


def norm_lang(x: Optional[str], default: str = "en") -> str:
    """Normalise 'ta', 'ta-IN', 'Tamil', 'tam_Taml' ... to a supported code (else `default`)."""
    x = (x or default).strip().lower()
    x = NAMES.get(x, x.replace("_", "-").split("-")[0])
    return x if x in NLLB else default


def script_counts(text: str) -> dict:
    counts = {"latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x250:
            counts["latin"] += 1
            continue
        for name, (lo, hi) in _SCRIPTS.items():
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return counts


def detect_input_language(text: str, hint: Optional[str] = None, selected: Optional[str] = None) -> str:
    """Language the question is written in. `hint` (explicit) wins; "auto"/None -> detect from script."""
    if hint and hint.strip().lower() not in ("auto", ""):
        return norm_lang(hint)
    counts = script_counts(text)
    total = sum(counts.values())
    if total == 0:
        return "en"
    script, n = max(((k, v) for k, v in counts.items() if k != "latin"), key=lambda kv: kv[1], default=(None, 0))
    if script is None or n / total < 0.3:                  # mostly Latin letters (maybe a few foreign words)
        return "en"
    if script == "deva":
        if _MARATHI_WORDS.search(text):           # distinctive Marathi words beat the UI selection
            return "mr"
        return selected if selected in ("hi", "mr") else "hi"
    if script == "bn":
        return "as" if re.search("[\u09F0\u09F1]", text) else "bn"
    return script


def text_in_language(text: str, lang: str) -> bool:
    """Does `text` plausibly contain `lang`? (Tamil text for 'ta', mostly Latin text for 'en'.)"""
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return True
    if lang == "en":
        return counts["latin"] / total >= 0.5
    key = {"hi": "deva", "mr": "deva", "ne": "deva", "as": "bn"}.get(lang, lang)
    return counts.get(key, 0) / total >= 0.3
