#!/usr/bin/env python3
"""SAWVIAL API: FastAPI server = index retrieval + fine-tuned LLM + grounding checks
(+ optional translation and local speech synthesis).

    python api.py                 # real models (run `python tools/download_models.py` once first)
    python api.py --mock          # canned answers, instant start (frontend testing only)
    python api.py --no_translate  # English-only, skips the translation model
    python api.py --base_only     # explicitly run the base model WITHOUT the LoRA adapter

Open http://localhost:8000 (serves sawvial/*.html).  API docs: http://localhost:8000/docs
"""
import argparse
import collections
import json
import logging
import queue
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm"))
sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sawvial import grounding, settings
from sawvial.eligibility import ANSWERABLE_STATUSES
from sawvial.index.passages import retrieve_passages
from sawvial.language import NLLB, detect_input_language, norm_lang
from sawvial.model_config import ModelConfigError, hf_cached
from sawvial.research.compare import compare_passages

log = logging.getLogger("sawvial.api")

CFG = dict(dataset_dir=str(settings.DATASET_DIR), index_dir=str(settings.INDEX_DIR),
           adapter=str(settings.ADAPTER_DIR), base_model=None, four_bit=False,
           no_adapter=False, require_adapter=False, mock=False, translate=True,
           translator_model=settings.TRANSLATOR_MODEL,
           max_new_tokens=settings.MAX_NEW_TOKENS, top_k=settings.TOP_K)
STATE: dict = {}
# Serialises calls into torch models (LLM, translator, TTS). It is held only for the model
# call itself - never during retrieval, grounding checks or request parsing.
INFER_LOCK = threading.Lock()

NO_ANSWER = "The provided sources do not contain enough information to answer this question."
OUT_OF_SCOPE_ANSWER = (
    "I can answer questions using this agricultural research dataset, including "
    "crops, soil, irrigation, water use, and soil/water conservation. Please ask "
    "about one of those topics.")
NO_DATA_ANSWER = (
    "I couldn't find relevant, reliable evidence for that question in this "
    "dataset. Try asking about crops, soil, irrigation, water use, or "
    "soil/water conservation.")
NO_CROP_EVIDENCE = (
    "The knowledge base does not currently contain enough evidence about {crops} "
    "to answer this reliably. I will not substitute general advice for a crop-specific "
    "recommendation. You can add research on this crop (see the README) or ask about "
    "crops, soil or water topics that the dataset covers.")
# shown only AFTER online research was attempted (or is switched off)
RESEARCH_MESSAGES = {
    "search_failed": ("The local knowledge base has no reliable evidence for this, and the online search could not be "
                      "completed ({error}). I will not guess. Check the internet connection / search settings, or try again later."),
    "no_results": "The local knowledge base has no reliable evidence for this, and the online search found no matching documents.",
    "no_usable_documents": ("The local knowledge base has no reliable evidence for this. I found online documents but could not "
                            "use any of them (inaccessible, too short, not from trusted sources, or not in English)."),
    "no_relevant_passages": ("The local knowledge base has no reliable evidence for this. I read {n} online document(s) but "
                             "none contained passages that answer the question."),
    "disabled": "Online research is switched off and the local knowledge base has no reliable evidence for this question.",
}

AGRICULTURE_TERMS = re.compile(
    r"\b(agricultur\w*|agronom\w*|farm(?:s|ing)?|crop(?:s|ping)?|potato(?:es)?|"
    r"soil|water|irrigat\w*|rainfed|rainfall|drought|watershed|"
    r"fertili[sz]\w*|manure|nutrient\w*|nitrogen|phosphorus|potassium|"
    r"erosion|runoff|groundwater|recharge|tillage|mulch\w*|compost|"
    r"salinity|saline|pH|yield\w*|water[\s-]*use efficiency|WUE|"
    r"evapotranspiration|water productivity|soil moisture|soil health|"
    r"pest\w*|crop disease|plant disease|soil organic carbon|"
    r"rice|paddy|wheat|maize|corn|millet\w*|sorghum|barley|cotton|sugarcane|"
    r"soybean|soyabean|groundnut|peanut|chickpea|pulses?|lentil|mustard|jute|tea|coffee|"
    r"tomato(?:es)?|onion|cabbage|brinjal|chil(?:l)?i|vegetable\w*|fruit\w*|"
    r"banana|mango|orchard|plant(?:s|ing|ation)?|seed(?:s|ling\w*)?|"
    r"sow(?:n|ing)?|harvest\w*|grow(?:ing)?|cultivat\w*|weed\w*|"
    r"herbicide\w*|pesticide\w*|insecticide\w*|fungicide\w*|insect\w*|"
    r"disease\w*|biochar|organic|livestock|field|land|"
    r"drip|sprinkler|rainwater|moisture|climate|season\w*)\b", re.I)
GREETING = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening|"
    r"thanks|thank you|ok|okay|yes|no)[.! ]*$", re.I)
FOLLOW_UP_OPENER = re.compile(
    r"^(?:and\b|what about\b|how about\b|why\b|how so\b|"
    r"explain (?:more|that|this|it)\b|tell me more\b|"
    r"can you explain\b|could you explain\b)", re.I)
FOLLOW_UP_WORDS = re.compile(
    r"\b(it|they|them|this|that|these|those|there|such a|such an|"
    r"one line|single line|short(?:er)?|brief(?:ly)?|concise|"
    r"summarize|summarise|bullet points?)\b", re.I)

SPEECH_LANGS = Literal["bn", "mr", "en", "hi", "ta", "te", "kn"]


def split_chunks(text: str, limit: int = 350) -> List[str]:
    chunks, cur = [], ""
    for s in re.split(r"(?<=[.!?।])\s+", text.strip()):
        if cur and len(cur) + len(s) + 1 > limit:
            chunks.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- source metadata
_DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    m = _DOI.search(doi or "")
    return m.group(0).rstrip(".,;)") if m else None


def safe_http_url(url: Optional[str]) -> Optional[str]:
    """Only absolute http(s) URLs are ever passed to the frontend as links."""
    try:
        u = urlparse((url or "").strip())
    except ValueError:
        return None
    return url.strip() if u.scheme in ("http", "https") and u.netloc else None


def source_entry(p: dict, cited: bool) -> dict:
    """API view of a passage's source. Missing metadata stays None - nothing is invented."""
    doi = normalize_doi(p.get("doi"))
    return {"id": p["id"], "document_id": p.get("document_id"), "title": p.get("title"),
            "authors": p.get("authors") or [], "year": p.get("year"), "journal": p.get("journal"),
            "doi": doi, "doi_url": f"https://doi.org/{doi}" if doi else None,
            "url": safe_http_url(p.get("url")), "page": p.get("page"),
            "section": p.get("section"), "status": p.get("status"),
            "score": round(float(p["score"]), 4), "cited": cited,
            "origin": p.get("origin", "local"),                       # local knowledge base | live web source
            "source_type": p.get("source_type") or "research_paper",  # official_guidance | research_paper | informational_article
            "trust_tier": p.get("trust_tier"), "retrieved_at": p.get("retrieved_at"),
            "published": p.get("published"),
            "snippet": (p.get("text") or "")[:300]}


def sources_for(passages: list, cited_ids: set) -> list:
    """One entry per source DOCUMENT (several passages of one paper/page are merged). The entry is the cited
    passage if any was cited, otherwise the best-ranked one; `ids` lists every passage used and `pages` their pages."""
    groups: dict = {}
    for p in passages:
        groups.setdefault(p.get("document_id") or p["id"], []).append(p)
    out = []
    for group in groups.values():
        cited = [p for p in group if p["id"] in cited_ids]
        entry = source_entry((cited or group)[0], bool(cited))
        entry["ids"] = [p["id"] for p in group]
        entry["pages"] = sorted({p["page"] for p in group if p.get("page")})
        out.append(entry)
    out.sort(key=lambda e: not e["cited"])               # cited sources first (stable otherwise)
    return out


# ---------------------------------------------------------------- question analysis
def is_agriculture_question(question: str, bundle: dict) -> bool:
    if AGRICULTURE_TERMS.search(question):
        return True
    ql = question.lower()
    return any(re.search(rf"\b{re.escape(c)}\b", ql) for c in bundle.get("crop_names", ()))


def is_follow_up(question: str) -> bool:
    """A short message leaning on the previous turn. Pronouns inside a normal, self-contained
    question ("this week", "that field") must NOT make it a follow-up."""
    q = question.strip()
    if FOLLOW_UP_OPENER.search(q):
        return True
    return len(q.split()) <= 7 and bool(FOLLOW_UP_WORDS.search(q))


def is_formatting_request(question: str) -> bool:
    return bool(re.search(
        r"\b(?:make|keep|write|put|format|shorten|summari[sz]e|answer)\b.*"
        r"\b(?:short(?:er)?|brief(?:ly)?|concise|one[- ]line|single[- ]line|"
        r"bullet points?|simpler|plain language)\b|"
        r"^\s*(?:in|as)\s+(?:one|a single)\s+line\s*[.!?]*$", question.strip(), re.I))


def find_anchor_question(history: list, bundle: dict) -> str:
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        content = (turn.get("content") or "").strip()
        if (content and not is_formatting_request(content) and not is_follow_up(content)
                and is_agriculture_question(content, bundle)):
            return content
    return ""


# ---------------------------------------------------------------- translation
class Translator:
    """NLLB-200 (600M). NOTE: licence is CC-BY-NC 4.0 (non-commercial)."""
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.dev = (STATE.get("llm_status") or {}).get("device", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        local = settings.OFFLINE
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=local)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            name, torch_dtype=torch.float16 if self.dev == "cuda" else torch.float32,
            local_files_only=local).to(self.dev).eval()

    def translate(self, text: str, src: str, tgt: str) -> str:
        chunks = split_chunks(text)
        if not chunks:
            return text
        self.tok.src_lang = src
        enc = self.tok(chunks, return_tensors="pt", padding=True, truncation=True,
                       max_length=256).to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(tgt),
                                      max_new_tokens=300, num_beams=3)
        return " ".join(self.tok.batch_decode(out, skip_special_tokens=True))


def safe_translate(tr, text: str, src: str, tgt: str, timings: Optional[dict] = None,
                   key: str = "translate") -> tuple:
    """-> (text, ok). Never fails the request; on error returns the untranslated text, ok=False."""
    if not tr or not text:
        return text, False
    t0 = time.time()
    try:
        with INFER_LOCK:
            out = tr.translate(text, src, tgt)
        return out, True
    except Exception as exc:
        log.warning("Translation %s->%s failed: %s", src, tgt, exc)
        return text, False
    finally:
        if timings is not None:
            timings[key] = timings.get(key, 0) + int((time.time() - t0) * 1000)


# ---------------------------------------------------------------- start-up
def load_all() -> None:
    """Load retriever, LLM and translator once. Raises ModelConfigError with a clear fix."""
    if STATE.get("loaded") or CFG["mock"]:
        return
    try:
        import rag_chat as rag
    except ImportError as exc:
        raise ModelConfigError(f"Model libraries are missing ({exc}). Install them with the setup lines in README.md "
                               "(CUDA PyTorch first, then requirements-llm.txt), or use --mock for UI testing") from exc
    STATE["rag"] = rag
    log.info("Loading retriever ...")
    STATE["bundle"], STATE["embedder"] = rag.load_retriever(CFG["index_dir"], CFG["dataset_dir"])
    from sawvial.index.retriever import warm_up
    warm_up(STATE["bundle"])
    log.info("Loading LLM ...")
    STATE["tok"], STATE["model"], STATE["llm_status"] = rag.load_llm(SimpleNamespace(
        adapter=CFG["adapter"], base_model=CFG["base_model"], no_adapter=CFG["no_adapter"],
        require_adapter=CFG["require_adapter"], four_bit=CFG["four_bit"]))
    STATE["translator"], STATE["translator_error"] = None, None
    if CFG["translate"]:
        STATE["translator_error"] = load_translator()
    STATE["loaded"] = True
    log.info("Ready.")


def load_translator() -> Optional[str]:
    """Load + self-test the NLLB translator. Returns None on success or an actionable error message."""
    name = CFG["translator_model"]
    if settings.OFFLINE and not hf_cached(name):
        msg = (f"Translation model {name} is not in the local cache. Download it once with: "
               f"python tools/download_models.py --only translator   (or set SAWVIAL_OFFLINE=0 to allow download)")
        log.warning(msg)
        return msg
    log.info("Loading translator ...")
    try:
        tr = Translator(name)
        probe = tr.translate("Soil moisture", NLLB["en"], NLLB["ta"])           # en -> ta must really work
        if not probe or not probe.strip():
            raise RuntimeError("self-test translation returned nothing")
        STATE["translator"] = tr
        return None
    except Exception as exc:                          # a broken translator must not stop English use
        msg = (f"Translation model {name} could not be loaded ({exc.__class__.__name__}: {exc}). "
               f"Reinstall with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements-llm.txt (needs sentencepiece, protobuf, transformers<5) "
               f"and re-download with: python tools/download_models.py --only translator")
        log.warning(msg)
        return msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()
    yield


app = FastAPI(title="SAWVIAL API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp


class RateLimiter:
    """Tiny per-client sliding-window limiter. Behind a reverse proxy every client shares the
    proxy's IP - use the proxy's own limiter there."""
    def __init__(self, per_minute: int):
        self.per_minute, self.hits, self.lock = per_minute, collections.defaultdict(collections.deque), threading.Lock()

    def check(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


LIMITER = RateLimiter(settings.RATE_LIMIT_PER_MIN)


def rate_limit(request: Request):
    if not LIMITER.check(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many requests - please wait a moment and try again.")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskIn(BaseModel):
    question: Optional[str] = Field(None, max_length=settings.MAX_QUESTION_CHARS + 200)
    query: Optional[str] = Field(None, max_length=settings.MAX_QUESTION_CHARS + 200)   # aliases
    message: Optional[str] = Field(None, max_length=settings.MAX_QUESTION_CHARS + 200)
    language: Optional[str] = Field("en", max_length=20)          # legacy alias of answer_language (the UI language)
    answer_language: Optional[str] = Field(None, max_length=20)   # language the REPLY should be in
    input_language: Optional[str] = Field(None, max_length=20)    # language the QUESTION is in ("auto"/None = detect)
    research: Literal["auto", "always", "never"] = "auto"         # online research: when local evidence is insufficient / always / never
    top_k: Optional[int] = Field(None, ge=1, le=8)
    history: Optional[List[ChatTurn]] = Field(None, max_length=48)
    request_id: Optional[str] = Field(None, max_length=120)


class CancelIn(BaseModel):
    request_id: str = Field(max_length=120)


class SpeechIn(BaseModel):
    text: str = Field(max_length=2000)
    language: SPEECH_LANGS
    request_id: Optional[str] = Field(None, max_length=120)


CANCEL_EVENTS: dict = {}
CANCEL_EVENTS_LOCK = threading.Lock()


@app.post("/api/cancel")
def cancel(body: CancelIn):
    with CANCEL_EVENTS_LOCK:
        event = CANCEL_EVENTS.get(body.request_id)
    if event:
        event.set()
    return {"status": "cancelling", "request_id": body.request_id}


@app.post("/api/speech/stream", dependencies=[Depends(rate_limit)])
def speech(body: SpeechIn):
    if not body.text.strip():
        raise HTTPException(400, "text is required")
    import speech as speech_mod                          # llm/speech.py (llm/ is on sys.path)
    if not speech_mod.is_cached(body.language) and settings.OFFLINE:
        raise HTTPException(503, f"No local voice model for '{body.language}'. "
                                 "Run `python tools/download_models.py --speech`; the browser voice "
                                 "will be used instead.")
    request_id = body.request_id or f"speech-{time.time_ns()}"
    cancel_event = threading.Event()
    with CANCEL_EVENTS_LOCK:
        if request_id in CANCEL_EVENTS:
            raise HTTPException(409, "request_id is already in use")
        CANCEL_EVENTS[request_id] = cancel_event

    def audio_stream():
        chunks = None
        try:
            chunks = iter(speech_mod.stream_audio(body.text, body.language, cancel_event))
            while not cancel_event.is_set():
                try:
                    with INFER_LOCK:
                        chunk = next(chunks)
                except StopIteration:
                    break
                yield chunk
        except Exception as exc:
            log.error("Speech stream %s failed: %s", request_id, exc)
            raise
        finally:
            if chunks is not None and hasattr(chunks, "close"):
                chunks.close()
            with CANCEL_EVENTS_LOCK:
                CANCEL_EVENTS.pop(request_id, None)

    return StreamingResponse(audio_stream(), media_type="application/octet-stream",
                             headers={"Cache-Control": "no-store",
                                      "X-Audio-Sample-Rate": str(speech_mod.SAMPLE_RATE)})


# ---------------------------------------------------------------- the answer pipeline
def _noop(*_a, **_k):
    return None


def _history_turns(history) -> list:
    turns, total = [], 0
    for turn in reversed(history or []):
        content = turn.content.strip()
        if not content or len(turns) >= 24 or total >= 8000:
            continue
        content = content[:min(700, 8000 - total)]
        turns.append({"role": turn.role, "content": content})
        total += len(content)
    turns.reverse()
    return turns


class Translation:
    """Per-request translation helper that keeps an honest record of what really happened."""
    def __init__(self, answer_lang: str, timings: dict):
        self.tr, self.answer_lang, self.timings = STATE.get("translator"), answer_lang, timings
        self.input = "not_needed"      # not_needed | ok | unavailable | failed
        self.output = "not_needed"
        self.warnings: List[str] = []

    def _run(self, text: str, src: str, tgt: str, key: str) -> tuple:
        if not self.tr:
            return text, "unavailable"
        out, ok = safe_translate(self.tr, text, NLLB[src], NLLB[tgt], self.timings, key)
        return (out, "ok") if ok else (text, "failed")

    def to_english(self, text: str, src: str) -> tuple:
        if src == "en":
            return text, "not_needed"
        return self._run(text, src, "en", "translate_in")

    def from_english(self, text_en: str) -> tuple:
        """-> (text, text_language). text_language is 'en' when the reply could NOT be translated."""
        if self.answer_lang == "en":
            return text_en, "en"
        out, status = self._run(text_en, "en", self.answer_lang, "translate_out")
        self.output = status
        if status != "ok":
            self.warnings.append("translation_unavailable" if status == "unavailable" else "translation_failed")
            return text_en, "en"
        return out, self.answer_lang

    def meta(self) -> dict:
        return {"input": self.input, "output": self.output, "available": bool(self.tr)}


def _research_summary(report: Optional[dict], comparison: Optional[dict] = None) -> dict:
    if not report:
        return {"status": "not_needed", "attempted": False}
    docs = [{k: d.get(k) for k in ("url", "title", "status", "tier", "source_type", "reason")}
            for d in report.get("documents", [])][:12]
    out = {"status": report.get("status"), "attempted": report.get("attempted", True), "error": report.get("error"),
           "candidates": report.get("candidates", 0), "documents": docs,
           "providers": [{k: p.get(k) for k in ("provider", "status", "results", "error")} for p in report.get("providers", [])]}
    if comparison:
        out["comparison"] = comparison
    return out


def run_ask(question: str, lang: str, top_k: int, history=None, cancel_event=None, *,
            input_hint: Optional[str] = None, progress=None, research: str = "auto") -> dict:
    """Pipeline: language handling -> LOCAL retrieval -> relevance check -> ONLINE research if needed ->
    grounded answer -> translation. `lang` is the language the user wants the ANSWER in; the language the
    QUESTION is written in is detected separately (or given via input_hint)."""
    t0 = time.time()
    timings: dict = {}
    progress = progress or _noop
    input_lang = detect_input_language(question, input_hint, selected=lang)
    tl = Translation(lang, timings)
    meta = {"input_language": input_lang, "answer_language": lang, "language": lang}

    def finish(res: dict) -> dict:
        timings["total"] = int((time.time() - t0) * 1000)
        res.update(meta)
        res.setdefault("translation", tl.meta())
        res["warnings"] = list(dict.fromkeys(tl.warnings + res.get("warnings", [])))
        res["latency_ms"] = timings["total"]
        res["timings_ms"] = dict(timings)
        if res["text_language"] != lang and "translation_unavailable" not in res["warnings"] \
                and "translation_failed" not in res["warnings"]:
            res["warnings"].append("translation_unavailable")      # never present English as the requested language
        log.info("ask in=%s ans=%s text=%s grounding=%s research=%s timings=%s", input_lang, lang, res["text_language"],
                 res.get("grounding", {}).get("status"), res.get("research", {}).get("status"), timings)
        return res

    def direct(text_en: str, status: str, rep: Optional[dict] = None, **extra) -> dict:
        """Every early return (greeting, out of scope, no evidence, cancelled, ...) goes through here, so the
        language metadata and the translation are always consistent."""
        progress("translating") if lang != "en" else None
        text, text_lang = tl.from_english(text_en)
        return finish({"answer": text, "text_language": text_lang, "grounded": False, "sources": [],
                       "grounding": {"status": status, **extra}, "research": _research_summary(rep),
                       "evidence_origin": "none"})

    if CFG["mock"]:
        mock_ans = "এটি একটি পরীক্ষামূলক উত্তর।" if lang == "bn" else "This is a mock answer for frontend testing."
        tl.output = "ok" if lang in ("bn", "en") else "unavailable"
        if lang not in ("bn", "en"):
            tl.warnings.append("translation_unavailable")
        return finish({"answer": mock_ans if lang in ("bn", "en") else "This is a mock answer for frontend testing.",
                       "text_language": lang if lang in ("bn", "en") else "en", "grounded": True, "mock": True,
                       "grounding": {"status": "mock"}, "research": {"status": "not_needed", "attempted": False},
                       "evidence_origin": "local",
                       "sources": [{"id": "SWM_000000", "title": "Mock source", "authors": ["A. Author"], "year": 2024,
                                    "journal": "Mock Journal", "doi": "10.1000/mock", "doi_url": "https://doi.org/10.1000/mock",
                                    "url": None, "page": 1, "score": 1.0, "cited": True, "origin": "local",
                                    "source_type": "research_paper", "snippet": "Mock evidence text."}]})

    rag, bundle = STATE["rag"], STATE["bundle"]
    prior = _history_turns(history)

    # ---- 1. language: translate the QUESTION from its own language (not from the answer language)
    q_en = question
    if input_lang != "en":
        progress("translating_in")
        q_en, tl.input = tl.to_english(question, input_lang)
        if tl.input != "ok":
            tl.warnings.append("input_translation_unavailable")
            return direct("I could not translate your question because the translation model is not available. "
                          "Please ask in English, or see /api/health for how to install the translator.",
                          "translation_unavailable")
        prior = [{"role": t["role"],
                  "content": tl.to_english(t["content"], detect_input_language(t["content"], None, lang))[0]}
                 for t in prior]
    else:
        prior = [{"role": t["role"],
                  "content": (tl.to_english(t["content"], d)[0] if (d := detect_input_language(t["content"], None, lang)) != "en" else t["content"])}
                 for t in prior]

    if GREETING.fullmatch(q_en.strip()):
        return direct("Hello! I can help with questions about crops, soil, irrigation, water use, and "
                      "soil/water conservation, using a research knowledge base and online sources.", "not_applicable")
    previous_question = find_anchor_question(prior, bundle)
    follow_up = bool(previous_question) and (is_follow_up(q_en) or is_formatting_request(q_en))
    if not is_agriculture_question(q_en, bundle) and not follow_up:
        return direct(OUT_OF_SCOPE_ANSWER, "not_applicable", reason="out_of_scope")
    retrieval_query = f"{previous_question} {q_en}" if follow_up else q_en

    # ---- 2. LOCAL knowledge base first
    progress("local_search")
    t1 = time.time()
    local = retrieve_passages(bundle, retrieval_query, top_k, STATE.get("embedder"))
    timings["retrieval"] = int((time.time() - t1) * 1000)
    local_ok = (not local["crop_missing"] and bool(local["passages"])
                and max(float(p["score"]) for p in local["passages"]) >= settings.MIN_RETRIEVAL_SCORE)
    passages, found, research_report = (local["passages"] if local_ok else []), local, None
    if cancel_event is not None and cancel_event.is_set():
        return direct("Stopped.", "cancelled")

    # ---- 3. ONLINE research when the local evidence is not enough (or when the user asks for it)
    want_online = research == "always" or (research == "auto" and not local_ok)
    if want_online and research != "never":
        from sawvial.research.live import live_research
        t_r = time.time()
        live = live_research(retrieval_query, top_k, progress=progress, cancel_event=cancel_event)
        timings["research"] = int((time.time() - t_r) * 1000)
        research_report = live["report"]
        if live["passages"]:
            passages = (passages[:max(1, top_k // 2)] if research == "always" else []) + live["passages"]
            passages = passages[:top_k]
            found = live
        if cancel_event is not None and cancel_event.is_set():
            return direct("Stopped.", "cancelled", research_report)

    if not passages:
        st = (research_report or {}).get("status") if (research_report or {}).get("attempted") else None
        if st in RESEARCH_MESSAGES:
            msg = RESEARCH_MESSAGES[st].format(error=(research_report or {}).get("error", ""), n=len((research_report or {}).get("documents", [])))
            return direct(msg, "insufficient_evidence", research_report, reason=f"research_{st}")
        if local["crop_missing"]:
            return direct(NO_CROP_EVIDENCE.format(crops=", ".join(local["crops"])), "insufficient_evidence",
                          research_report, reason="no_crop_evidence", crops=local["crops"])
        return direct(NO_DATA_ANSWER, "insufficient_evidence", research_report, reason="no_relevant_evidence")

    # ---- 4. generate + verify
    progress("generating")
    t2 = time.time()
    try:
        with INFER_LOCK:
            raw = rag.answer(STATE["tok"], STATE["model"], q_en, passages, CFG["max_new_tokens"],
                             history=prior if follow_up else [], cancel_event=cancel_event) or ""
    except Exception as exc:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            log.error("CUDA out of memory while generating an answer (even after the in-generation retry)")
            torch.cuda.empty_cache()
            return direct("The server ran out of GPU memory while generating this answer. Please try a "
                          "shorter question, or try again in a moment.", "gpu_out_of_memory", research_report)
        raise
    timings["generation"] = int((time.time() - t2) * 1000)
    if cancel_event is not None and cancel_event.is_set():
        return direct("Stopped.", "cancelled", research_report)

    t3 = time.time()
    report = grounding.assess(raw, passages)
    text, cited_ids = grounding.compose_supported_answer(report)
    removed = report["n_claims"] - report["n_supported"]
    regenerated = False
    if text:
        status = "supported" if not removed and not report["invalid_citations"] else "partially_supported"
    else:
        # Grounding validation failed outright (no sentence of the model's answer was supported).
        # Rather than immediately quoting raw source text as if it were a generated answer, give
        # the model ONE controlled retry with concrete feedback about what was wrong, and validate
        # that retry the same way. Only if the retry ALSO fails do we fall back to a verbatim
        # excerpt, clearly labelled as such (never invented, never presented as the model's answer).
        unsupported_reasons = sorted({s["reason"] for s in report["sentences"] if not s["supported"] and not s["refusal"]})
        feedback_bits = []
        if report["invalid_citations"]:
            feedback_bits.append(f"these citation ids do not exist in the supplied records: {', '.join(report['invalid_citations'])}")
        if unsupported_reasons:
            feedback_bits.append("claims were rejected for: " + "; ".join(unsupported_reasons))
        feedback = ("No claim in your answer could be verified against the supplied records "
                    f"({'; '.join(feedback_bits) if feedback_bits else 'no valid citation was found'})."
                    ) if (feedback_bits or report["n_claims"]) else None
        try:
            with INFER_LOCK:
                retry_raw = rag.answer(STATE["tok"], STATE["model"], q_en, passages, CFG["max_new_tokens"],
                                       history=prior if follow_up else [], cancel_event=cancel_event,
                                       feedback=feedback) or ""
        except Exception as exc:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                retry_raw = ""
            else:
                raise
        if cancel_event is not None and cancel_event.is_set():
            timings["grounding"] = int((time.time() - t3) * 1000)
            return direct("Stopped.", "cancelled", research_report)
        if retry_raw:
            retry_report = grounding.assess(retry_raw, passages)
            retry_text, retry_ids = grounding.compose_supported_answer(retry_report)
            if retry_text:
                report, text, cited_ids, regenerated = retry_report, retry_text, retry_ids, True
                removed = report["n_claims"] - report["n_supported"]
                status = "supported" if not removed and not report["invalid_citations"] else "partially_supported"

        if not text:
            fb = grounding.extractive_answer(q_en, passages)
            if fb:
                text, best = fb
                cited_ids, status = [best["id"]], "extractive_fallback"
            else:
                ex = grounding.excerpt_answer(passages)          # last resort: a verbatim, relevant excerpt (never invented)
                if ex is None:
                    timings["grounding"] = int((time.time() - t3) * 1000)
                    return direct(NO_DATA_ANSWER, "insufficient_evidence", research_report,
                                  reason="model_output_not_supported", invalid_citations=report["invalid_citations"])
                text, best = ex
                cited_ids, status = [best["id"]], "excerpt_only"
    timings["grounding"] = int((time.time() - t3) * 1000)

    # ---- 5. compare sources, state uncertainty
    cited_set = set(cited_ids)
    comparison = compare_passages(passages)         # ALL evidence shown, so disagreements are not hidden by what the model cited
    level = grounding.evidence_level(cited_ids, passages)
    origins = {p.get("origin", "local") for p in passages if p["id"] in cited_set}
    notes = [n for n, cond in (("single_source", level == "single_source"),
                               ("partially_supported", status == "partially_supported"),
                               ("extractive_fallback", status in ("extractive_fallback", "excerpt_only")),
                               ("web_unreviewed", "web" in origins),
                               ("regenerated_after_grounding_failure", regenerated),
                               ("differing_values", bool(comparison["differences"]))) if cond]
    progress("translating") if lang != "en" else None
    out_text, text_lang = tl.from_english(text)
    res = {"answer": out_text, "text_language": text_lang, "grounded": True,
           "grounding": {"status": status, "claims": report["n_claims"], "supported_claims": report["n_supported"],
                         "removed_unsupported": removed, "invalid_citations": report["invalid_citations"],
                         "evidence_level": level, "notes": notes},
           "evidence_origin": "web" if origins == {"web"} else ("local+web" if "web" in origins else "local"),
           "research": _research_summary(research_report, comparison),
           "sources": sources_for(passages, cited_set)}
    if input_lang != "en":
        res["question_en"] = q_en
    if lang != "en":
        res["answer_en"] = text
    return finish(res)


# ---------------------------------------------------------------- endpoints
def _device_report(llm: dict) -> dict:
    dev = dict(llm.get("device") or {})
    torch = sys.modules.get("torch")                  # only report live VRAM if torch is already loaded
    if torch is not None and dev.get("device") == "cuda":
        try:
            dev["vram_used_mb"] = int(torch.cuda.memory_allocated() // (1024 * 1024))
        except Exception:                             # noqa: BLE001
            pass
    return dev


@app.get("/api/health")
def health():
    llm = STATE.get("llm_status") or {}
    bundle = STATE.get("bundle") or {}
    meta = bundle.get("index_meta") or {}
    try:
        import speech as speech_mod
        speech_local = speech_mod.available_languages()
    except Exception:
        speech_local = []
    return {"status": "ok", "mock": CFG["mock"],
            "translator": bool(STATE.get("translator")),
            "translator_error": STATE.get("translator_error"),
            "translator_model": CFG["translator_model"], "translator_model_cached": hf_cached(CFG["translator_model"]),
            "llm": {"mode": llm.get("mode"), "adapter_status": llm.get("adapter_status"),
                    "message": llm.get("message"), "base_model": llm.get("base_model")},
            "index": {"n_units": meta.get("n_units"), "built_at": meta.get("built_at"),
                      "stale": bundle.get("stale"), "dense": bool(bundle.get("dense_index"))},
            "device": _device_report(llm),
            "research": {"enabled": settings.ONLINE_RESEARCH, "search_provider": settings.SEARCH_PROVIDER,
                         "min_trust": settings.RESEARCH_MIN_TRUST, "max_docs": settings.RESEARCH_MAX_DOCS},
            "answerable_statuses": sorted(ANSWERABLE_STATUSES),
            "speech_local": speech_local}


def _validated_question(body: "AskIn") -> str:
    q = (body.question or body.query or body.message or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    if len(q) > settings.MAX_QUESTION_CHARS:
        raise HTTPException(400, f"question too long (max {settings.MAX_QUESTION_CHARS} characters)")
    if not CFG["mock"] and not STATE.get("loaded"):
        raise HTTPException(503, "The assistant is still starting up; please retry in a moment.")
    return q


def _answer_lang(body: "AskIn") -> str:
    """The reply language: `answer_language` if given, else the legacy `language` field (the UI language)."""
    return norm_lang(body.answer_language or body.language)


@app.post("/api/ask", dependencies=[Depends(rate_limit)])
def ask(body: AskIn):
    q = _validated_question(body)
    request_id = body.request_id or f"ask-{time.time_ns()}"
    cancel_event = threading.Event()
    with CANCEL_EVENTS_LOCK:
        CANCEL_EVENTS[request_id] = cancel_event
    try:
        return run_ask(q, _answer_lang(body), body.top_k or CFG["top_k"], history=body.history,
                       cancel_event=cancel_event, input_hint=body.input_language, research=body.research)
    except HTTPException:
        raise
    except Exception:
        log.exception("ask failed")
        raise HTTPException(500, "Internal error while answering. Details are in the server log.")
    finally:
        with CANCEL_EVENTS_LOCK:
            CANCEL_EVENTS.pop(request_id, None)


@app.post("/api/ask/stream", dependencies=[Depends(rate_limit)])
def ask_stream(body: AskIn):
    """Same pipeline as /api/ask, streamed as NDJSON: {"stage": ...} lines while it works (local_search,
    online_search, reading, generating, translating...), then one {"final": {...}} (or {"error": {...}}) line."""
    q = _validated_question(body)
    request_id = body.request_id or f"ask-{time.time_ns()}"
    cancel_event = threading.Event()
    with CANCEL_EVENTS_LOCK:
        CANCEL_EVENTS[request_id] = cancel_event
    events: "queue.Queue" = queue.Queue()
    lang, top_k = _answer_lang(body), body.top_k or CFG["top_k"]

    def worker():
        try:
            res = run_ask(q, lang, top_k, history=body.history, cancel_event=cancel_event,
                          input_hint=body.input_language, research=body.research,
                          progress=lambda stage, **info: events.put({"stage": stage, **info}))
            events.put({"final": res})
        except Exception:
            log.exception("ask failed")
            events.put({"error": {"status": 500, "detail": "Internal error while answering. Details are in the server log."}})
        finally:
            events.put(None)

    def stream():
        threading.Thread(target=worker, daemon=True).start()
        done = False
        try:
            while True:
                ev = events.get()
                if ev is None:
                    done = True
                    break
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        finally:
            if not done:
                cancel_event.set()                      # client went away: stop research/generation
            with CANCEL_EVENTS_LOCK:
                CANCEL_EVENTS.pop(request_id, None)

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


# ---- serve the frontend (only html pages + css/js/assets, NOT the python package) ----
WEB = ROOT / "sawvial"
for d in ("css", "js", "assets"):
    if (WEB / d).is_dir():
        app.mount(f"/{d}", StaticFiles(directory=WEB / d), name=d)


@app.get("/")
def root():
    return RedirectResponse("/index.html" if (WEB / "index.html").exists() else "/ask.html")


@app.get("/{page}.html")
def page(page: str):
    f = WEB / f"{page}.html"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", page) or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only behind a firewall/proxy")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no_translate", action="store_true")
    ap.add_argument("--base_only", "--no_adapter", dest="no_adapter", action="store_true",
                    help="explicitly run the base model WITHOUT the LoRA adapter")
    ap.add_argument("--use_adapter", action="store_true", help="(deprecated: the adapter is on by default)")
    ap.add_argument("--require_adapter", action="store_true", help="fail if the adapter is missing")
    ap.add_argument("--adapter", default=CFG["adapter"])
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--dataset_dir", default=CFG["dataset_dir"])
    ap.add_argument("--index_dir", default=CFG["index_dir"])
    a = ap.parse_args()
    CFG.update(mock=a.mock, translate=not a.no_translate, no_adapter=a.no_adapter,
               require_adapter=a.require_adapter, adapter=a.adapter, base_model=a.base_model,
               four_bit=a.four_bit, dataset_dir=a.dataset_dir, index_dir=a.index_dir)
    try:
        load_all()                                   # fail here with a clean message, not a traceback
    except (ModelConfigError, FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"\nSAWVIAL could not start:\n  {exc}\n")
    if a.host not in ("127.0.0.1", "localhost"):
        log.warning("Listening on %s: set SAWVIAL_CORS_ORIGINS and put this behind HTTPS and a firewall.", a.host)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
