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
import logging
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
from sawvial.model_config import ModelConfigError

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

NLLB = {"en": "eng_Latn", "bn": "ben_Beng", "hi": "hin_Deva", "as": "asm_Beng",
        "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva", "ml": "mal_Mlym",
        "kn": "kan_Knda", "gu": "guj_Gujr", "pa": "pan_Guru", "or": "ory_Orya",
        "ur": "urd_Arab", "ne": "npi_Deva"}
NAMES = {"english": "en", "bengali": "bn", "bangla": "bn", "hindi": "hi", "assamese": "as",
         "tamil": "ta", "telugu": "te", "marathi": "mr", "malayalam": "ml", "kannada": "kn",
         "gujarati": "gu", "punjabi": "pa", "odia": "or", "oriya": "or", "urdu": "ur", "nepali": "ne"}
SPEECH_LANGS = Literal["bn", "mr", "en", "hi", "ta", "te", "kn"]


def norm_lang(x: Optional[str]) -> str:
    x = (x or "en").strip().lower()
    x = NAMES.get(x, x.replace("_", "-").split("-")[0])
    return x if x in NLLB else "en"


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
            "snippet": (p.get("text") or "")[:300]}


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
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
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
        raise ModelConfigError(f"Model libraries are missing ({exc}). Install them with: "
                               "pip install -r requirements-llm.txt   (or use --mock for UI testing)") from exc
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
        log.info("Loading translator ...")
        try:
            STATE["translator"] = Translator(CFG["translator_model"])
        except Exception as exc:                       # missing model must not stop English use
            STATE["translator_error"] = (
                f"{exc.__class__.__name__}: translation model {CFG['translator_model']} is unavailable "
                f"(run `python tools/download_models.py`). Non-English questions will be answered "
                f"in English only.")
            log.warning(STATE["translator_error"])
    STATE["loaded"] = True
    log.info("Ready.")


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
    language: Optional[str] = Field("en", max_length=20)
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
            if chunks is not None:
                chunks.close()
            with CANCEL_EVENTS_LOCK:
                CANCEL_EVENTS.pop(request_id, None)

    return StreamingResponse(audio_stream(), media_type="application/octet-stream",
                             headers={"Cache-Control": "no-store",
                                      "X-Audio-Sample-Rate": str(speech_mod.SAMPLE_RATE)})


# ---------------------------------------------------------------- the answer pipeline
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


def run_ask(question: str, lang: str, top_k: int, history=None, cancel_event=None) -> dict:
    t0 = time.time()
    timings: dict = {}

    def finish(res: dict) -> dict:
        timings["total"] = int((time.time() - t0) * 1000)
        res["latency_ms"] = timings["total"]
        res["timings_ms"] = dict(timings)
        log.info("ask lang=%s grounding=%s timings=%s", lang, res.get("grounding", {}).get("status"), timings)
        return res

    if CFG["mock"]:
        ans = "এটি একটি পরীক্ষামূলক উত্তর।" if lang == "bn" else "This is a mock answer for frontend testing."
        return finish({"answer": ans, "language": lang, "answer_language": lang, "grounded": True,
                       "mock": True, "grounding": {"status": "mock"},
                       "sources": [{"id": "SWM_000000", "title": "Mock source", "authors": ["A. Author"],
                                    "year": 2024, "journal": "Mock Journal", "doi": "10.1000/mock",
                                    "doi_url": "https://doi.org/10.1000/mock", "url": None, "page": 1,
                                    "score": 1.0, "cited": True, "snippet": "Mock evidence text."}]})

    rag, tr, bundle = STATE["rag"], STATE.get("translator"), STATE["bundle"]
    prior = _history_turns(history)
    q_en, translated_in, translation_ok = question, False, True
    if lang != "en":
        if tr:
            q_en, translation_ok = safe_translate(tr, question, NLLB[lang], "eng_Latn", timings, "translate_in")
            translated_in = translation_ok
            if translation_ok:
                prior = [{"role": t["role"],
                          "content": safe_translate(tr, t["content"], NLLB[lang], "eng_Latn", timings, "translate_in")[0]}
                         for t in prior]
        else:
            translation_ok = False

    def direct(text_en: str, status: str, **extra) -> dict:
        out, ok = (text_en, True) if lang == "en" else safe_translate(tr, text_en, "eng_Latn", NLLB[lang],
                                                                      timings, "translate_out")
        res = {"answer": out, "language": lang, "answer_language": lang if (lang == "en" or ok) else "en",
               "grounded": False, "sources": [], "grounding": {"status": status, **extra}}
        if lang != "en" and not ok:
            res["note"] = "translation_unavailable"
        return finish(res)

    if GREETING.fullmatch(q_en.strip()):
        return direct("Hello! I can help with questions about crops, soil, irrigation, water use, and "
                      "soil/water conservation in this research dataset.", "not_applicable")

    previous_question = find_anchor_question(prior, bundle)
    follow_up = bool(previous_question) and (is_follow_up(q_en) or is_formatting_request(q_en))
    if not is_agriculture_question(q_en, bundle) and not follow_up:
        return direct(OUT_OF_SCOPE_ANSWER, "not_applicable", reason="out_of_scope")

    retrieval_query = f"{previous_question} {q_en}" if follow_up else q_en
    t1 = time.time()
    found = retrieve_passages(bundle, retrieval_query, top_k, STATE.get("embedder"))
    timings["retrieval"] = int((time.time() - t1) * 1000)
    passages = found["passages"]

    if found["crop_missing"]:
        return direct(NO_CROP_EVIDENCE.format(crops=", ".join(found["crops"])), "insufficient_evidence",
                      reason="no_crop_evidence", crops=found["crops"])
    if not passages or max(float(p["score"]) for p in passages) < settings.MIN_RETRIEVAL_SCORE:
        return direct(NO_DATA_ANSWER, "insufficient_evidence", reason="no_relevant_evidence")

    t2 = time.time()
    with INFER_LOCK:
        raw = rag.answer(STATE["tok"], STATE["model"], q_en, passages, CFG["max_new_tokens"],
                         history=prior if follow_up else [], cancel_event=cancel_event) or ""
    timings["generation"] = int((time.time() - t2) * 1000)
    if cancel_event is not None and cancel_event.is_set():
        return direct("Stopped.", "cancelled")

    t3 = time.time()
    report = grounding.assess(raw, passages)
    text, cited_ids = grounding.compose_supported_answer(report)
    removed = report["n_claims"] - report["n_supported"]
    if text:
        status = "supported" if not removed and not report["invalid_citations"] else "partially_supported"
    else:
        fallback = grounding.extractive_answer(q_en, passages)
        if not fallback:
            timings["grounding"] = int((time.time() - t3) * 1000)
            return direct(NO_DATA_ANSWER, "insufficient_evidence", reason="model_output_not_supported",
                          invalid_citations=report["invalid_citations"])
        text, best = fallback
        cited_ids, status = [best["id"]], "extractive_fallback"
    timings["grounding"] = int((time.time() - t3) * 1000)

    level = grounding.evidence_level(cited_ids, passages)
    notes = [n for n, cond in (("single_source", level == "single_source"),
                               ("partially_supported", status == "partially_supported"),
                               ("extractive_fallback", status == "extractive_fallback")) if cond]
    out_text, ok = (text, True) if lang == "en" else safe_translate(tr, text, "eng_Latn", NLLB[lang],
                                                                    timings, "translate_out")
    res = {"answer": out_text, "language": lang,
           "answer_language": lang if (lang == "en" or ok) else "en",
           "grounded": True,
           "grounding": {"status": status, "claims": report["n_claims"], "supported_claims": report["n_supported"],
                         "removed_unsupported": removed, "invalid_citations": report["invalid_citations"],
                         "evidence_level": level, "notes": notes},
           "sources": [source_entry(p, p["id"] in set(cited_ids)) for p in passages]}
    if translated_in:
        res["question_en"] = q_en
    if lang != "en":
        res["answer_en"] = text
        if not ok:
            res["note"] = "translation_unavailable"
    return finish(res)


# ---------------------------------------------------------------- endpoints
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
            "llm": {"mode": llm.get("mode"), "adapter_status": llm.get("adapter_status"),
                    "message": llm.get("message"), "base_model": llm.get("base_model")},
            "index": {"n_units": meta.get("n_units"), "built_at": meta.get("built_at"),
                      "stale": bundle.get("stale"), "dense": bool(bundle.get("dense_index"))},
            "answerable_statuses": sorted(ANSWERABLE_STATUSES),
            "speech_local": speech_local}


@app.post("/api/ask", dependencies=[Depends(rate_limit)])
def ask(body: AskIn):
    q = (body.question or body.query or body.message or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    if len(q) > settings.MAX_QUESTION_CHARS:
        raise HTTPException(400, f"question too long (max {settings.MAX_QUESTION_CHARS} characters)")
    if not CFG["mock"] and not STATE.get("loaded"):
        raise HTTPException(503, "The assistant is still starting up; please retry in a moment.")
    request_id = body.request_id or f"ask-{time.time_ns()}"
    cancel_event = threading.Event()
    with CANCEL_EVENTS_LOCK:
        CANCEL_EVENTS[request_id] = cancel_event
    try:
        return run_ask(q, norm_lang(body.language), body.top_k or CFG["top_k"],
                       history=body.history, cancel_event=cancel_event)
    except HTTPException:
        raise
    except Exception:
        log.exception("ask failed")
        raise HTTPException(500, "Internal error while answering. Details are in the server log.")
    finally:
        with CANCEL_EVENTS_LOCK:
            CANCEL_EVENTS.pop(request_id, None)


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
