#!/usr/bin/env python3
"""SAWVIAL API: FastAPI server = FAISS retrieval + fine-tuned LLM (+ optional translation).

Put this file in the project root (next to build_dataset.py), then:

    python api.py                  # real model
    python api.py --mock           # canned answers, instant start (for frontend testing)
    python api.py --no_translate   # English-only, skips the translation model

Open http://localhost:8000  (serves sawvial/ask.html etc.)   Docs: http://localhost:8000/docs
"""
import argparse, re, sys, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm"))
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CFG = dict(dataset_dir=str(ROOT / "dataset"), index_dir=str(ROOT / "index"),
           adapter=str(ROOT / "llm" / "sawvial-lora"), base_model=None, four_bit=False,
           no_adapter=True, mock=False, translate=True,
           translator_model="facebook/nllb-200-distilled-600M",
           max_new_tokens=300, top_k=4)
STATE: dict = {}
LOCK = threading.Lock()          # one generation at a time on the GPU
NO_ANSWER = "The provided sources do not contain enough information to answer this question."
OUT_OF_SCOPE_ANSWER = (
    "I can answer questions using this agricultural research dataset, including "
    "crops, soil, irrigation, water use, and soil/water conservation. Please ask "
    "about one of those topics."
)
NO_DATA_ANSWER = (
    "I couldn't find relevant, reliable evidence for that question in this "
    "dataset. Try asking about crops, soil, irrigation, water use, or "
    "soil/water conservation."
)
# TF-IDF (offline) cosine scores for good matches are typically 0.05-0.25, so the old
# 0.11 cut-off rejected most normal farmer questions. get_passages() already requires
# real term overlap, so this only needs to drop near-zero matches.
MIN_RETRIEVAL_SCORE = 0.03
# if the LLM says "not enough information" but retrieval was this strong, show the
# best passage instead of a dead-end message.
STRONG_RETRIEVAL_SCORE = 0.08
AGRICULTURE_TERMS = re.compile(
    r"\b(agricultur\w*|agronom\w*|farm(?:s|ing)?|crop(?:s|ping)?|potato(?:es)?|"
    r"soil|water|irrigat\w*|rainfed|rainfall|drought|watershed|"
    r"fertili[sz]\w*|manure|nutrient\w*|nitrogen|phosphorus|potassium|"
    r"erosion|runoff|groundwater|recharge|tillage|mulch\w*|compost|"
    r"salinity|saline|pH|yield\w*|water[\s-]*use efficiency|WUE|"
    r"evapotranspiration|water productivity|soil moisture|soil health|"
    r"pest\w*|crop disease|plant disease|soil organic carbon|"
    r"rice|paddy|wheat|maize|corn|millet\w*|sorghum|barley|cotton|sugarcane|"
    r"soybean|soyabean|groundnut|chickpea|pulses?|lentil|mustard|jute|tea|coffee|"
    r"tomato(?:es)?|onion|cabbage|brinjal|chil(?:l)?i|vegetable\w*|fruit\w*|"
    r"banana|mango|orchard|paddy|plant(?:s|ing|ation)?|seed(?:s|ling\w*)?|"
    r"sow(?:n|ing)?|harvest\w*|grow(?:ing)?|cultivat\w*|weed\w*|"
    r"herbicide\w*|pesticide\w*|insecticide\w*|fungicide\w*|insect\w*|"
    r"disease\w*|fertili[sz]er\w*|biochar|organic|livestock|field|land|"
    r"drip|sprinkler|mulch|rainwater|moisture|climate|season\w*)\b",
    re.I,
)
GREETING = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening|"
    r"thanks|thank you|ok|okay|yes|no)[.! ]*$",
    re.I,
)
FOLLOW_UP_OPENER = re.compile(
    r"^(?:and\b|what about\b|how about\b|why\b|how so\b|"
    r"explain (?:more|that|this|it)\b|tell me more\b|"
    r"can you explain\b|could you explain\b)",
    re.I,
)
FOLLOW_UP_WORDS = re.compile(
    r"\b(it|they|them|this|that|these|those|there|such a|such an|"
    r"one line|single line|short(?:er)?|brief(?:ly)?|concise|"
    r"summarize|summarise|bullet points?)\b",
    re.I,
)
CITE = re.compile(r"\s*\[\s*([A-Z]+_\d+(?:\s*[,;]\s*[A-Z]+_\d+)*)\s*\]")
CITE_ID = re.compile(r"[A-Z]+_\d+")


def find_citations(text: str) -> set:
    return {i for group in CITE.findall(text) for i in CITE_ID.findall(group)}


def strip_citations(text: str, valid_ids: Optional[set] = None) -> str:
    """Remove [ID] markers from the visible answer (valid or not)."""
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", CITE.sub("", text)).strip()


_FALLBACK_STOP = {
    "the", "and", "for", "with", "that", "this", "what", "when", "where", "which",
    "how", "does", "did", "can", "should", "would", "could", "about", "into", "from",
    "have", "has", "are", "was", "were", "you", "your", "my", "our", "its", "any",
    "much", "many", "best", "good", "need", "use", "tell", "give", "show", "make",
}


def _terms(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z]{3,}", text.lower()):
        if w in _FALLBACK_STOP:
            continue
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and len(w) > 4:
            w = w[:-1]
        out.add(w)
    return out


def extractive_answer(question: str, passages: list):
    """Grounded fallback when the LLM output is unusable.

    Picks the evidence sentences that overlap most with the question and quotes them
    with the passage id. Returns None when nothing in the passages really matches.
    """
    q_terms = _terms(question)
    best = None  # (score, passage, sentences)
    for rank, p in enumerate(passages):
        text = re.sub(r"\s+", " ", (p.get("text") or "")).strip()
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.strip()) > 25]
        for i, sent in enumerate(sents):
            score = len(q_terms & _terms(sent)) - rank * 0.05
            if best is None or score > best[0]:
                best = (score, p, sents[i:i + 2])
    if not best or best[0] < 1:
        return None
    _, passage, sents = best
    out = " ".join(sents)
    if len(out) > 480:
        out = sents[0][:480].rsplit(" ", 1)[0]
    if out and out[-1] not in ".!?":
        out += "..."
    return f"According to the research records: {out} [{passage['id']}]", passage


NLLB = {"en": "eng_Latn", "bn": "ben_Beng", "hi": "hin_Deva", "as": "asm_Beng",
        "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva", "ml": "mal_Mlym",
        "kn": "kan_Knda", "gu": "guj_Gujr", "pa": "pan_Guru", "or": "ory_Orya",
        "ur": "urd_Arab", "ne": "npi_Deva"}
NAMES = {"english": "en", "bengali": "bn", "bangla": "bn", "hindi": "hi", "assamese": "as",
         "tamil": "ta", "telugu": "te", "marathi": "mr", "malayalam": "ml", "kannada": "kn",
         "gujarati": "gu", "punjabi": "pa", "odia": "or", "oriya": "or", "urdu": "ur", "nepali": "ne"}


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


def answer_matches_asked_crop(question: str, answer: str, passages: list) -> bool:
    """Reject generic answers that ignore a crop explicitly asked about."""
    question_lower = question.lower()
    answer_lower = answer.lower()

    def crop_is_mentioned(text: str, crop: str) -> bool:
        if crop.endswith("y"):
            plural = crop[:-1] + "ies"
        elif crop.endswith(("o", "ch", "sh", "x", "z", "s")):
            plural = crop + "es"
        else:
            plural = crop + "s"
        return bool(re.search(
            rf"\b(?:{re.escape(crop)}|{re.escape(plural)})\b", text,
        ))

    asked_crops = set()
    for passage in passages:
        for crop in passage.get("context", {}).get("crop", []):
            crop = str(crop).strip().lower()
            if not crop:
                continue
            if crop_is_mentioned(question_lower, crop):
                asked_crops.add(crop)
    return all(crop_is_mentioned(answer_lower, crop) for crop in asked_crops)


def is_agriculture_question(question: str, bundle: dict) -> bool:
    if AGRICULTURE_TERMS.search(question):
        return True
    question_lower = question.lower()
    crops = {
        str(crop).strip().lower()
        for unit in bundle["units_by_id"].values()
        for crop in unit.get("context", {}).get("crop", [])
        if isinstance(crop, str) and crop.strip()
    }
    return any(re.search(rf"\b{re.escape(crop)}\b", question_lower) for crop in crops)


def is_follow_up(question: str) -> bool:
    """A follow-up is a short message that leans on the previous turn.

    Pronoun-like words ("this week", "that field") inside a normal, self-contained
    question must NOT make it a follow-up, otherwise the previous topic gets mixed
    into the retrieval query and the answer comes back unrelated or empty.
    """
    q = question.strip()
    if FOLLOW_UP_OPENER.search(q):
        return True
    return len(q.split()) <= 7 and bool(FOLLOW_UP_WORDS.search(q))


def is_formatting_request(question: str) -> bool:
    return bool(re.search(
        r"\b(?:make|keep|write|put|format|shorten|summari[sz]e|answer)\b.*"
        r"\b(?:short(?:er)?|brief(?:ly)?|concise|one[- ]line|single[- ]line|"
        r"bullet points?|simpler|plain language)\b|"
        r"^\s*(?:in|as)\s+(?:one|a single)\s+line\s*[.!?]*$",
        question.strip(),
        re.I,
    ))


def find_anchor_question(history: list[dict], bundle: dict) -> str:
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        content = (turn.get("content") or "").strip()
        if content and not is_formatting_request(content) and not is_follow_up(content):
            if is_agriculture_question(content, bundle):
                return content
    return ""


def crop_irrigation_summary(question: str, passages: list[dict]):
    if not re.search(r"\b(water|irrigat\w*)\b", question, re.I):
        return None
    for passage in passages:
        if not passage.get("crop_focused"):
            continue
        match = re.search(
            r"\b(\d+)\s+irrigations?\s+applied\b", passage.get("text", ""), re.I,
        )
        crops = passage.get("context", {}).get("crop", [])
        if not match or not crops:
            continue
        crop = crops[0]
        answer = (
            f"The dataset records {match.group(1)} irrigations for {crop}; "
            f"it does not specify the timing or water amount. [{passage['id']}]"
        )
        return answer, passage
    return None


class Translator:
    """NLLB-200 (600M). NOTE: licence is CC-BY-NC 4.0 (non-commercial)."""
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            name, torch_dtype=torch.float16 if self.dev == "cuda" else torch.float32,
            local_files_only=True).to(self.dev).eval()

    def translate(self, text: str, src: str, tgt: str) -> str:
        chunks = split_chunks(text)
        if not chunks:
            return text
        self.tok.src_lang = src
        enc = self.tok(chunks, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(tgt),
                                      max_new_tokens=300, num_beams=3)
        return " ".join(self.tok.batch_decode(out, skip_special_tokens=True))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CFG["mock"]:
        import rag_chat as rag
        STATE["rag"] = rag
        print("Loading retriever ...")
        STATE["bundle"], STATE["embedder"] = rag.load_retriever(CFG["index_dir"], CFG["dataset_dir"])
        print("Loading LLM ...")
        STATE["tok"], STATE["model"] = rag.load_llm(SimpleNamespace(
            adapter=CFG["adapter"], base_model=CFG["base_model"],
            no_adapter=CFG["no_adapter"], four_bit=CFG["four_bit"]))
        if CFG["translate"]:
            print("Loading translator ...")
            STATE["translator"] = Translator(CFG["translator_model"])
        print("Ready.")
    yield


app = FastAPI(title="SAWVIAL API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskIn(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None      # accepted aliases, in case the frontend uses another name
    message: Optional[str] = None
    language: Optional[str] = "en"   # "bn", "bn-IN", "Bengali", "hi", ...
    top_k: Optional[int] = None
    history: Optional[List[ChatTurn]] = None
    request_id: Optional[str] = None


class CancelIn(BaseModel):
    request_id: str


class SpeechIn(BaseModel):
    text: str
    language: Literal["bn", "mr", "en"]
    request_id: Optional[str] = None


CANCEL_EVENTS: dict[str, threading.Event] = {}
CANCEL_EVENTS_LOCK = threading.Lock()


@app.post("/api/cancel")
def cancel(body: CancelIn):
    with CANCEL_EVENTS_LOCK:
        event = CANCEL_EVENTS.get(body.request_id)
    if event:
        event.set()
    return {"status": "cancelling", "request_id": body.request_id}


@app.post("/api/speech/stream")
def speech(body: SpeechIn):
    if not body.text.strip():
        raise HTTPException(400, "text is required")
    if len(body.text) > 2000:
        raise HTTPException(400, "text too long (max 2000 characters)")
    request_id = body.request_id or f"speech-{time.time_ns()}"
    if len(request_id) > 120:
        raise HTTPException(400, "request_id too long")
    cancel_event = threading.Event()
    with CANCEL_EVENTS_LOCK:
        if request_id in CANCEL_EVENTS:
            raise HTTPException(409, "request_id is already in use")
        CANCEL_EVENTS[request_id] = cancel_event
    try:
        try:
            from llm.speech import SAMPLE_RATE, stream_audio
        except ModuleNotFoundError as exc:
            if exc.name != "transformers":
                raise
            raise HTTPException(
                503, "Install the API dependencies from requirements.txt",
            ) from exc
        def audio_stream():
            try:
                chunks = iter(stream_audio(body.text, body.language, cancel_event))
                while not cancel_event.is_set():
                    try:
                        with LOCK:
                            chunk = next(chunks)
                    except StopIteration:
                        break
                    yield chunk
            except Exception as exc:
                print(f"Speech stream {request_id} failed: {exc}", file=sys.stderr)
                raise
            finally:
                if "chunks" in locals():
                    chunks.close()
                with CANCEL_EVENTS_LOCK:
                    CANCEL_EVENTS.pop(request_id, None)

        return StreamingResponse(
            audio_stream(),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            },
        )
    except Exception:
        with CANCEL_EVENTS_LOCK:
            CANCEL_EVENTS.pop(request_id, None)
        raise


def dataset_crop_summary(question: str, bundle: dict) -> Optional[tuple[str, list[dict]]]:
    """Answer corpus-wide WUE crop-list questions from all eligible records."""
    if not re.search(r"\b(which|what)\s+crops?\b", question, re.I):
        return None
    if not re.search(r"\b(water[\s-]+use efficiency|WUE|IWUE)\b", question, re.I):
        return None

    crop_names = {
        str(crop).strip().lower()
        for unit in bundle["units_by_id"].values()
        for crop in unit.get("context", {}).get("crop", [])
        if isinstance(crop, str) and crop.strip()
    }
    supported: dict[str, dict] = {}
    for unit in bundle["units_by_id"].values():
        if unit.get("quality", {}).get("validation_status") not in {"validated", "needs_review"}:
            continue
        evidence = unit.get("knowledge", {}).get("evidence") or ""
        title = unit.get("source", {}).get("title") or ""
        searchable = f"{evidence} {title}"
        if not re.search(r"\b(water[\s-]+use efficiency|WUE|IWUE)\b", searchable, re.I):
            continue
        for crop in crop_names.intersection(
            str(value).strip().lower()
            for value in unit.get("context", {}).get("crop", [])
        ):
            plural = crop[:-1] + "ies" if crop.endswith("y") else crop + "s"
            if re.search(rf"\b(?:{re.escape(crop)}|{re.escape(plural)})\b", searchable, re.I):
                supported.setdefault(crop, unit)

    if not supported:
        return None

    names = sorted(supported)
    sources = []
    for crop in names:
        unit = supported[crop]
        sources.append({
            "id": unit["id"],
            "title": unit.get("source", {}).get("title"),
            "page": unit.get("provenance", {}).get("page"),
            "score": 1.0,
            "cited": True,
            "snippet": unit.get("knowledge", {}).get("evidence", "")[:300],
        })
    cited_names = ", ".join(f"{crop} [{supported[crop]['id']}]" for crop in names)
    answer = (
        "Validated or reviewable dataset records explicitly mention water-use "
        f"efficiency alongside these crops: {cited_names}. This is a corpus-wide "
        "list of textual matches; it does not mean every crop was compared in "
        "the same study or under the same conditions."
    )
    return answer, sources


def safe_translate(tr, text: str, src: str, tgt: str) -> str:
    """Translate, but never fail the whole request if the translator errors."""
    if not tr or not text:
        return text
    try:
        return tr.translate(text, src, tgt)
    except Exception as exc:
        print(f"Translation {src}->{tgt} failed: {exc}", file=sys.stderr)
        return text


def run_ask(question: str, lang: str, top_k: int, history: Optional[List[ChatTurn]] = None,
             cancel_event: Optional[threading.Event] = None) -> dict:
    t0 = time.time()
    if CFG["mock"]:
        ans = "এটি একটি পরীক্ষামূলক উত্তর।" if lang == "bn" else "This is a mock answer for frontend testing."
        return {"answer": ans, "language": lang, "grounded": True, "mock": True,
                "sources": [{"id": "SWM_000000", "title": "Mock source", "page": 1, "score": 1.0,
                             "cited": True, "snippet": "Mock evidence text."}],
                "latency_ms": int((time.time() - t0) * 1000)}

    rag, tr = STATE["rag"], STATE.get("translator")
    q_en, translated = question, False
    prior_turns = []
    total_history_chars = 0
    for turn in reversed(history or []):
        content = turn.content.strip()
        if not content or len(prior_turns) >= 24 or total_history_chars >= 8000:
            continue
        content = content[:min(700, 8000 - total_history_chars)]
        prior_turns.append({"role": turn.role, "content": content})
        total_history_chars += len(content)
    prior_turns.reverse()
    with LOCK:
        if lang != "en" and tr:
            q_en = safe_translate(tr, question, NLLB[lang], "eng_Latn"); translated = True
            prior_turns = [
                {"role": turn["role"],
                 "content": safe_translate(tr, turn["content"], NLLB[lang], "eng_Latn")}
                for turn in prior_turns
            ]

        def direct_response(text: str) -> dict:
            out_text = safe_translate(tr, text, "eng_Latn", NLLB[lang]) if lang != "en" else text
            return {
                "answer": out_text,
                "language": lang,
                "grounded": False,
                "sources": [],
                "latency_ms": int((time.time() - t0) * 1000),
            }

        if GREETING.fullmatch(q_en.strip()):
            return direct_response(
                "Hello! I can help with questions about crops, soil, irrigation, "
                "water use, and soil/water conservation in this research dataset."
            )

        previous_question = find_anchor_question(prior_turns, STATE["bundle"])
        follow_up = bool(previous_question) and (
            is_follow_up(q_en) or is_formatting_request(q_en)
        )
        if not is_agriculture_question(q_en, STATE["bundle"]) and not follow_up:
            return direct_response(OUT_OF_SCOPE_ANSWER)

        summary = dataset_crop_summary(q_en, STATE["bundle"])
        if summary:
            clean, sources = summary
            clean = strip_citations(clean)
            out_text = safe_translate(tr, clean, "eng_Latn", NLLB[lang]) if lang != "en" else clean
            return {
                "answer": out_text,
                "language": lang,
                "grounded": True,
                "sources": sources,
                "latency_ms": int((time.time() - t0) * 1000),
            }
        retrieval_query = f"{previous_question} {q_en}" if follow_up else q_en
        passages = rag.get_passages(
            STATE["bundle"], STATE["embedder"], retrieval_query, top_k,
        )
        summary_question = (
            retrieval_query
            if follow_up and is_formatting_request(q_en)
            else q_en
        )
        crop_irrigation = crop_irrigation_summary(summary_question, passages)
        if crop_irrigation:
            clean, passage = crop_irrigation
            clean = strip_citations(clean)
            out_text = safe_translate(tr, clean, "eng_Latn", NLLB[lang]) if lang != "en" else clean
            return {
                "answer": out_text,
                "language": lang,
                "grounded": True,
                "sources": [{
                    "id": passage["id"],
                    "title": passage.get("title"),
                    "page": passage.get("page"),
                    "score": round(float(passage["score"]), 4),
                    "cited": True,
                    "snippet": passage["text"][:300],
                }],
                "latency_ms": int((time.time() - t0) * 1000),
            }
        if not passages or max(float(p["score"]) for p in passages) < MIN_RETRIEVAL_SCORE:
            return direct_response(NO_DATA_ANSWER)
        context_history = prior_turns if follow_up else []
        raw = rag.answer(STATE["tok"], STATE["model"], q_en, passages,
                         CFG["max_new_tokens"], history=context_history,
                         cancel_event=cancel_event) or ""
        if cancel_event is not None and cancel_event.is_set():
            return direct_response("Stopped.")
        available_citations = {p["id"] for p in passages}
        cited = find_citations(raw) & available_citations      # ignore invented ids
        clean = strip_citations(raw)
        top_score = max(float(p["score"]) for p in passages)
        fallback = extractive_answer(q_en, passages)

        unusable = (
            not clean
            or (find_citations(raw) and not cited)              # only made-up citations
            or not answer_matches_asked_crop(q_en, clean, passages)
        )
        says_no_info = clean.lower().startswith("the provided sources do not")
        if fallback and (unusable or (says_no_info and top_score >= STRONG_RETRIEVAL_SCORE)):
            clean, best = fallback
            clean = strip_citations(clean)
            cited = {best["id"]}
        elif unusable:
            return direct_response(NO_DATA_ANSWER)
        grounded = bool(passages) and not clean.lower().startswith("the provided sources do not")
        out_text = clean
        if lang != "en" and tr:
            out_text = safe_translate(tr, clean, "eng_Latn", NLLB[lang])
    res = {"answer": out_text, "language": lang, "grounded": grounded,
           "sources": [{"id": p["id"], "title": p["title"], "page": p["page"],
                        "score": round(float(p["score"]), 4), "cited": p["id"] in cited,
                        "snippet": p["text"][:300]} for p in passages],
           "latency_ms": int((time.time() - t0) * 1000)}
    if translated:
        res["question_en"] = q_en
        res["answer_en"] = clean
    if lang != "en" and not tr:
        res["note"] = "translation unavailable - answer is in English"
    return res


@app.get("/api/health")
def health():
    return {"status": "ok", "mock": CFG["mock"], "translator": bool(STATE.get("translator")),
            "adapter": None if CFG["no_adapter"] else CFG["adapter"]}


@app.post("/api/ask")
def ask(body: AskIn):
    q = (body.question or body.query or body.message or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    if len(q) > 1000:
        raise HTTPException(400, "question too long (max 1000 characters)")
    if body.request_id and len(body.request_id) > 120:
        raise HTTPException(400, "request_id too long")
    request_id = body.request_id or f"ask-{time.time_ns()}"
    cancel_event = threading.Event()
    with CANCEL_EVENTS_LOCK:
        CANCEL_EVENTS[request_id] = cancel_event
    try:
        return run_ask(q, norm_lang(body.language), max(1, min(body.top_k or CFG["top_k"], 8)),
                       history=body.history, cancel_event=cancel_event)
    finally:
        with CANCEL_EVENTS_LOCK:
            CANCEL_EVENTS.pop(request_id, None)


# ---- serve the frontend (only the html pages + css/js/assets, NOT the python package) ----
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no_translate", action="store_true")
    ap.add_argument("--no_adapter", action="store_true")
    ap.add_argument("--use_adapter", dest="no_adapter", action="store_false",
                    help="Use the experimental fine-tuned adapter (disabled by default).")
    ap.set_defaults(no_adapter=True)
    ap.add_argument("--adapter", default=CFG["adapter"])
    ap.add_argument("--dataset_dir", default=CFG["dataset_dir"])
    ap.add_argument("--index_dir", default=CFG["index_dir"])
    a = ap.parse_args()
    CFG.update(mock=a.mock, translate=not a.no_translate, no_adapter=a.no_adapter,
               adapter=a.adapter, dataset_dir=a.dataset_dir, index_dir=a.index_dir)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
