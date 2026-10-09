"""Progressive local speech synthesis for Bengali, Marathi, and English."""
import re
import threading
from typing import Iterator, Optional

import torch
from transformers import AutoTokenizer, VitsModel

MODEL_IDS = {
    "bn": "facebook/mms-tts-ben",
    "mr": "facebook/mms-tts-mar",
    "en": "facebook/mms-tts-eng",
}
SAMPLE_RATE = 16000
MAX_CHUNK_CHARS = 200

_MODEL_LOCK = threading.Lock()
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODELS: dict[str, tuple[VitsModel, AutoTokenizer]] = {}


def _load_model(language: str) -> tuple[VitsModel, AutoTokenizer]:
    cached = _MODELS.get(language)
    if cached is not None:
        return cached

    model_id = MODEL_IDS[language]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(_DEVICE).eval()
    _MODELS[language] = (model, tokenizer)
    return model, tokenizer


def _split_text(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text.strip())
    chunks: list[str] = []
    current = ""
    for part in filter(None, parts):
        while len(part) > MAX_CHUNK_CHARS:
            split_at = part.rfind(" ", 0, MAX_CHUNK_CHARS + 1)
            if split_at <= 0:
                split_at = MAX_CHUNK_CHARS
            if current:
                chunks.append(current)
                current = ""
            chunks.append(part[:split_at].strip())
            part = part[split_at:].strip()
        if current and len(current) + len(part) + 1 > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = ""
        current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def stream_audio(
    text: str,
    language: str,
    cancel_event: Optional[threading.Event] = None,
) -> Iterator[bytes]:
    if language not in MODEL_IDS:
        raise ValueError(f"Unsupported speech language: {language}")
    if not text.strip():
        raise ValueError("Speech text must not be empty")

    for chunk in _split_text(text):
        if cancel_event and cancel_event.is_set():
            return
        with _MODEL_LOCK, torch.inference_mode():
            model, tokenizer = _load_model(language)
            inputs = tokenizer(chunk, return_tensors="pt").to(_DEVICE)
            waveform = model(**inputs).waveform[0].float().cpu().numpy()
        if cancel_event and cancel_event.is_set():
            return
        if waveform.size:
            yield waveform.astype("<f4", copy=False).tobytes()
