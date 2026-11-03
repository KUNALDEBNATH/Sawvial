"""Progressive local speech synthesis (Meta MMS-TTS / VITS) for the 7 SAWVIAL languages.

Honest limitations
------------------
* Only languages whose model is already in the local Hugging Face cache can be spoken
  locally (``python tools/download_models.py`` fetches them). For other languages the
  frontend falls back to the browser's own speech synthesis, which depends on the voices
  installed on the user's device.
* bn / mr / en were used by the original project. hi / ta / te / kn use the matching
  facebook/mms-tts-* checkpoints; they have NOT been listened to or verified in this
  repository's test environment and MMS voices are single, fairly robotic voices - they
  do not provide regional accents.
"""
import re
import threading
from typing import Iterator, Optional

from sawvial import settings

MODEL_IDS = {
    "bn": "facebook/mms-tts-ben",
    "mr": "facebook/mms-tts-mar",
    "en": "facebook/mms-tts-eng",
    "hi": "facebook/mms-tts-hin",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "kn": "facebook/mms-tts-kan",
}
SAMPLE_RATE = 16000
MAX_CHUNK_CHARS = 200

_MODEL_LOCK = threading.Lock()
_MODELS: dict = {}


class SpeechUnavailable(RuntimeError):
    """Local speech model for this language is not available (message tells how to fix)."""


def is_cached(language: str) -> bool:
    """True if the model for `language` is in the local HF cache (no network, no torch import)."""
    model_id = MODEL_IDS.get(language)
    if not model_id:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(model_id, "config.json")
        return isinstance(found, str)
    except Exception:
        return False


def available_languages() -> list:
    return [code for code in MODEL_IDS if is_cached(code)]


def _load_model(language: str):
    cached = _MODELS.get(language)
    if cached is not None:
        return cached
    import torch
    from transformers import AutoTokenizer, VitsModel
    model_id = MODEL_IDS[language]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=settings.OFFLINE)
        device = "cuda" if (settings.DEVICE != "cpu" and torch.cuda.is_available()) else "cpu"
        model = VitsModel.from_pretrained(model_id, local_files_only=settings.OFFLINE).to(device).eval()
    except OSError as exc:
        raise SpeechUnavailable(
            f"Local speech model {model_id} is not downloaded. Run: python tools/download_models.py "
            f"--speech (or set SAWVIAL_OFFLINE=0 to allow download on first use).") from exc
    _MODELS[language] = (model, tokenizer, device)
    return _MODELS[language]


def _split_text(text: str) -> list:
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text.strip())
    chunks: list = []
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


def stream_audio(text: str, language: str,
                 cancel_event: Optional[threading.Event] = None) -> Iterator[bytes]:
    if language not in MODEL_IDS:
        raise ValueError(f"Unsupported speech language: {language}")
    if not text.strip():
        raise ValueError("Speech text must not be empty")
    import torch

    for chunk in _split_text(text):
        if cancel_event and cancel_event.is_set():
            return
        with _MODEL_LOCK, torch.inference_mode():
            model, tokenizer, device = _load_model(language)
            inputs = tokenizer(chunk, return_tensors="pt").to(device)
            waveform = model(**inputs).waveform[0].float().cpu().numpy()
        if cancel_event and cancel_event.is_set():
            return
        if waveform.size:
            yield waveform.astype("<f4", copy=False).tobytes()
