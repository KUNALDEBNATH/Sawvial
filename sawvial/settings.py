"""Central runtime configuration for SAWVIAL.

Every value can be overridden with an environment variable (see .env.example).
Nothing here is a secret. Paths are resolved relative to the project root, never
relative to the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    return env_str(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in env_str(name, default).split(",") if x.strip()]


def resolve_path(value: str | os.PathLike) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (ROOT / p)


# ---- data locations -------------------------------------------------------
DATASET_DIR = resolve_path(env_str("SAWVIAL_DATASET_DIR", "dataset"))
INDEX_DIR = resolve_path(env_str("SAWVIAL_INDEX_DIR", "index"))
RESEARCH_DIR = resolve_path(env_str("SAWVIAL_RESEARCH_DIR", "research_store"))

# ---- models ---------------------------------------------------------------
BASE_MODEL = env_str("SAWVIAL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = resolve_path(env_str("SAWVIAL_ADAPTER", "llm/sawvial-lora"))
TRANSLATOR_MODEL = env_str("SAWVIAL_TRANSLATOR_MODEL", "facebook/nllb-200-distilled-600M")
# True  -> never contact the Hugging Face hub at run time (models must be pre-downloaded
#          with `python tools/download_models.py`).
OFFLINE = env_bool("SAWVIAL_OFFLINE", True)

# ---- API ------------------------------------------------------------------
CORS_ORIGINS = env_list("SAWVIAL_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
RATE_LIMIT_PER_MIN = env_int("SAWVIAL_RATE_LIMIT_PER_MIN", 30)   # 0 disables
MAX_QUESTION_CHARS = 1000
TOP_K = env_int("SAWVIAL_TOP_K", 4)
MAX_NEW_TOKENS = env_int("SAWVIAL_MAX_NEW_TOKENS", 300)

# ---- retrieval thresholds -------------------------------------------------
# Lexical (TF-IDF cosine) score below which retrieved text is treated as "no evidence".
MIN_RETRIEVAL_SCORE = float(env_str("SAWVIAL_MIN_RETRIEVAL_SCORE", "0.03"))

# ---- online research ------------------------------------------------------
SEARCH_PROVIDER = env_str("SAWVIAL_SEARCH_PROVIDER", "openalex")   # openalex | crossref | none
SEARCH_CONTACT_EMAIL = env_str("SAWVIAL_CONTACT_EMAIL", "")       # polite-pool courtesy for OpenAlex/Crossref
FETCH_TIMEOUT_S = env_int("SAWVIAL_FETCH_TIMEOUT_S", 20)
FETCH_MAX_BYTES = env_int("SAWVIAL_FETCH_MAX_MB", 25) * 1024 * 1024
FETCH_MAX_REDIRECTS = 3
RESEARCH_MAX_PDF_PAGES = env_int("SAWVIAL_RESEARCH_MAX_PDF_PAGES", 120)
# If non-empty, ONLY these domains (and sub-domains) may be fetched for research.
RESEARCH_ALLOWED_DOMAINS = env_list("SAWVIAL_RESEARCH_ALLOWED_DOMAINS", "")
