"""Central runtime configuration for SAWVIAL.

Every value can be overridden with an environment variable (see .env.example).
Nothing here is a secret. Paths are resolved relative to the project root, never
relative to the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> int:
    """Minimal .env reader (KEY=VALUE per line, # comments). Real environment variables always win.
    Returns how many variables were set. No dependency needed."""
    n = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key.startswith("SAWVIAL_") and value and key not in os.environ:
            os.environ[key] = value
            n += 1
    return n


load_env_file(ROOT / ".env")


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

# ---- hardware -------------------------------------------------------------
# cuda (default): use the NVIDIA GPU and REFUSE to start on CPU by accident (e.g. a CPU-only PyTorch build).
# auto: use the GPU if PyTorch can see one, else CPU.   cpu: force CPU.
DEVICE = env_str("SAWVIAL_DEVICE", "cuda").lower()

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
# Key-less scholarly search: auto (= OpenAlex + Crossref) | openalex | crossref | none. No API key is used anywhere.
SEARCH_PROVIDER = env_str("SAWVIAL_SEARCH_PROVIDER", "auto")
FETCH_TIMEOUT_S = env_int("SAWVIAL_FETCH_TIMEOUT_S", 20)
FETCH_MAX_BYTES = env_int("SAWVIAL_FETCH_MAX_MB", 25) * 1024 * 1024
FETCH_MAX_REDIRECTS = 3
RESEARCH_MAX_PDF_PAGES = env_int("SAWVIAL_RESEARCH_MAX_PDF_PAGES", 120)
# If non-empty, ONLY these domains (and sub-domains) may be fetched for research.
RESEARCH_ALLOWED_DOMAINS = env_list("SAWVIAL_RESEARCH_ALLOWED_DOMAINS", "")

# ---- online research inside /api/ask ----------------------------------------------
ONLINE_RESEARCH = env_bool("SAWVIAL_ONLINE_RESEARCH", True)     # False = local knowledge base only
RESEARCH_MAX_DOCS = env_int("SAWVIAL_RESEARCH_MAX_DOCS", 4)       # documents read per question
RESEARCH_TIME_BUDGET_S = env_int("SAWVIAL_RESEARCH_TIME_BUDGET_S", 60)
# Minimum trust tier of the FINAL (post-redirect) site for a page to support an answer:
#   official (ICAR/TNAU/FAO/CGIAR...) | reputable (+ universities, publishers, Vikaspedia) | any
RESEARCH_MIN_TRUST = env_str("SAWVIAL_RESEARCH_MIN_TRUST", "reputable")
RESEARCH_CACHE_DAYS = env_int("SAWVIAL_RESEARCH_CACHE_DAYS", 30)  # re-use already-downloaded documents
RESEARCH_HOST_DELAY_S = float(env_str("SAWVIAL_RESEARCH_HOST_DELAY_S", "1.0"))   # min gap between requests to one host
RESPECT_ROBOTS = env_bool("SAWVIAL_RESPECT_ROBOTS", True)
