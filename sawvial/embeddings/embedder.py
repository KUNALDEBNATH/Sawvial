"""
sawvial.embeddings.embedder
==============================
Keeps the embedding backend modular and configurable, per master-spec
sections 19/28: no dataset logic hard-codes a specific embedding
model. Supports:

    - sentence-transformers (default: all-MiniLM-L6-v2; also works
      with multilingual models and BAAI/bge-m3 by passing --model)
    - a fully offline TF-IDF fallback with no external downloads or
      commercial APIs required

The embedding configuration actually used (model name, dimension,
offline flag) is recorded and persisted in index_metadata.json for
reproducibility (sec. 29).
"""
from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

log = logging.getLogger("sawvial.embedder")

_WORD = re.compile(r"(?u)\b\w\w+\b")


def _tokenize(text: str) -> List[str]:
    """Normalize common English plurals so user queries match corpus terms."""
    tokens = []
    for token in _WORD.findall(text.lower()):
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("oes"):
            token = token[:-2]
        elif (len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes", "sses"))):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        tokens.append(token)
    return tokens


def make_tfidf_vectorizer(max_features: int | None = 768):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    stop_words = [
        token for word in ENGLISH_STOP_WORDS for token in _tokenize(word)
    ]
    return TfidfVectorizer(
        max_features=max_features, sublinear_tf=True, stop_words=stop_words,
        tokenizer=_tokenize, token_pattern=None, ngram_range=(1, 2),
    )


class TFIDFEmbedder:
    """Fully offline embedder. Deterministic, no downloads required."""

    def __init__(self, max_features: int = 768):
        self._vectorizer = make_tfidf_vectorizer(max_features)
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        from sklearn.preprocessing import normalize as sk_normalize
        if not self._fitted:
            self.fit(texts)
        mat = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return sk_normalize(mat, norm="l2")

    @property
    def dimension(self) -> int:
        return len(self._vectorizer.vocabulary_) if self._fitted else 0


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=32, show_progress_bar=len(texts) > 50,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


def load_embedder(model_name: str, offline: bool):
    if offline:
        log.info("Offline mode requested — using TF-IDF embedder.")
        return TFIDFEmbedder()
    try:
        log.info(f"Loading embedding model: {model_name}")
        return SentenceTransformerEmbedder(model_name)
    except Exception as e:
        log.warning(f"Could not load '{model_name}' ({e}); falling back to TF-IDF.")
        return TFIDFEmbedder()


def embed_texts(embedder, texts: List[str]) -> np.ndarray:
    """Used only during dataset embedding (build time). Fits a TF-IDF
    embedder on the full corpus being embedded. NOT used for query-time
    encoding — see sawvial.index.retriever.query, which must reuse an
    already-fitted embedder rather than refitting on the single query
    string."""
    if isinstance(embedder, TFIDFEmbedder) and not embedder._fitted:
        embedder.fit(texts)
    return embedder.encode(texts)
