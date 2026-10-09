"""
sawvial.config
==============
Central configuration: loads taxonomy.json, exposes allowed domains,
and holds reproducibility metadata (versions, chunking config, etc.)

Nothing in this module hard-codes the taxonomy — it is entirely
data-driven from taxonomy.json so the domain hierarchy can be edited
without touching Python code.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

SOFTWARE_VERSION = "2.0.0"


class Taxonomy:
    """Loads and exposes the taxonomy.json contents."""

    def __init__(self, taxonomy_path: str | Path):
        self.path = Path(taxonomy_path)
        with open(self.path, encoding="utf-8") as f:
            self._raw: Dict[str, Any] = json.load(f)

        self.version: str = self._raw.get("taxonomy_version", "unknown")
        self.allowed_domains: List[str] = list(self._raw["allowed_domains"])
        self.domains: Dict[str, Any] = self._raw["domains"]
        self.concept_aliases: Dict[str, List[str]] = self._raw.get("concept_aliases", {})
        self.context_vocab: Dict[str, List[str]] = self._raw.get("context_vocabularies", {})
        self.importance_signals: Dict[str, Any] = self._raw.get("importance_signals", {})

        # Build reverse index: subdomain -> domain, topic -> (domain, subdomain)
        self.subdomain_to_domain: Dict[str, str] = {}
        self.topic_to_domain_subdomain: Dict[str, tuple] = {}
        # topic/subdomain keyword surface forms for scoring (derived from names)
        self.domain_keyword_index: Dict[str, Dict[str, List[str]]] = {}

        for domain, dconf in self.domains.items():
            assert domain in self.allowed_domains, (
                f"taxonomy.json defines domain '{domain}' outside allowed_domains"
            )
            self.domain_keyword_index[domain] = {}
            for subdomain, sconf in dconf.get("subdomains", {}).items():
                self.subdomain_to_domain[subdomain] = domain
                topics = sconf.get("topics", [])
                surface_terms = [subdomain.replace("_", " ")]
                for topic in topics:
                    self.topic_to_domain_subdomain[topic] = (domain, subdomain)
                    surface_terms.append(topic.replace("_", " "))
                self.domain_keyword_index[domain][subdomain] = surface_terms

    def validate_domain(self, domain: str) -> bool:
        return domain in self.allowed_domains

    def resolve_concept(self, surface_form: str) -> str:
        """Normalize a surface form (e.g. 'SOC') to its canonical concept id."""
        sf = surface_form.strip().lower()
        for canonical, aliases in self.concept_aliases.items():
            if sf == canonical.replace("_", " ") or sf in [a.lower() for a in aliases]:
                return canonical
        return sf.replace(" ", "_")


@dataclass
class ChunkingConfig:
    min_chunk_chars: int = 80
    max_chunk_chars: int = 1600
    target_chunk_chars: int = 700


@dataclass
class ExtractionConfig:
    min_importance_score: float = 0.28   # below this -> rejected / evidence_only
    review_importance_score: float = 0.4  # below this -> needs_review


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    offline: bool = False
    normalize: bool = True
    dimension: int = 0  # filled in after model load


@dataclass
class BuildConfig:
    taxonomy: Taxonomy
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    input_dir: str = "./papers"
    output_dir: str = "./dataset"
    index_dir: str = "./index"

    def build_metadata(self) -> Dict[str, Any]:
        return {
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "software_version": SOFTWARE_VERSION,
            "taxonomy_version": self.taxonomy.version,
            "python_version": platform.python_version(),
            "embedding_model": self.embedding.model_name,
            "embedding_offline_mode": self.embedding.offline,
            "chunking_config": self.chunking.__dict__,
            "extraction_config": self.extraction.__dict__,
        }
