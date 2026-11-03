"""
Research record store (JSONL, same knowledge-unit schema as dataset/knowledge_units.jsonl).

    research_store/records.jsonl     one knowledge unit per line (ids SWR_xxxxxx)
    research_store/sources.jsonl     one line per downloaded document (dedupe + provenance)
    research_store/downloads/        the original downloaded files (named by sha256)
    research_store/reports/          one JSON report per research run

Status workflow (definitions in sawvial/eligibility.py):

    pending_review --verify(reviewer)--> verified --publish--> dataset/knowledge_units.jsonl
          |                                 |
          +-------------reject--------------+----> rejected

Nothing becomes answerable without an explicit, named human `verify`. Generated answers are
never stored as knowledge: only extracted source evidence is.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sawvial import settings

TRANSITIONS = {
    "pending_review": {"verified", "rejected"},
    "verified": {"rejected"},
    "rejected": {"pending_review"},
}


def _read(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _norm_url(u: Optional[str]) -> Optional[str]:
    return u.split("#")[0].rstrip("/").lower() if u else None


class ResearchStore:
    def __init__(self, directory: Optional[Path] = None):
        self.dir = Path(directory or settings.RESEARCH_DIR)
        self.records_file, self.sources_file = self.dir / "records.jsonl", self.dir / "sources.jsonl"
        self.downloads, self.reports = self.dir / "downloads", self.dir / "reports"

    # ---- sources (documents) ----
    def sources(self) -> List[Dict[str, Any]]:
        return _read(self.sources_file)

    def find_duplicate_source(self, doi: Optional[str], url: Optional[str], sha256: Optional[str]):
        for s in self.sources():
            if (sha256 and s.get("content_sha256") == sha256) or \
               (doi and s.get("doi") and s["doi"].lower() == doi.lower()) or \
               (url and _norm_url(s.get("final_url")) == _norm_url(url)):
                return s
        return None

    def find_cached_source(self, url: str, max_age_days: int) -> Optional[Dict[str, Any]]:
        """A source we already downloaded from this URL recently (re-used instead of downloading again)."""
        now, want = datetime.now(timezone.utc), _norm_url(url)
        for s in self.sources():
            if want in (_norm_url(s.get("requested_url")), _norm_url(s.get("final_url"))):
                try:
                    age = now - datetime.fromisoformat(s["retrieved_at"])
                except (KeyError, ValueError):
                    continue
                if age.days <= max_age_days:
                    return s
        return None

    def records_for_urls(self, urls: Iterable[str]) -> List[Dict[str, Any]]:
        """Non-rejected records collected from these (final) URLs."""
        want = {_norm_url(u) for u in urls}
        return [r for r in self.records() if r["quality"]["validation_status"] != "rejected"
                and _norm_url(r["source"].get("final_url") or r["source"].get("url")) in want]

    def add_source(self, source: Dict[str, Any]) -> None:
        _write(self.sources_file, self.sources() + [source])

    # ---- records (units) ----
    def records(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = _read(self.records_file)
        return [r for r in rows if status is None or r["quality"]["validation_status"] == status]

    def add_records(self, units: List[Dict[str, Any]]) -> Dict[str, int]:
        """Append units (status must already be pending_review). Skips duplicate evidence."""
        rows = self.records()
        seen = {r["provenance"]["evidence_hash"] for r in rows}
        nxt = 1 + max((int(r["id"].split("_")[1]) for r in rows), default=0)
        added = dup = 0
        for u in units:
            if u["provenance"]["evidence_hash"] in seen:
                dup += 1
                continue
            seen.add(u["provenance"]["evidence_hash"])
            u["id"] = f"SWR_{nxt:06d}"
            nxt += 1
            u["quality"]["validation_status"] = "pending_review"
            rows.append(u)
            added += 1
        _write(self.records_file, rows)
        return {"added": added, "duplicates_skipped": dup}

    def set_status(self, ids: Iterable[str], new_status: str, reviewer: str, note: str = "") -> List[str]:
        """Human review decision. Returns the ids changed; raises ValueError on a bad request."""
        if new_status not in {"verified", "rejected", "pending_review"}:
            raise ValueError(f"invalid status {new_status!r}")
        if not reviewer.strip():
            raise ValueError("a reviewer name is required (--reviewer)")
        wanted, changed = set(ids), []
        rows = self.records()
        known = {r["id"] for r in rows}
        if wanted - known:
            raise ValueError(f"unknown record ids: {sorted(wanted - known)}")
        for r in rows:
            if r["id"] not in wanted:
                continue
            cur = r["quality"]["validation_status"]
            if new_status not in TRANSITIONS.get(cur, set()):
                raise ValueError(f"{r['id']}: cannot change {cur} -> {new_status}")
            if new_status == "verified" and r["quality"].get("flags"):
                raise ValueError(f"{r['id']} is flagged ({r['quality']['flags']}); it cannot be verified")
            r["quality"]["validation_status"] = new_status
            r["quality"].setdefault("review_log", []).append({
                "status": new_status, "reviewer": reviewer.strip(), "note": note,
                "at": datetime.now(timezone.utc).isoformat()})
            changed.append(r["id"])
        _write(self.records_file, rows)
        return changed

    def publishable(self) -> List[Dict[str, Any]]:
        return [r for r in self.records("verified") if not r["quality"].get("published_as")]

    def mark_published(self, mapping: Dict[str, str]) -> None:
        rows = self.records()
        for r in rows:
            if r["id"] in mapping:
                r["quality"]["published_as"] = mapping[r["id"]]
        _write(self.records_file, rows)
