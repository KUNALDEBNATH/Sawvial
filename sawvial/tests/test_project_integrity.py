"""Whole-project checks: nothing references a file or module that is missing (the 'overlay ZIP' failure mode)."""
import importlib
import pkgutil
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "sawvial"


def test_every_sawvial_module_imports():
    import sawvial
    failures = []
    for m in pkgutil.walk_packages(sawvial.__path__, "sawvial."):
        try:
            importlib.import_module(m.name)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{m.name}: {exc!r}")
    assert not failures, failures


@pytest.mark.parametrize("mod", ["api", "build_dataset", "research_cli", "evaluate_rag"])
def test_entry_points_import(mod):
    importlib.import_module(mod)


def test_llm_modules_import_when_torch_is_installed():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sys.path.insert(0, str(ROOT / "llm"))
    for m in ("common", "rag_chat", "speech", "prepare_sft_data"):
        importlib.import_module(m)


def test_html_pages_reference_only_existing_files():
    missing = []
    for page in WEB.glob("*.html"):
        text = page.read_text(encoding="utf-8")
        for ref in re.findall(r'(?:src|href)="([^"#?]+)(?:\?[^"]*)?"', text):
            if re.match(r"^(https?:|mailto:|tel:|data:|javascript:|/)", ref) or ref.endswith(".html"):
                if ref.endswith(".html") and not re.match(r"^(https?:|/)", ref) and not (WEB / ref).exists():
                    missing.append(f"{page.name} -> {ref}")
                continue
            if not (WEB / ref).exists():
                missing.append(f"{page.name} -> {ref}")
    assert not missing, missing


def test_required_project_files_exist():
    for f in ("taxonomy.json", "requirements.txt", "requirements-llm.txt", ".env.example", "README.md",
              "llm/sawvial-lora/adapter_config.json", "llm/sawvial-lora/adapter_model.safetensors",
              "dataset/knowledge_units.jsonl", "dataset/documents.jsonl", "tools/download_models.py", "tools/audit_dataset.py"):
        assert (ROOT / f).exists(), f


def test_every_settings_variable_is_documented():
    code = "".join(p.read_text(encoding="utf-8") for p in ROOT.rglob("*.py")
                   if ".git" not in p.parts and "tests" not in p.parts and "node_modules" not in p.parts)
    used = set(re.findall(r"SAWVIAL_[A-Z_]+", code)) - {"SAWVIAL_TEST_A"}
    documented = set(re.findall(r"SAWVIAL_[A-Z_]+", (ROOT / ".env.example").read_text()))
    assert used <= documented, sorted(used - documented)


def test_frontend_calls_endpoints_that_exist():
    import api
    routes = {r.path for r in api.app.routes}
    js = (WEB / "js" / "sawvial-api.js").read_text(encoding="utf-8")
    for ep in set(re.findall(r'"(/api/[a-z/]+)"', js)):
        assert ep in routes, f"frontend calls {ep} but the API has no such route"


def test_no_api_keys_or_secret_settings_exist_anywhere():
    banned = re.compile(r"(?i:API_KEY|X-Subscription|Bearer |brave|searxng)|\b[A-Z][A-Z_]*_TOKEN\b")
    hits = []
    for p in list(ROOT.rglob("*.py")) + [ROOT / ".env.example", ROOT / "README.md", ROOT / "requirements.txt", ROOT / "requirements-llm.txt"]:
        if ".git" in p.parts or "node_modules" in p.parts or p.name.startswith("test_"):
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if banned.search(line) and "no api key" not in line.lower() and "no api keys" not in line.lower():
                hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:90]}")
    assert not hits, hits
