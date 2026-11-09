import json
import types

import pytest

from sawvial import settings
from sawvial.device import pick_device
from sawvial.index import retriever as R
from sawvial.model_config import ModelConfigError


def fake_torch(cuda_available, cuda_version, name="NVIDIA GeForce RTX 4060 Laptop GPU", mem=8 * 1024 ** 3):
    return types.SimpleNamespace(
        __version__="2.8.0", version=types.SimpleNamespace(cuda=cuda_version),
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available,
                                   get_device_properties=lambda i: types.SimpleNamespace(name=name, total_memory=mem)))


# ---------------------------------------------------------------- GPU policy
def test_uses_the_gpu_when_available(monkeypatch):
    monkeypatch.setattr(settings, "DEVICE", "cuda")
    d = pick_device(fake_torch(True, "12.8"))
    assert d["device"] == "cuda" and "4060" in d["name"] and d["vram_mb"] == 8192


def test_cpu_only_torch_build_is_refused_with_the_exact_fix(monkeypatch):
    monkeypatch.setattr(settings, "DEVICE", "cuda")
    with pytest.raises(ModelConfigError) as e:
        pick_device(fake_torch(False, None))
    msg = str(e.value)
    assert "CPU-only build" in msg and "download.pytorch.org/whl/cu128" in msg and "--force-reinstall" in msg


def test_cuda_build_but_no_visible_gpu_points_at_the_driver(monkeypatch):
    monkeypatch.setattr(settings, "DEVICE", "cuda")
    with pytest.raises(ModelConfigError, match="nvidia-smi"):
        pick_device(fake_torch(False, "12.8"))


def test_auto_and_cpu_modes_are_explicit_choices(monkeypatch):
    monkeypatch.setattr(settings, "DEVICE", "auto")
    assert pick_device(fake_torch(False, None))["device"] == "cpu"
    monkeypatch.setattr(settings, "DEVICE", "cpu")
    assert pick_device(fake_torch(True, "12.8"))["device"] == "cpu"


def test_default_policy_is_cuda():
    import importlib, os
    assert os.environ.get("SAWVIAL_DEVICE") is None and importlib.reload(settings).DEVICE == "cuda"


def test_llm_loader_never_defaults_to_4bit_and_uses_the_device_policy():
    src = (settings.ROOT / "llm" / "rag_chat.py").read_text(encoding="utf-8")
    assert 'use_4bit = bool(getattr(args, "four_bit", False))' in src and "pick_device(torch)" in src


# ---------------------------------------------------------------- the index repairs itself - no manual command
def test_missing_index_is_built_automatically(built):
    import shutil
    shutil.rmtree(built["index"])
    b = R.ensure_index_bundle(built["index"], built["dataset"])
    assert len(b["ids"]) == 6 and not b["stale"]


def test_half_committed_index_is_repaired(built):
    (built["index"] / "lexical_matrix.npz").unlink()          # the situation in a fresh git clone
    assert len(R.ensure_index_bundle(built["index"], built["dataset"])["ids"]) == 6


def test_stale_index_is_rebuilt_to_include_new_records(built, units):
    from tests.conftest import make_unit
    new = make_unit("SWM_000050", "Sprinkler irrigation of sunflower improved water productivity under semi arid conditions of the deccan region.", ["sunflower"], doc="dnew")
    with open(built["dataset"] / "knowledge_units.jsonl", "a") as f:
        f.write(json.dumps(new) + "\n")
    assert "SWM_000050" in R.ensure_index_bundle(built["index"], built["dataset"])["ids"]


def test_index_built_with_another_scikit_learn_is_rebuilt(built):
    meta = json.loads((built["index"] / "index_metadata.json").read_text()); meta["sklearn_version"] = "0.0.1"
    (built["index"] / "index_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(R.IndexRebuildNeeded, match="scikit-learn"):
        R.load_index_bundle(built["index"], built["dataset"])
    b = R.ensure_index_bundle(built["index"], built["dataset"])
    assert json.loads((built["index"] / "index_metadata.json").read_text())["sklearn_version"] != "0.0.1" and b["ids"]


def test_corrupt_vectorizer_is_repaired(built):
    (built["index"] / "lexical_vectorizer.joblib").write_bytes(b"garbage")
    assert R.ensure_index_bundle(built["index"], built["dataset"])["ids"]


def test_healthy_index_is_not_rebuilt(built, monkeypatch):
    from sawvial.index import faiss_builder
    monkeypatch.setattr(faiss_builder, "build_index", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not rebuild")))
    assert R.ensure_index_bundle(built["index"], built["dataset"])["ids"]


def test_clear_error_when_the_dataset_itself_is_missing(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        R.ensure_index_bundle(tmp_path / "index", tmp_path / "dataset")
