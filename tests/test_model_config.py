import json
import struct
from pathlib import Path

import pytest

from sawvial.model_config import inspect_adapter

REAL = Path(__file__).resolve().parent.parent / "llm" / "sawvial-lora"


def fake_adapter(d: Path, base="Qwen/Qwen2.5-1.5B-Instruct", hidden=1536, layers=2, peft="LORA"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base, "peft_type": peft, "r": 8}))
    hdr = {f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight": {"dtype": "F32", "shape": [8, hidden], "data_offsets": [0, 0]}
           for i in range(layers)}
    raw = json.dumps(hdr).encode()
    (d / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    return d


def test_valid_adapter(tmp_path):
    r = inspect_adapter(fake_adapter(tmp_path / "a"), "Qwen/Qwen2.5-1.5B-Instruct", 1536, 2)
    assert r.ok and r.num_layers == 2 and r.hidden_size == 1536


def test_wrong_base_model_is_an_error_with_fix_hint(tmp_path):
    r = inspect_adapter(fake_adapter(tmp_path / "a"), "Qwen/Qwen2.5-3B-Instruct")
    assert r.status == "invalid" and "SAWVIAL_BASE_MODEL" in r.message()


def test_shape_mismatch_detected(tmp_path):
    r = inspect_adapter(fake_adapter(tmp_path / "a"), "Qwen/Qwen2.5-1.5B-Instruct", base_hidden_size=2048, base_num_layers=36)
    assert r.status == "invalid" and "hidden size" in r.message() and "layers" in r.message()


def test_missing_and_corrupt(tmp_path):
    assert inspect_adapter(tmp_path / "none", "x").status == "missing"
    d = fake_adapter(tmp_path / "bad"); (d / "adapter_model.safetensors").write_bytes(b"garbage")
    assert inspect_adapter(d, "Qwen/Qwen2.5-1.5B-Instruct").status == "invalid"
    d2 = fake_adapter(tmp_path / "ia3", peft="IA3")
    assert inspect_adapter(d2, "Qwen/Qwen2.5-1.5B-Instruct").status == "invalid"


@pytest.mark.skipif(not (REAL / "adapter_config.json").exists(), reason="repo adapter not present")
def test_repository_adapter_matches_qwen25_15b():
    r = inspect_adapter(REAL, "Qwen/Qwen2.5-1.5B-Instruct", base_hidden_size=1536, base_num_layers=28)
    assert r.ok, r.message()
