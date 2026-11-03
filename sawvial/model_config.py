"""Torch-free checks that a LoRA adapter really belongs to the base model.

Reads only small JSON files and the safetensors *header* (no weights are loaded), so it
runs instantly and works in tests without torch/transformers installed.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ModelConfigError(RuntimeError):
    """A model/adapter problem the user can fix; the message says how."""


@dataclass
class AdapterCheck:
    status: str                      # "ok" | "missing" | "invalid"
    path: str
    base_model: Optional[str] = None
    rank: Optional[int] = None
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    four_bit_trained: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def message(self) -> str:
        return "; ".join(self.problems) if self.problems else self.status


def _safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("file too short to be safetensors")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 100_000_000:
            raise ValueError("implausible safetensors header size")
        return json.loads(f.read(n))


def inspect_adapter(adapter_dir: str | Path, base_model: str,
                    base_hidden_size: Optional[int] = None,
                    base_num_layers: Optional[int] = None) -> AdapterCheck:
    """Validate an adapter directory against the configured base model.

    base_hidden_size / base_num_layers are optional (taken from the base model's
    config.json when it is available locally); when given they are cross-checked
    against the tensor shapes stored in the adapter.
    """
    d = Path(adapter_dir)
    res = AdapterCheck(status="ok", path=str(d))
    cfg_file, weights = d / "adapter_config.json", d / "adapter_model.safetensors"
    if not d.is_dir() or not cfg_file.exists() or not weights.exists():
        res.status = "missing"
        res.problems.append(
            f"adapter files not found in {d} (need adapter_config.json and adapter_model.safetensors)")
        return res
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    except Exception as exc:
        res.status = "invalid"
        res.problems.append(f"adapter_config.json unreadable: {exc}")
        return res

    res.base_model = cfg.get("base_model_name_or_path")
    res.rank = cfg.get("r")
    if cfg.get("peft_type") != "LORA":
        res.problems.append(f"peft_type is {cfg.get('peft_type')!r}, expected 'LORA'")
    if res.base_model and res.base_model != base_model:
        res.problems.append(
            f"adapter was trained for base model {res.base_model!r} but the configured base is "
            f"{base_model!r}; set SAWVIAL_BASE_MODEL or pass --base_model {res.base_model}")

    info_file = d / "sawvial_train_info.json"
    if info_file.exists():
        try:
            info = json.loads(info_file.read_text(encoding="utf-8"))
            res.four_bit_trained = bool(info.get("four_bit"))
            if info.get("base_model") and info["base_model"] != base_model:
                res.problems.append(
                    f"sawvial_train_info.json says base model {info['base_model']!r}, "
                    f"configured base is {base_model!r}")
        except Exception as exc:
            res.problems.append(f"sawvial_train_info.json unreadable: {exc}")

    try:
        header = _safetensors_header(weights)
    except Exception as exc:
        res.status = "invalid"
        res.problems.append(f"adapter_model.safetensors unreadable: {exc}")
        return res
    layers, q_in = set(), None
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        m = re.search(r"layers\.(\d+)\.self_attn\.q_proj\.lora_A", name)
        if m:
            layers.add(int(m.group(1)))
            q_in = meta["shape"][1]
        if re.search(r"layers\.\d+\.", name):
            layers.add(int(re.search(r"layers\.(\d+)\.", name).group(1)))
    res.num_layers = (max(layers) + 1) if layers else None
    res.hidden_size = q_in
    if not layers:
        res.problems.append("adapter contains no per-layer LoRA tensors")
    if base_hidden_size and q_in and q_in != base_hidden_size:
        res.problems.append(f"adapter expects hidden size {q_in}, base model has {base_hidden_size}")
    if base_num_layers and res.num_layers and res.num_layers != base_num_layers:
        res.problems.append(f"adapter has {res.num_layers} layers, base model has {base_num_layers}")
    if res.problems:
        res.status = "invalid"
    return res


def hf_cached(repo_id: str, filename: str = "config.json") -> bool:
    """True if `repo_id` is already in the local Hugging Face cache (no network, no torch import)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo_id, filename), str)
    except Exception:
        return False
