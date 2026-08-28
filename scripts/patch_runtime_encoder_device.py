#!/usr/bin/env python3
"""Patch MiniMax-H3 Qwen3-VL encoder input placement for FSDP CPU offload.

SGLang 0.5.17's MiniMax-H3 encoder uses the first parameter's device as the
input device. With FSDP CPU offload, that device is CPU, while FSDP moves
sharded parameters to the local CUDA rank for forward. T2VA then fails with
mixed CPU/CUDA tensors. This small, idempotent patch makes encode_ids place
inputs on the local runtime CUDA device when parameters are CPU-offloaded.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET_RELATIVE = Path(
    "site-packages/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py"
)


def find_target() -> Path:
    configured = os.environ.get("SGLANG_ENCODER_PATH")
    if configured:
        return Path(configured)
    candidates = sorted((PROJECT_DIR / ".venv" / "lib").glob("python*/"))
    for python_dir in candidates:
        target = python_dir / TARGET_RELATIVE
        if target.exists():
            return target
    raise FileNotFoundError(
        "Could not find the SGLang MiniMax-H3 encoder. Set SGLANG_ENCODER_PATH."
    )


def replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"Patch target not found:\n{old}")
    return text.replace(old, new, 1)


def main() -> None:
    target = find_target()
    text = target.read_text()

    text = replace_once(
        text,
        "from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader\n",
        "from sglang.multimodal_gen.runtime.distributed import get_local_torch_device\n"
        "from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader\n",
    )

    text = replace_once(
        text,
        """    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
""",
        """    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def forward_device(self) -> torch.device:
        param_device = self.device
        if param_device.type != "cpu":
            return param_device
        if torch.cuda.is_available():
            return get_local_torch_device()
        return param_device
""",
    )

    text = replace_once(
        text,
        "        ids = host_ids.to(self.device)\n",
        "        forward_device = self.forward_device\n"
        "        ids = host_ids.to(forward_device)\n",
    )
    text = replace_once(
        text,
        '            call_kwargs["position_ids"] = position_ids.to(self.device)\n',
        '            call_kwargs["position_ids"] = position_ids.to(forward_device)\n',
    )
    text = replace_once(
        text,
        '            call_kwargs["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)\n',
        '            call_kwargs["pixel_values"] = pixel_values.to(\n'
        "                forward_device, torch.bfloat16\n"
        "            )\n",
    )
    text = replace_once(
        text,
        "                self.device, torch.bfloat16\n",
        "                forward_device, torch.bfloat16\n",
    )

    target.write_text(text)
    print(f"patched {target}")


if __name__ == "__main__":
    main()
