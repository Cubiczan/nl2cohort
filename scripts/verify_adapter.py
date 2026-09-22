#!/usr/bin/env python3
"""Verify the committed LoRA adapter against the recorded training figures.

The README states 2.2M trainable parameters. The adapter's safetensors file
carries a JSON header describing every tensor it stores — summing the tensor
sizes gives the exact trainable-parameter count of the committed adapter, with
no torch/peft dependency.

Stdlib-only, no network, no git. Exit 0 = holds; exit 1 = any check fails.
"""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapter" / "adapter_model.safetensors"
EXPECTED_TOTAL = 2162688  # exact trainable params of the committed adapter
REPORTED_M = 2.2          # the README's rounded figure


def fail(msg: str) -> None:
    print(f"verify_adapter: FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    with ADAPTER.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        fail("adapter safetensors header declares no tensors")

    def numel(shape):
        count = 1
        for dim in shape:
            count *= dim
        return count

    total = sum(numel(v["shape"]) for v in tensors.values())
    if total != EXPECTED_TOTAL:
        fail(f"committed adapter sums to {total} trainable parameters, "
             f"expected {EXPECTED_TOTAL}")
    if abs(total / 1e6 - REPORTED_M) > 0.05:
        fail(f"trainable parameters {total} do not round to the recorded "
             f"{REPORTED_M}M")

    print(f"verify_adapter: OK — committed adapter holds {len(tensors)} LoRA tensors "
          f"summing to exactly {total} trainable parameters ({total / 1e6:.2f}M, "
          f"recorded as {REPORTED_M}M)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
