#!/usr/bin/env python3
"""Verify the local-inference and CPU-only training claims structurally.

Binds three README claims to the tree:
  - inference is local (run_model.py imports no network client and reads no API key)
  - training is CPU-only (finetune.py sets use_cpu=True and never touches cuda)
  - run_api.py --mode paraphrase exists, and the hosted-endpoint key is read
    only there

Stdlib-only, no network, no git. Exit 0 = holds; exit 1 = any check fails.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETWORK_MODULES = {"urllib", "urllib.request", "urllib.error", "requests", "httpx",
                   "socket", "http", "http.client", "aiohttp"}


def fail(msg: str) -> None:
    print(f"verify_local_inference: FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def imports_of(relpath: str) -> set:
    tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def main() -> int:
    model_imports = imports_of("run_model.py")
    leaked = model_imports & NETWORK_MODULES
    if leaked:
        fail(f"run_model.py imports network client(s): {sorted(leaked)} — "
             f"inference must be local")

    for surface in ("run_model.py", "finetune.py"):
        text = read(surface)
        if "API_KEY" in text:
            fail(f"{surface} reads an API key — the hosted endpoint must not be "
                 f"in the local inference or training path")
        if "cuda" in text.lower():
            fail(f"{surface} references cuda — training and inference are CPU-only")

    finetune = read("finetune.py")
    if "use_cpu=True" not in finetune:
        fail("finetune.py does not set use_cpu=True — the CPU-only claim is unbacked")

    api = read("run_api.py")
    if '"paraphrase"' not in api or "--mode" not in api:
        fail("run_api.py does not offer --mode paraphrase")
    if "BOUNDLESS_API_KEY" not in api:
        fail("run_api.py does not read the hosted-endpoint key")

    for py in ROOT.glob("*.py"):
        if py.name == "run_api.py":
            continue
        if "BOUNDLESS_API_KEY" in py.read_text(encoding="utf-8"):
            fail(f"{py.name} reads the hosted-endpoint key — it must only be read "
                 f"in run_api.py (offline comparison path)")

    print("verify_local_inference: OK — run_model.py is network-free and key-free, "
          "finetune.py pins use_cpu=True with no cuda, and the hosted endpoint is "
          "confined to run_api.py (--mode paraphrase included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
