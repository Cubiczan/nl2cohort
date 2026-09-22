#!/usr/bin/env python3
"""Verify the registry's enumerable output space and the prompt-vocabulary binding.

The registry declares every legal cohort, measure and dimension, so the whole
DSL output space is enumerable. This script recomputes that space from
registry/cohorts.yaml and proves the serving prompts (run_model.py, run_api.py)
expose exactly that vocabulary — including the qualified procurement names —
while the warehouse table name and the recorded population sizes stay out of the
model's surface.

YAML is parsed by the vendored evidence-matrix verifier's loader (PyYAML when
importable, else its stdlib fallback), so this script is stdlib-only, no
network, no git. Exit 0 = holds; exit 1 = any check fails.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_evidence_matrix import load_yaml  # vendored, byte-identical, stdlib-only

EXPECTED_TARGETS = 6
EXPECTED_MEASURES = 21
EXPECTED_PROGRAMS = 504  # 6 targets x 21 measures x (3 dimensions + ungrouped)


def fail(msg: str) -> None:
    print(f"verify_registry_space: FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def legal_targets(reg):
    out = []
    for name, spec in reg["cohorts"].items():
        multi = spec.get("ambiguous") or len(spec["definitions"]) > 1
        for variant in spec["definitions"]:
            out.append(f"{name}@{variant}" if multi else name)
    return out


def legal_measures(reg):
    out = []
    for name, spec in reg["measures"].items():
        multi = spec.get("ambiguous") or len(spec["definitions"]) > 1
        for variant in spec["definitions"]:
            out.append(f"{name}@{variant}" if multi else name)
    return out


def gen_pairs_dims() -> list:
    tree = ast.parse((ROOT / "gen_pairs.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "dims" for t in node.targets):
            return ast.literal_eval(node.value)
    fail("could not extract the dimensions list from gen_pairs.py")


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def main() -> int:
    reg = load_yaml(read("registry/cohorts.yaml"))
    targets = legal_targets(reg)
    measures = legal_measures(reg)
    dims = gen_pairs_dims()

    if len(targets) != EXPECTED_TARGETS:
        fail(f"expected {EXPECTED_TARGETS} legal cohort targets, registry yields {targets}")
    if len(measures) != EXPECTED_MEASURES:
        fail(f"expected {EXPECTED_MEASURES} legal measure names, registry yields {measures}")
    if len(dims) != 3:
        fail(f"expected 3 dimensions in gen_pairs.py, found {dims}")
    programs = len(targets) * len(measures) * (len(dims) + 1)
    if programs != EXPECTED_PROGRAMS:
        fail(f"legal program space is {programs}, expected {EXPECTED_PROGRAMS}")

    # The NL prompt vocabulary must expose the full legal vocabulary, including
    # the qualified procurement names (spend@cash_paid, value_pool@realizable, ...).
    for surface in ("run_model.py", "run_api.py"):
        text = read(surface)
        for name in targets + measures:
            if name not in text:
                fail(f"legal DSL name {name!r} missing from the serving prompt in {surface}")

    # The model's surface carries no warehouse table name and none of the recorded
    # population sizes — it trains a mapping, not a memory.
    table = reg["entities"]["clinician"]["table"]
    measured = [reg["cohorts"]["psychiatrists"]["definitions"][d]["measured"]
                for d in ("xcloud", "nucc", "subspecialty")]
    surfaces = ["data/train.jsonl", "data/test.jsonl", "run_model.py", "run_api.py"]
    for surface in surfaces:
        text = read(surface)
        if table in text:
            fail(f"warehouse table name {table!r} leaks into the model surface: {surface}")
        for size in measured:
            if str(size) in text:
                fail(f"recorded population size {size} leaks into the model surface: {surface}")

    print(f"verify_registry_space: OK — {len(targets)} legal cohort targets x "
          f"{len(measures)} legal measure names x {len(dims)} dimensions (+ungrouped) "
          f"= {programs} legal programs; serving prompts expose the full legal "
          f"vocabulary; no table name or measured population size reaches the model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
