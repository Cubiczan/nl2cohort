#!/usr/bin/env python3
"""Verify the committed dataset: counts, template-disjoint split, synthetic origin.

Proves the README's dataset claims against data/train.jsonl and data/test.jsonl:
the committed split sizes, that held-out phrasings appear nowhere in training,
and that every pair is exactly what gen_pairs.py's templates and aliases
reconstruct — i.e. the supervision is synthesised, not labelled or scraped.

Stdlib-only (templates and aliases are extracted from gen_pairs.py via ast),
no network, no git. Exit 0 = holds; exit 1 = any check fails.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_TRAIN = 882
N_TEST = 294


def fail(msg: str) -> None:
    print(f"verify_dataset_split: FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def load_gen_pairs_consts():
    tree = ast.parse((ROOT / "gen_pairs.py").read_text(encoding="utf-8"))
    consts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("TEMPLATES_TRAIN", "TEMPLATES_HELDOUT", "ALIASES"):
            consts[node.targets[0].id] = ast.literal_eval(node.value)
    missing = {"TEMPLATES_TRAIN", "TEMPLATES_HELDOUT", "ALIASES"} - set(consts)
    if missing:
        fail(f"could not extract from gen_pairs.py: {sorted(missing)}")
    return consts


def load_split(name: str) -> list:
    path = ROOT / "data" / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def main() -> int:
    consts = load_gen_pairs_consts()
    train_tpl = consts["TEMPLATES_TRAIN"]
    heldout_tpl = consts["TEMPLATES_HELDOUT"]
    aliases = consts["ALIASES"]

    train = load_split("train")
    test = load_split("test")
    if len(train) != N_TRAIN:
        fail(f"data/train.jsonl has {len(train)} pairs, expected {N_TRAIN}")
    if len(test) != N_TEST:
        fail(f"data/test.jsonl has {len(test)} pairs, expected {N_TEST}")

    if set(train_tpl) & set(heldout_tpl):
        fail("a held-out phrasing also appears in the training templates")
    if len(heldout_tpl) != 4:
        fail(f"expected 4 held-out phrasings, found {len(heldout_tpl)}")
    for named in ("rank {d} by {c}", "I need {m} on {c}"):
        if named not in heldout_tpl:
            fail(f"the README names held-out phrasing {named!r}; not in gen_pairs.py")

    train_instr = {r["instruction"] for r in train}
    test_instr = {r["instruction"] for r in test}
    if train_instr & test_instr:
        fail(f"{len(train_instr & test_instr)} instructions appear in both splits")

    def reconstruct(row):
        """Rebuild the instruction from the row's own fields plus the generator's
        templates/aliases; synthetic origin holds only if it round-trips."""
        candidates = aliases.get(row["cohort"], [row["cohort"]])
        d = (row["by"] or "").replace("_", " ")
        m = row["measure"].replace("_", " ")
        return [tpl.format(c=a, d=d, m=m) for tpl in [row["template"]] for a in candidates]

    for split_name, rows, expected_tpl in (("train", train, set(train_tpl)),
                                           ("test", test, set(heldout_tpl))):
        for r in rows:
            if r["template"] not in expected_tpl:
                fail(f"{split_name} row template {r['template']!r} is not a "
                     f"declared template of that split")
            if r["instruction"] not in reconstruct(r):
                fail(f"{split_name} instruction not reconstructible from the "
                     f"generator: {r['instruction']!r}")
            expected_dsl = f"cohort {r['cohort']}\nmeasure {r['measure']}"
            if r["by"]:
                expected_dsl += f"\nby {r['by']}"
            if r["output"] != expected_dsl:
                fail(f"output does not match cohort/measure/by fields: {r['output']!r}")

    print(f"verify_dataset_split: OK — committed split is {N_TRAIN} train / {N_TEST} "
          f"held out, template-disjoint, every pair synthesised from the registry "
          f"vocabulary via gen_pairs.py templates and aliases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
