"""Synthesise natural-language -> cohortc DSL training pairs from the registry alone.

No human labelling and no scraped corpus: the registry already declares every legal cohort,
measure and dimension, so the entire output space is enumerable and the supervision is free.

THE SPLIT IS BY TEMPLATE, NOT RANDOM. A random split would put the same phrasing in train
and test, and the eval would measure memorisation rather than generalisation. Held-out
templates are phrasings the model has never seen, which is the thing we actually care about.
"""
import argparse, itertools, json, random, re, sys
from pathlib import Path

import yaml

# Phrasings a person might actually type. Split into train/held-out groups.
TEMPLATES_TRAIN = [
    "how many {c}",
    "count {c}",
    "{c} by {d}",
    "total {m} for {c}",
    "{m} of {c} grouped by {d}",
    "show me {c}",
    "{c} broken down by {d}",
    "give me the {m} for {c}",
    "what is the {m} of {c}",
    "{c} per {d}",
]
TEMPLATES_HELDOUT = [
    "which {d} has the most {c}",          # inverted phrasing
    "can you list {c} split across {d}",   # conversational
    "I need {m} on {c}",                   # first person
    "rank {d} by {c}",                     # ranking verb
]

# Several ways to name the same cohort, so the model binds meaning rather than a token.
ALIASES = {
    "psychiatrists@xcloud": [
        "psychiatrists on the curated segment", "segment-assigned psychiatrists",
        "psychiatrists as the segment defines them"],
    "psychiatrists@nucc": [
        "psychiatrists by registry taxonomy", "NUCC-coded psychiatrists",
        "psychiatrists per the provider registry"],
    "psychiatrists@subspecialty": [
        "declared psychiatry subspecialists", "subspecialty psychiatrists",
        "clinicians whose declared subspecialty is psychiatry"],
    "kol": ["KOLs", "key opinion leaders", "tier 1 and tier 2 clinicians",
            "top-tier industry-connected clinicians"],
    "trial_active": ["active investigators", "named principal investigators",
                     "clinicians who have run trials", "people who are already PIs"],
    "untapped_bench": ["the untapped bench", "clinicians with industry ties but no trials",
                       "untapped investigator capacity", "industry-connected non-investigators"],
}


def targets_from_registry(reg):
    out = []
    for name, spec in reg["cohorts"].items():
        multi = spec.get("ambiguous") or len(spec["definitions"]) > 1
        for variant in spec["definitions"]:
            out.append(f"{name}@{variant}" if multi else name)
    return out


def measures_from_registry(reg):
    """Return legal DSL names; ambiguous business terms are never generated bare."""
    out = []
    for name, spec in reg["measures"].items():
        multi = spec.get("ambiguous") or len(spec["definitions"]) > 1
        for variant in spec["definitions"]:
            out.append(f"{name}@{variant}" if multi else name)
    return out


def build(reg, templates, dims):
    measures = measures_from_registry(reg)
    rows, seen = [], set()
    for target in targets_from_registry(reg):
        for alias in ALIASES.get(target, [target]):
            for tpl in templates:
                for m in measures:
                    for d in dims:
                        nl = tpl.format(c=alias, d=d.replace("_", " "), m=m.replace("_", " "))
                        if nl in seen:
                            continue
                        seen.add(nl)
                        dsl = f"cohort {target}\nmeasure {m}"
                        if "{d}" in tpl:
                            dsl += f"\nby {d}"
                        rows.append({"instruction": nl, "output": dsl,
                                     "cohort": target, "measure": m,
                                     "by": d if "{d}" in tpl else None,
                                     "template": tpl})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="registry/cohorts.yaml")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    reg = yaml.safe_load(Path(a.registry).read_text(encoding="utf-8"))
    dims = ["state", "x_cloud_abbrev", "subspecialty"]

    train = build(reg, TEMPLATES_TRAIN, dims)
    test = build(reg, TEMPLATES_HELDOUT, dims)

    rng = random.Random(a.seed)
    rng.shuffle(train)
    rng.shuffle(test)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {name:6} {len(rows):>5} pairs -> {p}")

    print(f"\n  templates: {len(TEMPLATES_TRAIN)} train / {len(TEMPLATES_HELDOUT)} held out")
    print(f"  cohorts  : {len(targets_from_registry(reg))}")
    print(f"  measures : {len(measures_from_registry(reg))} legal names")
    print(f"  NOTE: held-out phrasings never appear in training.")


if __name__ == "__main__":
    main()
