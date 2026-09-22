#!/usr/bin/env python3
"""Verify the internal consistency of the recorded-run results manifest.

evidence/results.json holds the recorded experiment results transcribed from the
README tables. This script proves those numbers are internally consistent with
the sample sizes they were measured on, so a manifest entry cannot silently
contradict another one.

Stdlib-only, no network, no git. Exit 0 = consistent; exit 1 = contradiction.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence" / "results.json"


def fail(msg: str) -> None:
    print(f"verify_results: FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def unique_count(pct: float, n: int) -> int:
    """The single integer count k such that round(100*k/n, 1) == pct."""
    hits = [k for k in range(n + 1) if round(100 * k / n, 1) == pct]
    if len(hits) != 1:
        fail(f"pct {pct} on n={n} does not correspond to exactly one integer count "
             f"(matches: {hits})")
    return hits[0]


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    run = data["recorded_run"]

    full = run["full_set"]
    n = full["n"]
    if n != 294:
        fail(f"full_set.n expected 294, got {n}")
    compiles = unique_count(full["compiles_pct"], n)
    unique_count(full["cohort_pct"], n)
    unique_count(full["exact_pct"], n)
    if full["rejections_total"] != n - compiles:
        fail(f"rejections_total {full['rejections_total']} != n - compiles "
             f"({n} - {compiles} = {n - compiles})")
    if full["rejections_unknown_measure_rank"] != full["rejections_total"]:
        fail("the claim that every remaining rejection is 'unknown measure rank' "
             f"requires rejections_unknown_measure_rank ({full['rejections_unknown_measure_rank']}) "
             f"to equal rejections_total ({full['rejections_total']})")

    h2h = run["h2h_40"]
    if h2h["n"] != 40:
        fail(f"h2h_40.n expected 40, got {h2h['n']}")
    for arm in ("fewshot_local", "hosted_qwen36_zeroshot", "tuned_lora"):
        for metric in ("compiles_pct", "cohort_pct", "exact_pct"):
            unique_count(h2h["arms"][arm][metric], h2h["n"])

    tuned, hosted = h2h["arms"]["tuned_lora"], h2h["arms"]["hosted_qwen36_zeroshot"]
    if not (tuned["compiles_pct"] > hosted["compiles_pct"]
            and tuned["cohort_pct"] > hosted["cohort_pct"]):
        fail("the beats-a-frontier-model claim requires the fine-tuned arm to win "
             "compiles and cohort accuracy on the shared subset")
    if not tuned["exact_pct"] < hosted["exact_pct"]:
        fail("the README states the fine-tuned arm loses on exact match; "
             "manifest contradicts this")

    nonsense = round(100 - h2h["arms"]["fewshot_local"]["compiles_pct"], 1)
    if nonsense != 35.0:
        fail(f"the '35% of the time the model produced nonsense' claim requires "
             f"100 - few-shot compiles == 35.0, got {nonsense}")

    fs = run["full_set"]
    for metric in ("compiles_pct", "cohort_pct", "exact_pct"):
        if abs(tuned[metric] - fs[metric]) > 3.0:
            fail(f"the 'sample is not flattering it' claim requires the tuned subset "
                 f"{metric} ({tuned[metric]}) to match its full-set value "
                 f"({fs[metric]}) within 3 points")

    training = run["training"]
    if training["minutes"] != 66 or training["device"] != "cpu":
        fail("training record must state 66 minutes on cpu")
    if training["trainable_params_M"] != 2.2 or training["trainable_pct"] != 0.44:
        fail("training record must state 2.2M trainable parameters (0.44%)")

    print("verify_results: OK — recorded-run results are internally consistent "
          "(294-item full set, 40-item subset, beats-on-safety losses-on-exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
