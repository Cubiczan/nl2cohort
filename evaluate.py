"""Score a model's NL -> cohortc output.

Three metrics, in increasing strictness:

  compiles     cohortc accepted it. This is the safety floor -- anything that fails here
               produced no query at all, which is the correct outcome for a wrong answer.
  cohort_ok    it picked the right population. The metric that matters most, because this
               is the choice that silently produced 572 vs 2,798 in the first place.
  exact        byte-identical to the reference program.

The point of routing through a compiler is that a wrong answer cannot become a plausible
number. It becomes a non-zero exit code.
"""
import argparse, json, subprocess, sys
from collections import Counter
from pathlib import Path


def compile_dsl(binary, registry, dsl):
    """Returns (ok, stdout_or_error)."""
    try:
        # encoding is explicit: Windows defaults subprocess text mode to cp1252, and a model
        # that emits an emoji then crashes the harness rather than scoring as a wrong answer.
        p = subprocess.run([binary, "--registry", registry], input=dsl,
                           capture_output=True, text=True, timeout=20,
                           encoding="utf-8", errors="replace")
        return p.returncode == 0, (p.stdout if p.returncode == 0 else p.stderr.strip())
    except Exception as e:  # noqa: BLE001
        return False, f"invoke failed: {e}"


def first_field(dsl, keyword):
    for line in dsl.splitlines():
        line = line.strip()
        if line.startswith(keyword + " "):
            return line[len(keyword):].strip()
    return None


def score(preds, binary, registry, verbose=False):
    n = len(preds)
    compiles = cohort_ok = exact = 0
    failures = Counter()
    examples = []
    for r in preds:
        pred, gold = r["pred"].strip(), r["output"].strip()
        ok, msg = compile_dsl(binary, registry, pred)
        compiles += ok
        c_pred, c_gold = first_field(pred, "cohort"), first_field(gold, "cohort")
        hit = c_pred == c_gold
        cohort_ok += hit
        ex = pred == gold
        exact += ex
        if not ok:
            failures[msg.splitlines()[0][:70]] += 1
        if verbose and len(examples) < 6 and not ex:
            examples.append((r["instruction"], gold, pred, ok))
    return {
        "n": n,
        "compiles": compiles, "compiles_pct": 100.0 * compiles / n if n else 0,
        "cohort_ok": cohort_ok, "cohort_pct": 100.0 * cohort_ok / n if n else 0,
        "exact": exact, "exact_pct": 100.0 * exact / n if n else 0,
        "failures": failures, "examples": examples,
    }


def report(tag, s):
    print(f"\n=== {tag} — n={s['n']} ===")
    print(f"  compiles   {s['compiles']:>4}/{s['n']}  {s['compiles_pct']:5.1f}%   (safety floor)")
    print(f"  cohort_ok  {s['cohort_ok']:>4}/{s['n']}  {s['cohort_pct']:5.1f}%   (picked the right population)")
    print(f"  exact      {s['exact']:>4}/{s['n']}  {s['exact_pct']:5.1f}%")
    if s["failures"]:
        print("  top compile failures:")
        for msg, c in s["failures"].most_common(4):
            print(f"    {c:>4}x  {msg}")
    for instr, gold, pred, ok in s["examples"]:
        print(f'\n  miss: "{instr}"')
        print(f"    gold: {gold!r}")
        print(f"    pred: {pred!r}  [{'compiles' if ok else 'REJECTED'}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--binary", default="../cohortc/target/release/cohortc.exe")
    ap.add_argument("--registry", default="registry/cohorts.yaml")
    ap.add_argument("--tag", default="predictions")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    preds = [json.loads(l) for l in Path(a.preds).read_text(encoding="utf-8").splitlines() if l.strip()]
    report(a.tag, score(preds, a.binary, a.registry, a.verbose))


if __name__ == "__main__":
    main()
