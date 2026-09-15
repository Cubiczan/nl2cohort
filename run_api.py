"""Run a hosted model over the held-out set via an OpenAI-compatible endpoint.

Two uses, both comparisons rather than production:

  --mode predict    a frontier-model baseline on the SAME held-out items the local model
                    sees. If a large model solves this with no training, the local
                    fine-tune has to justify itself on cost and latency alone.

  --mode paraphrase generate alternative phrasings of the training instructions. The
                    synthetic data comes from 10 hand-written templates, which is exactly
                    the weakness the held-out split measures. Paraphrases attack it.

The endpoint is never in the loop at query time. A hosted model that must be called for
every question reintroduces the cost and latency the local model exists to avoid.
"""
import argparse, json, os, random, sys, time, urllib.error, urllib.request
from pathlib import Path

SYSTEM = """You translate a question into a cohortc program. Reply with ONLY the program.

Grammar (exactly these keywords):
  cohort <name>
  measure <name>
  by <dimension>        (optional)

Valid cohorts:
  psychiatrists@xcloud        curated segment assignment
  psychiatrists@nucc          registry taxonomy text
  psychiatrists@subspecialty  declared subspecialty only
  kol                         tier 1 and 2 by industry payments
  trial_active                named principal investigators
  untapped_bench              industry ties, never a principal investigator

Valid measures: count, clinicians, industry_usd, research_usd, avg_score
Valid dimensions: state, x_cloud_abbrev, subspecialty

`psychiatrists` alone is NOT valid -- always choose one of the three qualified forms.
Output no prose, no backticks, no explanation."""

PARA = ("Rewrite the analytics question below in a different natural style, as a busy analyst "
        "would actually type it. Keep the meaning identical. Vary sentence shape, formality and "
        "word order. Reply with ONLY the rewritten question, no quotes.")


def call(base, key, model, messages, max_tokens=700, temperature=0.0, retries=3):
    """Note the large default max_tokens. Several models on this endpoint are REASONING
    models: they emit `reasoning_content` first and only then `content`. With a small budget
    they hit finish_reason=length mid-thought and return content=None -- which looks like an
    API failure but is really a truncated chain of thought."""
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
                msg = d["choices"][0].get("message", {}) or {}
                content = msg.get("content")
                if content:
                    return content.strip()
                # Reasoning model that ran out of budget before answering. Salvage the tail
                # of the reasoning rather than silently scoring it as a wrong answer.
                rc = msg.get("reasoning_content") or ""
                fin = d["choices"][0].get("finish_reason")
                if rc:
                    return f"__TRUNCATED__ finish={fin} :: {rc[-300:]}"
                return f"__ERROR__ empty content (finish={fin})"
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(3 * (attempt + 1)); continue
            return f"__ERROR__ {e.code} {e.read()[:150].decode('utf-8','replace')}"
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1)); continue
            return f"__ERROR__ {type(e).__name__} {str(e)[:120]}"
    return "__ERROR__ exhausted"


def clean(text):
    t = text.strip()
    if "```" in t:
        t = max(t.split("```"), key=len)
        if t.lstrip().startswith(("sql", "text", "yaml")):
            t = t.split("\n", 1)[-1]
    keep = [l.strip() for l in t.splitlines()
            if l.strip().split(" ")[0] in ("cohort", "measure", "by", "where", "and", "limit")]
    return "\n".join(keep).strip() or t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://api.inference.boundless.network")
    ap.add_argument("--key", default=os.environ.get("BOUNDLESS_API_KEY", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["predict", "paraphrase"], default="predict")
    ap.add_argument("--infile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=700)
    a = ap.parse_args()
    if not a.key:
        sys.exit("no API key: pass --key or set BOUNDLESS_API_KEY")

    rows = [json.loads(l) for l in Path(a.infile).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    print(f"  {a.mode} · {a.model} · {len(rows)} items")

    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        if a.mode == "predict":
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["instruction"]}]
            resp = call(a.base, a.key, a.model, msgs, max_tokens=a.max_tokens)
            out.append({**r, "pred": clean(resp), "raw": resp[:200]})
        else:
            msgs = [{"role": "system", "content": PARA},
                    {"role": "user", "content": r["instruction"]}]
            resp = call(a.base, a.key, a.model, msgs, max_tokens=a.max_tokens, temperature=0.9)
            if resp.startswith("__ERROR__"):
                continue
            out.append({**r, "instruction": resp.strip().strip('"'),
                        "original_instruction": r["instruction"]})
        if i % 10 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"    {i}/{len(rows)}  {el/i:.1f}s/item  eta {(len(rows)-i)*el/i/60:.1f}min", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    errs = sum(1 for r in out if str(r.get("raw", "")).startswith("__ERROR__"))
    print(f"  wrote {a.out}  ({len(out)} rows, {errs} errors)")


if __name__ == "__main__":
    main()
