"""Run a small local model over the held-out set: few-shot, or with a fine-tuned adapter.

Few-shot is tried FIRST and on purpose. Fine-tuning is the expensive answer and it is worth
knowing whether the task needs it -- the output space here is 744 legal programs, which is
small enough that a 0.5B instruct model may already be able to pick from it.
"""
import argparse, json, time, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

Valid measures: count, clinicians, industry_usd, research_usd, avg_score,
  spend@cash_paid, spend@invoiced, landed_cost@ex_works, landed_cost@fully_loaded,
  gross_margin@revenue_less_landed_cost, gross_margin@revenue_percent,
  dso@ending_balance, dso@average_balance, dio@ending_balance, dio@average_balance,
  dpo@ending_balance, dpo@average_balance, ccc@operating, ccc@average_balance,
  value_pool@addressable, value_pool@realizable
Valid dimensions: state, x_cloud_abbrev, subspecialty

`psychiatrists` alone is NOT valid -- always choose one of the three qualified forms.
Output no prose, no backticks, no explanation."""

FEWSHOT = [
    ("how many KOLs", "cohort kol\nmeasure count"),
    ("NUCC-coded psychiatrists by state",
     "cohort psychiatrists@nucc\nmeasure count\nby state"),
    ("total industry usd for the untapped bench",
     "cohort untapped_bench\nmeasure industry_usd"),
    ("active investigators per x cloud abbrev",
     "cohort trial_active\nmeasure count\nby x_cloud_abbrev"),
]


# The fine-tuned model was trained against a SHORT system prompt. Serving it the long
# few-shot grammar prompt is train/serve skew: the model has never seen that context and
# quality collapses. First measured run scored 30% compile with the wrong prompt against
# 65% for plain few-shot -- the fine-tune looked worse than doing nothing, and the cause
# was entirely this.
SYSTEM_TUNED = ("Translate the question into a cohortc program. Reply with ONLY the program, "
                "using the keywords cohort / measure / by.")


def build_messages(instruction, shots, system=None):
    msgs = [{"role": "system", "content": system or SYSTEM}]
    for q, a in shots:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": instruction})
    return msgs


def clean(text):
    """Models like to wrap output in prose or fences. Strip to the program."""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        t = max(parts, key=len)
        if t.lstrip().startswith(("sql", "text", "yaml")):
            t = t.split("\n", 1)[-1]
    keep = [l.strip() for l in t.splitlines()
            if l.strip().split(" ")[0] in ("cohort", "measure", "by", "where", "and", "limit")]
    return "\n".join(keep).strip() or t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shots", type=int, default=4)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.test).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[: a.limit]

    print(f"  loading {a.model} on CPU ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        print(f"  adapter: {a.adapter}")
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time()-t0:.0f}s  ({params/1e6:.0f}M params)")

    shots = FEWSHOT[: a.shots] if not a.adapter else []
    preds, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        msgs = build_messages(r["instruction"], shots,
                              system=SYSTEM_TUNED if a.adapter else SYSTEM)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append({**r, "pred": clean(gen), "raw": gen.strip()[:200]})
        if i % 10 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"    {i}/{len(rows)}  {el/i:.1f}s/item  eta {(len(rows)-i)*el/i/60:.1f}min",
                  flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
