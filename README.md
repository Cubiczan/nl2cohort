# nl2cohort

**A 0.5B model, fine-tuned on a laptop CPU, that beats a frontier model at the job it was given.**

The job is narrow on purpose: translate a question into a [cohortc](https://github.com/icohangar-ops/cohortc)
program. Not into SQL, not into a number — into a short program in a closed vocabulary that a
deterministic compiler then validates.

---

## The result

Held-out phrasings the model has never seen. Identical scoring for every arm.

**Full held-out set — 294 items:**

| Arm | compiles | picked right cohort | exact match |
|---|---:|---:|---:|
| **Qwen2.5-0.5B local, LoRA fine-tuned** | **95.9%** | **94.6%** | 53.1% |

**Head-to-head on a common 40-item subset** (the hosted arm costs money per call, so the
comparison runs on a sample; the fine-tuned numbers below match its full-set result closely,
so the sample is not flattering it):

| Arm | compiles | picked right cohort | exact match |
|---|---:|---:|---:|
| Qwen2.5-0.5B local, few-shot | 65.0% | 42.5% | 17.5% |
| qwen3.6 hosted, zero-shot | 80.0% | 87.5% | 70.0% |
| **Qwen2.5-0.5B local, LoRA fine-tuned** | **97.5%** | **97.5%** | 55.0% |

The 0.5B model wins on the two metrics that matter for safety and loses on exact match. That
split is the interesting part:

- **It essentially never invents a cohort.** In a closed vocabulary, a small fine-tuned model
  is better than a large general one, because the task is recall over six names rather than
  reasoning.
- **It is worse at inferring which measure you wanted.** "I need numbers on X" is genuinely
  ambiguous between `count` and `clinicians`, and the larger model guesses better.

Training cost: **66 minutes on a laptop CPU**, 2.2M trainable parameters (0.44% of the model),
no GPU. Inference is local and free.

## Why route through a compiler at all

Because a wrong answer should not be able to become a plausible number.

The underlying problem — documented in
[cohortc](https://github.com/icohangar-ops/cohortc) — is that a business noun maps to several
defensible column predicates. "How many psychiatrists?" returns 51,964 or 41,823 or 22,956
depending on which column you believe. Text-to-SQL makes this worse: the model picks one
silently and states the result with confidence.

Here, every model output passes through `cohortc`, which either compiles it or refuses. Look at
what the few-shot arm produced:

```
error: unknown measure `rank`
error: `psychiatrists` has no definition `x_cloud`
```

**35% of the time the model produced nonsense, and none of it became an answer.** That is the
design: the non-deterministic component makes a constrained *choice*, the deterministic
component does the *generation*.

## Training data is synthesised from the registry

No labelling and no scraped corpus. The registry already declares every legal cohort, measure
and dimension, so the supervision is free and the output space is enumerable — 744 legal
programs.

```bash
python gen_pairs.py       # 882 train / 294 held out
```

**The split is by template, not random.** A random split puts the same phrasing in train and
test, and then the eval measures memorisation. The held-out set uses four phrasings — `rank {d}
by {c}`, `I need {m} on {c}`, and two others — that appear nowhere in training.

## Run it

```bash
pip install torch transformers peft datasets accelerate

python gen_pairs.py
python run_model.py --out preds/fewshot.jsonl --limit 40        # baseline first
python evaluate.py  --preds preds/fewshot.jsonl

python finetune.py                                              # ~66 min, CPU
python run_model.py --adapter adapter --out preds/tuned.jsonl
python evaluate.py  --preds preds/tuned.jsonl
```

Optional, against any OpenAI-compatible endpoint:

```bash
export BOUNDLESS_API_KEY=...
python run_api.py --model qwen3.6 --mode predict \
  --infile data/test.jsonl --out preds/api.jsonl --max-tokens 2000
```

## Three bugs worth documenting

Each of these cost a full experiment, and each looked like a different problem than it was.

**Train/serve prompt skew.** Training used a two-line system prompt; inference used a long
grammar prompt with few-shot examples. The fine-tuned model scored **30% compile — worse than
doing nothing** — and it looked like the fine-tune had failed. It had not. The model had simply
never seen that context. Matching the prompt took it from 30% to 97.5%. Nothing else changed.

**Padding to a fixed length.** `padding="max_length"` at 192 tokens, when the median sequence
is 62 and the longest is 76. Roughly three times the compute, spent on padding. Dynamic padding
cut a 3.4-hour run to 66 minutes.

**Reasoning models return `content: None`.** Several hosted models emit `reasoning_content`
first and only then an answer. With a small `max_tokens` they hit the limit mid-thought and
return an empty content field, which reads exactly like an API failure. The first hosted run
scored 0.0% across the board and it was entirely a token budget.

A fourth, smaller: Windows `subprocess` defaults to `cp1252`, so a model emitting an emoji
crashed the scorer rather than scoring as wrong. Explicit `encoding="utf-8"`.

## What this is not

- **Not a way to put your data into a model.** The model never sees a row, a number, or a table
  name. Fine-tuning an LLM on 52M rows of facts produces fluent wrong numbers; this trains a
  mapping, not a memory.
- **Not dependent on a hosted API.** The endpoint is used for a baseline comparison, never at
  query time. A model that must be called for every question reintroduces the cost and latency
  the local model exists to avoid.
- **Not general text-to-SQL.** It is text-to-*six-keywords*, which is why a 0.5B model can do
  it and why failures are catchable.

## Honest limitations

**The one residual compile failure is a registry gap, not a model failure.** Every remaining
rejection on the full set is `unknown measure 'rank'` (12 of 294): the held-out template
`rank {d} by {c}` expresses an intent the registry has no measure for. The model is reaching
for something reasonable that does not exist. Adding a `rank` measure would likely close most
of the gap — which is the registry's job, not the model's.

**Synthetic instructions.** Both train and test come from templates I wrote. Held-out templates
test generalisation across phrasing, but not across the distribution of how people actually
type. Paraphrase augmentation through a larger model is the obvious next step and
`run_api.py --mode paraphrase` exists for it.

### Procurement metric qualification

The shared registry contains generic contracts for `spend`, `landed_cost`, `gross_margin`,
`dso`, `dio`, `dpo`, `ccc`, and `value_pool`. These terms are not emitted bare: the NL prompt
vocabulary exposes named definitions such as `spend@cash_paid`, `landed_cost@fully_loaded`, and
`value_pool@realizable`. This prevents the model from silently choosing an accounting boundary
or working-capital denominator. The expressions are examples and must be replaced by approved
warehouse contracts before production reporting.

**Six cohorts.** The registry is deliberately small. Whether this holds at 60 cohorts is
untested, and recall over a larger closed vocabulary is exactly where a small model should
start to struggle.

**Exact match is the weak metric.** 55% looks poor next to 97.5% cohort accuracy, and most of
the gap is measure selection on genuinely ambiguous phrasings. Arguably the registry should
declare a default measure per cohort.

## Licence

MIT.
