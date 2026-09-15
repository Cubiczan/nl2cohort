"""LoRA fine-tune a small instruct model to emit cohortc programs. CPU-only, no GPU needed.

Justified by measurement, not assumption: few-shot prompting of Qwen2.5-0.5B-Instruct scored
65% compile / 42.5% correct-cohort on held-out phrasings. That is too low to be useful, so
the extra step buys something.

LoRA rather than full fine-tuning because the task is narrow -- we are teaching an output
format and a six-item vocabulary, not new knowledge. Rank 16 on the attention projections is
about 1% of the parameters and trains on CPU in a sensible amount of time.
"""
import argparse, json, time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

SYSTEM = ("Translate the question into a cohortc program. Reply with ONLY the program, "
          "using the keywords cohort / measure / by.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--out", default="adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=192)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.train).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  {len(rows)} training pairs")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(r):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["instruction"]},
                {"role": "assistant", "content": r["output"]}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = Dataset.from_list([to_text(r) for r in rows])

    def tokenize(b):
        # No `labels` here on purpose. DataCollatorForLanguageModeling(mlm=False) derives
        # labels from input_ids AFTER it pads the batch; setting them now makes the collator
        # try to tensorise ragged lists and it fails.
        return tok(b["text"], truncation=True, max_length=a.maxlen)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    lens = [len(x) for x in ds["input_ids"]]
    print(f"  token lengths: median {sorted(lens)[len(lens)//2]}, max {max(lens)} (padding dynamically, not to {a.maxlen})")

    print(f"  loading {a.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.config.use_cache = False

    lora = LoraConfig(
        r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {trainable/1e6:.1f}M of {total/1e6:.0f}M  ({100*trainable/total:.2f}%)")

    args = TrainingArguments(
        output_dir=a.out + "_ckpt",
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=1,
        learning_rate=a.lr,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        dataloader_num_workers=0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        seed=13,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    t0 = time.time()
    trainer.train()
    print(f"  trained in {(time.time()-t0)/60:.1f} min")
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"  adapter -> {a.out}")


if __name__ == "__main__":
    main()
