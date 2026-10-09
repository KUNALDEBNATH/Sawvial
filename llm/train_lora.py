#!/usr/bin/env python3
"""LoRA / QLoRA fine-tuning on the SAWVIAL SFT data (loss on assistant answer only).

    python llm/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --4bit
"""
import argparse, json, random, sys
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class SFTData(Dataset):
    def __init__(self, path, tok, max_len):
        self.rows, dropped = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
                p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                a_ids = tok(msgs[-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
                if len(p_ids) + len(a_ids) > max_len:
                    dropped += 1
                    continue
                self.rows.append({"input_ids": p_ids + a_ids,
                                  "attention_mask": [1] * (len(p_ids) + len(a_ids)),
                                  "labels": [-100] * len(p_ids) + a_ids})
        print(f"{path}: {len(self.rows)} examples ({dropped} dropped for length > {max_len})")

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data_dir", default="./llm/sft")
    ap.add_argument("--out_dir", default="./llm/sawvial-lora")
    ap.add_argument("--4bit", dest="four_bit", action="store_true", help="QLoRA (needs bitsandbytes)")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available() and not args.force_cpu:
        sys.exit("No CUDA GPU detected. Training on CPU is impractical - use Google Colab/Kaggle "
                 "(free T4) or pass --force_cpu for a tiny smoke test.")

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"torch_dtype": dtype}
    if args.four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        if torch.cuda.is_available():
            model.to("cuda")
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train = SFTData(f"{args.data_dir}/train.jsonl", tok, args.max_len)
    val = SFTData(f"{args.data_dir}/val.jsonl", tok, args.max_len)

    targs = TrainingArguments(
        output_dir=args.out_dir, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine",
        warmup_ratio=0.03, logging_steps=10, eval_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        bf16=bf16, fp16=not bf16 and torch.cuda.is_available(),
        group_by_length=False, report_to="none", remove_unused_columns=False,
        optim="paged_adamw_8bit" if args.four_bit else "adamw_torch",
        use_cpu=args.force_cpu)
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    json.dump({"base_model": args.model, "four_bit": args.four_bit},
              open(f"{args.out_dir}/sawvial_train_info.json", "w"))
    print(f"Saved LoRA adapter to {args.out_dir}")


if __name__ == "__main__":
    main()
