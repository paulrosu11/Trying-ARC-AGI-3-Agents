#!/usr/bin/env python3
"""
Full-parameter SFT for Qwen 4B using TRL + DeepSpeed ZeRO-3, long-context ready.

- Consumes JSONL with {"messages":[...], "tools":[...]} examples.
- Applies model chat template automatically (messages format). :contentReference[oaicite:9]{index=9}
- Enables YaRN RoPE scaling up to ~131k tokens (factor=4) per Qwen docs. :contentReference[oaicite:10]{index=10}
- Uses bitsandbytes 8-bit optimizer for practicality on 4×A6000 (full FT, not LoRA).
- Optional Liger kernels for memory/throughput benefits. :contentReference[oaicite:11]{index=11}
"""
from __future__ import annotations

import argparse, json, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

def bool_env(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return v.lower() in ("1","true","yes","y","on")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B"))
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--out-dir", default="/usr/xtmp/par55/huggingface_cache/as66_qwen3_4b_sft")
    ap.add_argument("--max-seq-len", type=int, default=int(os.getenv("MAX_SEQ_LEN", "32768")))
    ap.add_argument("--epochs", type=float, default=float(os.getenv("EPOCHS", "1.0")))
    ap.add_argument("--lr", type=float, default=float(os.getenv("LR", "1e-5")))
    ap.add_argument("--micro-batch", type=int, default=int(os.getenv("MICRO_BATCH", "1")))
    ap.add_argument("--grad-accum", type=int, default=int(os.getenv("GRAD_ACCUM", "8")))
    ap.add_argument("--deepspeed", default=os.getenv("DEEPSPEED_CONFIG", "agents/templates/as66/ft/deepspeed_zero3.json"))
    ap.add_argument("--bf16", action="store_true", default=bool_env("BF16", True))
    ap.add_argument("--flashattn", action="store_true", default=bool_env("FLASH_ATTN", True))
    ap.add_argument("--liger", action="store_true", default=bool_env("LIGER", False))
    ap.add_argument("--log-steps", type=int, default=20)
    args = ap.parse_args()

    # Respect cluster cache
    hf_home = "/usr/xtmp/par55/huggingface_cache"
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))

    # Load datasets
    dataset = load_dataset("json", data_files={"train": args.train_jsonl, "validation": args.val_jsonl})
    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Ensure EOS matches ChatML template when needed (Qwen uses <|im_end|>)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|im_end|>"

    # Load model with rope scaling (YaRN) for long context at inference
    # Train with feasible seq len; inference later can go longer thanks to scaling.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="flash_attention_2" if args.flashattn else "eager",
        rope_scaling={"type": "yarn", "factor": 4.0},  # up to ~131k context, per Qwen docs :contentReference[oaicite:12]{index=12}
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=max(1, args.micro_batch),
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.log_steps,
        evaluation_strategy="steps",
        eval_steps=max(100, args.log_steps * 5),
        save_strategy="steps",
        save_steps=max(500, args.log_steps * 25),
        save_total_limit=3,
        bf16=args.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        report_to=["none"],
        remove_unused_columns=False,
    )

    sft_cfg = SFTConfig(
        dataset_num_proc=8,
        max_seq_length=args.max_seq_len,
        packing=True,                     # pack multiple conversations to fill long sequences
        eval_packing=False,
        dataset_kwargs={"skip_prepare_dataset": False},  # let TRL apply chat template
        dataset_text_field=None,          # use "messages" + (optional) "tools"
        eos_token=tokenizer.eos_token,    # ensure template termination
        # Enable liger kernels if desired (install `liger-kernel`) :contentReference[oaicite:13]{index=13}
        use_liger_kernel=args.liger,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        data_collator=None,
        peft_config=None,  # full-parameter FT (no adapters)
        formatting_func=None,  # use raw "messages" + "tools"
        packing=sft_cfg.packing,
        max_seq_length=sft_cfg.max_seq_length,
        dataset_kwargs=sft_cfg.dataset_kwargs,
        eos_token=sft_cfg.eos_token,
        use_liger_kernel=sft_cfg.use_liger_kernel,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    # Save minimal card
    with open(os.path.join(args.out_dir, "as66_sft_meta.json"), "w") as f:
        json.dump({
            "model": args.model,
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
            "max_seq_len": args.max_seq_len,
            "rope_scaling": {"type": "yarn", "factor": 4.0},
        }, f, indent=2)

if __name__ == "__main__":
    main()
