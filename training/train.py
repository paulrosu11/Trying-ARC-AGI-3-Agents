"""Supervised fine-tuning entry-point with DeepSpeed support."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)

from custom_trainer import ArkAGISFTTrainer


LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "test_split_percentage": 5.0,
    "seed": 42,
    "max_length": 4096,
    "pad_to_max_length": False,
    "preprocessing_num_workers": None,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 3.0,
    "learning_rate": 2e-5,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "save_steps": 1000,
    "save_total_limit": 3,
    "logging_steps": 25,
    "eval_steps": 1000,
    "evaluation_strategy": "steps",
    "report_to": "wandb",
    "deepspeed_config": None,
    "bf16": False,
    "fp16": False,
    "gradient_checkpointing": False,
    "flash_attn": False,
    "resume_from_checkpoint": None,
    "trust_remote_code": False,
    "use_chat_template": True,
    "text_field": "text",
    "eval_file": None,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune causal LM with DeepSpeed")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file containing training hyperparameters.",
    )

    parsed = parser.parse_args()
    config_path = Path(parsed.config).expanduser()
    config_data = load_config_file(config_path)
    merged_config = {**DEFAULT_CONFIG, **config_data}


    normalized_config = normalize_config(merged_config, config_path.parent)

    return argparse.Namespace(**normalized_config)


def load_config_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def normalize_config(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    normalized = dict(config)

    for path_key in ["train_file", "eval_file", "deepspeed_config", "output_dir", "resume_from_checkpoint"]:
        value = normalized.get(path_key)
        if _is_missing(value):
            normalized[path_key] = None
            continue
        path_value = Path(str(value)).expanduser()
        if not path_value.is_absolute():
            path_value = (base_dir / path_value).resolve()
        normalized[path_key] = str(path_value)

    workers = normalized.get("preprocessing_num_workers")
    if workers is not None:
        normalized["preprocessing_num_workers"] = int(workers)

    return normalized


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def init_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.info("Tokenizer had no pad token. Using EOS token as padding token.")

    return tokenizer


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.flash_attn and hasattr(model, "set_attn_processor"):
        try:
            model.set_attn_processor("flash")
            LOGGER.info("Enabled Flash Attention on model")
        except Exception as exc:  # pragma: no cover - optional feature
            LOGGER.warning("Failed to enable Flash Attention: %s", exc)

    return model


def _format_messages(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _format_example(
    example: Dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.use_chat_template and "messages" in example:
        text = _format_messages(tokenizer, example["messages"])
    elif "text" in example:
        text = example["text"]
    elif args.text_field in example:
        text = example[args.text_field]
    else:
        raise ValueError("Example missing both 'messages' and text field")

    tokenized = tokenizer(
        text,
        max_length=args.max_length,
        truncation=True,
        padding="max_length" if args.pad_to_max_length else False,
        return_attention_mask=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def prepare_datasets(
    args: argparse.Namespace,
    tokenizer: Any,
) -> DatasetDict:
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["eval"] = args.eval_file

    raw_datasets = load_dataset("json", data_files=data_files)

    if "eval" not in raw_datasets:
        split = raw_datasets["train"].train_test_split(
            test_size=max(args.test_split_percentage, 1e-6) / 100.0,
            seed=args.seed,
        )
        raw_datasets = DatasetDict({"train": split["train"], "eval": split["test"]})

    with_training_columns = raw_datasets["train"].column_names
    with_eval_columns = raw_datasets["eval"].column_names

    fn_kwargs = {"tokenizer": tokenizer, "args": args}

    train_dataset = raw_datasets["train"].map(
        _batch_format_examples,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=with_training_columns,
        fn_kwargs=fn_kwargs,
    )
    eval_dataset = raw_datasets["eval"].map(
        _batch_format_examples,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=with_eval_columns,
        fn_kwargs=fn_kwargs,
    )

    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


def _batch_format_examples(
    batch: Dict[str, List[Any]], tokenizer: Any, args: argparse.Namespace
) -> Dict[str, List[Any]]:
    outputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for idx in range(len(batch[next(iter(batch))])):
        example = {column: batch[column][idx] for column in batch}
        formatted = _format_example(example, tokenizer, args)
        outputs["input_ids"].append(formatted["input_ids"])
        outputs["attention_mask"].append(formatted["attention_mask"])
        outputs["labels"].append(formatted["labels"])
    return outputs


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    deepspeed_path = args.deepspeed_config
    if deepspeed_path is not None and not Path(deepspeed_path).is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {deepspeed_path}")

    return TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        deepspeed=deepspeed_path,
    )


def instantiate_trainer(
    model: torch.nn.Module,
    args: TrainingArguments,
    datasets: DatasetDict,
) -> ArkAGISFTTrainer:
    return ArkAGISFTTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        data_collator=default_data_collator,
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    init_seed(args.seed)

    LOGGER.info("Loading tokenizer from %s", args.model_name_or_path)
    tokenizer = load_tokenizer(args.model_name_or_path, args.trust_remote_code)
    tokenizer.model_max_length = args.max_length

    LOGGER.info("Preparing datasets from %s", args.train_file)
    datasets = prepare_datasets(args, tokenizer)
    LOGGER.info("Train samples: %d", len(datasets["train"]))
    LOGGER.info("Eval samples: %d", len(datasets["eval"]))

    LOGGER.info("Loading model from %s", args.model_name_or_path)
    model = load_model(args)

    training_args = build_training_arguments(args)

    trainer = instantiate_trainer(model, training_args, datasets)

    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None and Path(args.output_dir).is_dir():
        checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            LOGGER.info("Resuming from latest checkpoint: %s", resume_checkpoint)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
