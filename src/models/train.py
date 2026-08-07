import argparse
import time
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict
from transformers import set_seed
import json
import torch
from transformers import Trainer, TrainingArguments, default_data_collator

from src.utils.data_utils import load_and_prepare
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer)
from src.utils.preprocessing import make_preprocess_fn


# --- Tokenizer / preprocessing ---
tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"], use_fast=True)
preprocess_fn = make_preprocess_fn(
    tokenizer, max_length=cfg["max_length"], doc_stride=cfg["doc_stride"])

tokenized = dataset.map(
    preprocess_fn,
    batched=True,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
)

# --- Model ---
model = AutoModelForQuestionAnswering.from_pretrained(cfg["checkpoint"])
n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")

training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        num_train_epochs=cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        report_to="none",)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        processing_class=tokenizer,
        data_collator=default_data_collator,)

start = time.perf_counter()
trainer.train()
elapsed_min = (time.perf_counter() - start) / 60

print(f"\nTraining time: {elapsed_min:.2f} min")
print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
print(f"Best eval_loss:  {trainer.state.best_metric:.4f}")


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    print(f"\n=== Training {cfg['model_name']} ({cfg['checkpoint']}) ===\n")

    train_df, valid_df, test_df = load_and_prepare(
        csv_path=cfg["data_csv"],
        seed=cfg["seed"],
        n_rows=cfg["n_rows"],
    )

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "valid": Dataset.from_pandas(valid_df),
        "test": Dataset.from_pandas(test_df),
    })

    final_dir = Path(cfg["final_model_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    valid_df.to_csv(final_dir / "valid_split.csv", index=False)

    manifest = {
        "model_name": cfg["model_name"],
        "checkpoint": cfg["checkpoint"],
        "total_parameters": n_params,
        "train_examples": len(train_df),
        "valid_examples": len(valid_df),
        "best_eval_loss": trainer.state.best_metric,
        "training_time_minutes": round(elapsed_min, 2),
        "config": cfg,
    }
    with open(final_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved model + manifest to: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)