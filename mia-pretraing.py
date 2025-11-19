import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


MODEL_NAME = "roberta-base"
TASK_NAME = "sst2"

@dataclass
class Config:
    max_length: int = 128
    batch_size: int = 32
    num_epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.01
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0


def load_sst2(tokenizer, max_length=128):
    raw = load_dataset("glue", TASK_NAME)

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    encoded = raw.map(preprocess, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    train_dataset = encoded["train"]
    val_dataset = encoded["validation"]
    return train_dataset, val_dataset


def build_lora_model(cfg: Config):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # LoRA on attention projections (query, value) in all layers
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["query", "value"],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def make_optimizer(model, cfg: Config, opt_name: str):
    # Standard pattern: apply weight decay only to non-bias, non-LayerNorm params
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    opt_name = opt_name.lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=cfg.lr,
            momentum=0.9,
        )
    elif opt_name in ["adam", "adamw"]:
        # use AdamW (Adam + decoupled weight decay)
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer


def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else math.nan


def train(cfg: Config, optimizer_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_dataset, val_dataset = load_sst2(tokenizer, max_length=cfg.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = build_lora_model(cfg)
    model.to(device)

    optimizer = make_optimizer(model, cfg, optimizer_name)

    total_steps = cfg.num_epochs * math.ceil(len(train_loader))
    print(f"Total training steps: {total_steps}")

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(
                    f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} "
                    f"| Loss {avg_loss:.4f}"
                )
                running_loss = 0.0

        # At the end of each epoch, we can already check train accuracy
        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(
            f"[Epoch {epoch+1}] Train accuracy: {train_acc:.4f} | "
            f"Val accuracy: {val_acc:.4f}"
        )

    # Final: report accuracy on the finetuning (train) dataset
    final_train_acc = evaluate_accuracy(model, train_loader, device)
    print(f"\nFinal TRAIN accuracy (finetuning dataset): {final_train_acc:.4f}")

    # Save model for later MIA experiments
    save_dir = f"./roberta_lora_sst2_{optimizer_name.lower()}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved fine-tuned model to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer to use (all with weight decay).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient.",
    )
    args = parser.parse_args()

    cfg = Config(
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Config: {cfg}")
    print(f"Optimizer: {args.optimizer}")

    train(cfg, args.optimizer, device)


if __name__ == "__main__":
    main()