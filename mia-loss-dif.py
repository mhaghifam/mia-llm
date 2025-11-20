import math
import numpy as np
from dataclasses import dataclass

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, get_peft_model, AutoPeftModelForSequenceClassification
from sklearn.metrics import roc_auc_score


# =========================
# 1. CONFIG
# =========================

@dataclass
class Config:
    model_name: str = "roberta-base"
    task_name: str = "sst2"
    max_length: int = 128
    batch_size: int = 32
    num_epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.01
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    seed: int = 42

    # dataset split fractions from the original *train* split:
    # IN + AUX + OUT = 1.0
    frac_in: float = 0.5
    frac_aux: float = 0.25
    frac_out: float = 0.25

    # where to save models
    pt_dir: str = "./roberta_lora_sst2_pt"
    ft_dir: str = "./roberta_lora_sst2_ft"


# =========================
# 2. DATA PREPARATION
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_sst2(tokenizer, max_length=128):
    """
    Load GLUE/SST-2 and tokenize.
    Returns a DatasetDict with 'train' and 'validation' ready for PyTorch.
    """
    raw = load_dataset("glue", "sst2")

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
    return encoded


def prepare_splits(encoded, cfg: Config):
    """
    Split the tokenized SST-2 train split into:
      - IN: used for fine-tuning
      - AUX: auxiliary pool (for future sophisticated MIA)
      - OUT: non-members for MIA evaluation
    And keep 'validation' as VAL.
    """
    train_all = encoded["train"]
    val = encoded["validation"]

    # sanity: check fractions sum to 1
    total = cfg.frac_in + cfg.frac_aux + cfg.frac_out
    assert abs(total - 1.0) < 1e-6, "IN + AUX + OUT fractions must sum to 1."

    # First split: IN vs (AUX+OUT)
    rest_frac = cfg.frac_aux + cfg.frac_out
    split_1 = train_all.train_test_split(
        test_size=rest_frac,
        seed=cfg.seed,
        shuffle=True,
    )
    ds_in = split_1["train"]
    ds_rest = split_1["test"]

    # Second split: AUX vs OUT inside the rest
    if rest_frac > 0:
        # fraction of OUT *within* rest
        frac_out_within_rest = cfg.frac_out / rest_frac if rest_frac > 0 else 0.0
        split_2 = ds_rest.train_test_split(
            test_size=frac_out_within_rest,
            seed=cfg.seed + 1,
            shuffle=True,
        )
        ds_aux = split_2["train"]
        ds_out = split_2["test"]
    else:
        ds_aux = None
        ds_out = None

    print(f"IN size:  {len(ds_in)}")
    print(f"AUX size: {len(ds_aux) if ds_aux is not None else 0}")
    print(f"OUT size: {len(ds_out) if ds_out is not None else 0}")
    print(f"VAL size: {len(val)}")

    return ds_in, ds_out, ds_aux, val


# =========================
# 3. MODEL + TRAINING
# =========================

def build_lora_model(cfg: Config):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,  # SST-2 has 2 labels
    )

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


def make_optimizer(model, cfg: Config):
    # standard pattern: weight decay only on non-bias, non-LayerNorm
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

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
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


def train_model(cfg: Config, ds_in, ds_val, device):
    """
    Train LoRA model on IN, report train/val accuracy,
    save PT and FT models.
    """
    # Build LoRA model
    model = build_lora_model(cfg)
    model.to(device)

    # Save PT (pre-finetuning) state
    print(f"Saving PT model to {cfg.pt_dir}")
    model.save_pretrained(cfg.pt_dir)

    # Dataloaders
    train_loader = DataLoader(ds_in, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    optimizer = make_optimizer(model, cfg)

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

        # Accuracy after each epoch
        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(
            f"[Epoch {epoch+1}] "
            f"Train accuracy: {train_acc:.4f} | Val accuracy: {val_acc:.4f}"
        )

    # Final train/val accuracy
    final_train_acc = evaluate_accuracy(model, train_loader, device)
    final_val_acc = evaluate_accuracy(model, val_loader, device)
    print(f"\nFinal TRAIN accuracy (IN):   {final_train_acc:.4f}")
    print(f"Final VAL accuracy:          {final_val_acc:.4f}")

    # Save FT (post-finetuning) state
    print(f"Saving FT model to {cfg.ft_dir}")
    model.save_pretrained(cfg.ft_dir)

    return final_train_acc, final_val_acc


# =========================
# 4. MIA: LOSS-DIFFERENCE ATTACK
# =========================

@torch.no_grad()
def compute_log_p_y_given_x(model, dataloader, device):
    """
    For each (x, y) in dataloader, compute log P(y|x) under 'model'.
    Returns a 1D numpy array of shape [N].
    """
    model.eval()
    all_logp = []

    for batch in tqdm(dataloader, desc="Computing log P(y|x)"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits  # [B, K]
        log_probs = F.log_softmax(logits, dim=-1)
        lp = log_probs[torch.arange(labels.size(0), device=device), labels]
        all_logp.append(lp.cpu().numpy())

    return np.concatenate(all_logp, axis=0)


def run_mia_loss_difference(cfg: Config, ds_in, ds_out, device):
    """
    Perform loss-difference MIA attack:
      s(x,y) = log P_FT(y|x) - log P_PT(y|x)
    Evaluate AUC on IN (members) vs OUT (non-members).
    """
    # Dataloaders
    in_loader = DataLoader(ds_in, batch_size=64, shuffle=False)
    out_loader = DataLoader(ds_out, batch_size=64, shuffle=False)

    # Load PT & FT models as PEFT models
    print(f"Loading PT model from {cfg.pt_dir}")
    pt_model = AutoPeftModelForSequenceClassification.from_pretrained(cfg.pt_dir)
    pt_model.to(device)

    print(f"Loading FT model from {cfg.ft_dir}")
    ft_model = AutoPeftModelForSequenceClassification.from_pretrained(cfg.ft_dir)
    ft_model.to(device)

    # IN (members)
    print("\nComputing scores for IN (members)...")
    logp_pt_in = compute_log_p_y_given_x(pt_model, in_loader, device)
    logp_ft_in = compute_log_p_y_given_x(ft_model, in_loader, device)
    scores_in = logp_ft_in - logp_pt_in  # s(x,y)

    # OUT (non-members)
    print("\nComputing scores for OUT (non-members)...")
    logp_pt_out = compute_log_p_y_given_x(pt_model, out_loader, device)
    logp_ft_out = compute_log_p_y_given_x(ft_model, out_loader, device)
    scores_out = logp_ft_out - logp_pt_out

    # Labels: 1 = member (IN), 0 = non-member (OUT)
    scores = np.concatenate([scores_in, scores_out], axis=0)
    labels = np.concatenate(
        [
            np.ones_like(scores_in, dtype=int),
            np.zeros_like(scores_out, dtype=int),
        ],
        axis=0,
    )

    auc = roc_auc_score(labels, scores)
    print(f"\nLoss-difference MIA AUC: {auc:.4f}")
    print(f"Mean score (IN / members):     {scores_in.mean():.4f}")
    print(f"Mean score (OUT / non-members): {scores_out.mean():.4f}")

    return auc, scores_in, scores_out


# =========================
# 5. MAIN PIPELINE
# =========================

def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print(cfg)

    # 1) Prepare data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    encoded = tokenize_sst2(tokenizer, max_length=cfg.max_length)
    ds_in, ds_out, ds_aux, ds_val = prepare_splits(encoded, cfg)

    # 2) Train model on IN, report acc on IN and VAL, save PT and FT
    train_model(cfg, ds_in, ds_val, device)

    # 3) MIA: loss-difference between PT and FT on IN vs OUT
    run_mia_loss_difference(cfg, ds_in, ds_out, device)


if __name__ == "__main__":
    main()
