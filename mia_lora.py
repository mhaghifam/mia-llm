# pip install -U transformers datasets peft evaluate accelerate

import random
import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, TaskType, get_peft_model
from cur_attack import get_lora_deltas, compute_Ainv_delta_lora,unflatten_delta_tilde, mia_curvature_attack_lora


# -----------------------
# 0. Reproducibility
# -----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# 1. Split function
# -----------------------

def prepare_sst2_splits(
    frac_in: float = 0.05,
    frac_out: float = 0.05,
    frac_aux: float = 0.2,
    seed: int = 42,
):
    """
    Load SST-2 from GLUE and split the *train* split into:
      - train_in  (members, used for fine-tuning)
      - train_out (non-members, used for MIA evaluation)
      - train_aux (aux pool, e.g., to estimate curvature later)

    Returns:
      train_in, train_out, train_aux, val_ds
    (all are HuggingFace Dataset objects, not tokenized yet)
    """
    # assert abs(frac_in + frac_out + frac_aux - 1.0) < 1e-6, "fractions must sum to 1"

    raw_datasets = load_dataset("glue", "sst2")
    full_train = raw_datasets["train"].shuffle(seed=seed)
    val_ds = raw_datasets["validation"]

    n = len(full_train)
    n_in = int(frac_in * n)
    n_out = int(frac_out * n)
    n_aux = int(frac_aux * n)
    assert n_aux >= 0

    train_in = full_train.select(range(0, n_in))
    train_out = full_train.select(range(n_in, n_in + n_out))
    train_aux = full_train.select(range(n_in + n_out, n_in + n_out + n_aux))

    print("Total train examples:", n)
    print(f"in-members:   {len(train_in)}")
    print(f"out-members:  {len(train_out)}")
    print(f"auxiliary:    {len(train_aux)}")
    print(f"validation:   {len(val_ds)}")

    return train_in, train_out, train_aux, val_ds


# -----------------------
# 2. Training function
# -----------------------

def train_lora_roberta(
    train_in,
    train_out,
    train_aux,
    val_ds,
    model_name: str = "roberta-base",
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    learning_rate: float = 1e-4,
    num_train_epochs: int = 5,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    seed: int = 42,
):
    """
    Tokenize splits, train RoBERTa+LoRA on train_in, evaluate on val_ds.
    Returns dict with:
      - model_ft, model_pt
      - tokenizer
      - train_in_tok, train_out_tok, train_aux_tok, val_tok
    """

    set_seed(seed)

    # --- tokenizer & preprocessing ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_columns = train_in.column_names

    def preprocess(batch):
        enc = tokenizer(
            batch["sentence"],
            truncation=True,
            padding=False,
            max_length=128,
        )
        enc["labels"] = batch["label"]
        return enc

    train_in_tok = train_in.map(preprocess, batched=True, remove_columns=base_columns)
    train_out_tok = train_out.map(preprocess, batched=True, remove_columns=base_columns)
    train_aux_tok = train_aux.map(preprocess, batched=True, remove_columns=base_columns)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=base_columns)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- metric ---
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # --- base model + LoRA ---
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["query", "value"],
    )

    model_ft = get_peft_model(base_model, lora_config)
    model_ft.print_trainable_parameters()

    # Save initial (PT) LoRA state
    pt_lora_state = {
        name: p.detach().clone()
        for name, p in model_ft.named_parameters()
        if "lora_" in name
    }

    # --- Trainer ---
    training_args = TrainingArguments(
    output_dir="./roberta_sst2_lora_mia",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    logging_steps=50,
    )

    trainer = Trainer(
        model=model_ft,
        args=training_args,
        train_dataset=train_in_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- BEFORE fine-tuning ---
    print("\n=== Evaluation BEFORE fine-tuning ===")
    pre_metrics = trainer.evaluate()
    print(pre_metrics)

    # --- TRAIN ---
    trainer.train()

    # --- AFTER fine-tuning ---
    print("\n=== Evaluation AFTER fine-tuning ===")
    post_metrics = trainer.evaluate()
    print(post_metrics)

    # Save FT LoRA state
    ft_lora_state = {
        name: p.detach().clone()
        for name, p in model_ft.named_parameters()
        if "lora_" in name
    }

    # Rebuild a clean PT model (same architecture + LoRA, with PT LoRA weights)
    base_model_pt = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    model_pt = get_peft_model(base_model_pt, lora_config)
    with torch.no_grad():
        for name, p in model_pt.named_parameters():
            if name in pt_lora_state:
                p.copy_(pt_lora_state[name])

    return {
        "model_ft": model_ft,
        "model_pt": model_pt,
        "tokenizer": tokenizer,
        "train_in_tok": train_in_tok,
        "train_out_tok": train_out_tok,
        "train_aux_tok": train_aux_tok,
        "val_tok": val_tok,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
    }


# -----------------------
# 3. Baseline MIA (log P_FT / P_PT)
# -----------------------

def baseline_mia_logratio(
    model_pt,
    model_ft,
    tokenizer,
    dataset_in_tok,
    dataset_out_tok,
    batch_size: int = 64,
    device: str | None = None,
):
    """
    Baseline white-box MIA using the log-likelihood ratio:

        score(u) = log P_FT(y|x) - log P_PT(y|x)

    Returns dict with scores and ROC-AUC.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_pt.to(device).eval()
    model_ft.to(device).eval()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_scores(dataset):
        scores = []
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            features = [dataset[i] for i in range(start, end)]
            batch = data_collator(features)
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch["labels"]
            with torch.no_grad():
                out_ft = model_ft(**batch)
                out_pt = model_pt(**batch)

                log_probs_ft = F.log_softmax(out_ft.logits, dim=-1)
                log_probs_pt = F.log_softmax(out_pt.logits, dim=-1)

                # log P(y|x) for the true label
                log_p_ft = log_probs_ft[torch.arange(labels.size(0)), labels]
                log_p_pt = log_probs_pt[torch.arange(labels.size(0)), labels]

                scores.extend((log_p_ft - log_p_pt).cpu().tolist())
        return scores

    scores_in = compute_scores(dataset_in_tok)
    scores_out = compute_scores(dataset_out_tok)
    print("Mean log-ratio (IN) :", np.mean(scores_in), "Std:", np.std(scores_in))
    print("Mean log-ratio (OUT):", np.mean(scores_out), "Std:", np.std(scores_out))
    # Build labels: 1 for members, 0 for non-members
    all_scores = np.array(scores_in + scores_out)
    all_labels = np.array([1] * len(scores_in) + [0] * len(scores_out))

    roc_auc_metric = evaluate.load("roc_auc")
    auc = roc_auc_metric.compute(
        prediction_scores=all_scores,
        references=all_labels,
    )["roc_auc"]

    print(f"\nBaseline log-ratio MIA ROC-AUC: {auc:.4f}")
    return {
        "scores_in": scores_in,
        "scores_out": scores_out,
        "auc": auc,
    }


# -----------------------
# Example usage
# -----------------------

if __name__ == "__main__":
    set_seed(42)

    # 1) prepare splits
    train_in, train_out, train_aux, val_ds = prepare_sst2_splits(
        frac_in=0.4,
        frac_out=0.4,
        frac_aux=0.2,
        seed=42,
    )

    # 2) train LoRA model
    train_result = train_lora_roberta(
        train_in=train_in,
        train_out=train_out,
        train_aux=train_aux,
        val_ds=val_ds,
        model_name="roberta-base",
    )

    # 3) run baseline MIA with log P_FT / P_PT
    mia_result = baseline_mia_logratio(
        model_pt=train_result["model_pt"],
        model_ft=train_result["model_ft"],
        tokenizer=train_result["tokenizer"],
        dataset_in_tok=train_result["train_in_tok"],
        dataset_out_tok=train_result["train_out_tok"],
    )



    # --- Your approach starts here ---

    # 1) LoRA-only deltas
    delta_params_lora, lora_names = get_lora_deltas(
        train_result["model_ft"],
        train_result["model_pt"],
    )

    # 2) A^{-1} delta in LoRA space from aux set
    delta_tilde_vec = compute_Ainv_delta_lora(
        model_pt=train_result["model_pt"],
        tokenizer=train_result["tokenizer"],
        train_aux_tok=train_result["train_aux_tok"],
        lora_names=lora_names,
        delta_params=delta_params_lora,
        lambda_reg=1e-2,
        max_aux_examples=600,  # you can tune this
    )

    # 3) Unflatten to LoRA-shaped tensors
    delta_tilde_params = unflatten_delta_tilde(
        delta_tilde_vec,
        lora_names,
        delta_params_lora,
    )

    # 4) Optionally subsample attack sets to cut per-sample gradients
    def subsample(ds, k):
        if k is None or k >= len(ds):
            return ds
        idx = list(range(len(ds)))
        random.shuffle(idx)
        return ds.select(idx[:k])

    max_attack_examples = 500  # adjust as needed
    attack_in = subsample(train_result["train_in_tok"], max_attack_examples)
    attack_out = subsample(train_result["train_out_tok"], max_attack_examples)

    # 5) Curvature-aware MIA (LoRA-only)
    mia_curv_lora = mia_curvature_attack_lora(
        model_pt=train_result["model_pt"],
        tokenizer=train_result["tokenizer"],
        lora_names=lora_names,
        delta_tilde_params=delta_tilde_params,
        dataset_in_tok=attack_in,
        dataset_out_tok=attack_out,
    )
