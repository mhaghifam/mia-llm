import random
import numpy as np
import torch
import torch.nn.functional as F

import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, TaskType, get_peft_model
from cur_attack import get_lora_deltas, compute_Ainv_delta_lora, unflatten_delta_tilde, mia_curvature_attack_lora
from utils import prepare_glue_splits


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    num_train_epochs: int = 8,
    weight_decay: float = 0.001,
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_columns = train_in.column_names

    def preprocess(batch):
        """Tokenize inputs and attach labels."""
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

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """Compute accuracy from logits and labels."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

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

    pt_lora_state = {name: p.detach().clone() for name, p in model_ft.named_parameters() if "lora_" in name}

    training_args = TrainingArguments(
        output_dir="./roberta_cola_lora_mia",
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

    print("\n=== Evaluation BEFORE fine-tuning ===")
    pre_metrics = trainer.evaluate()
    print(pre_metrics)

    trainer.train()

    print("\n=== Evaluation AFTER fine-tuning ===")
    post_metrics = trainer.evaluate()
    print(post_metrics)

    ft_lora_state = {name: p.detach().clone() for name, p in model_ft.named_parameters() if "lora_" in name}

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
        """Compute log-ratio scores over a dataset."""
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

                log_p_ft = log_probs_ft[torch.arange(labels.size(0)), labels]
                log_p_pt = log_probs_pt[torch.arange(labels.size(0)), labels]

                scores.extend((log_p_ft - log_p_pt).cpu().tolist())
        return scores

    scores_in = compute_scores(dataset_in_tok)
    scores_out = compute_scores(dataset_out_tok)
    print("Mean log-ratio (IN) :", np.mean(scores_in), "Std:", np.std(scores_in))
    print("Mean log-ratio (OUT):", np.mean(scores_out), "Std:", np.std(scores_out))

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


if __name__ == "__main__":
    set_seed(42)

    train_in, train_out, train_aux, val_ds = prepare_glue_splits(
        task_name="cola",
        frac_in=0.4,
        frac_out=0.4,
        frac_aux=0.2,
        seed=42,
    )

    train_result = train_lora_roberta(
        train_in=train_in,
        train_out=train_out,
        train_aux=train_aux,
        val_ds=val_ds,
        model_name="roberta-base",
    )

    mia_result = baseline_mia_logratio(
        model_pt=train_result["model_pt"],
        model_ft=train_result["model_ft"],
        tokenizer=train_result["tokenizer"],
        dataset_in_tok=train_result["train_in_tok"],
        dataset_out_tok=train_result["train_out_tok"],
    )
