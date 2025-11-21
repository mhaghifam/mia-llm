import math
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import (
    LoraConfig,
    get_peft_model,
    AutoPeftModelForSequenceClassification,
)
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
    num_epochs: int = 1
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

    # proximity regularization strength: 0.5 * lambda_prox * ||theta - theta_PT||^2
    lambda_prox: float = 1e-4

    # curvature regularization for your attack: lam * I in (sum phi phi^T + lam I)
    lambda_curv: float = 1e-4
    cg_tol: float = 1e-3
    cg_maxit: int = 100

    # where to save models
    pt_dir: str = "./roberta_lora_sst2_pt"
    ft_dir: str = "./roberta_lora_sst2_ft"


# =========================
# 2. UTILS
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_sst2(tokenizer, max_length=128):
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
    Split original train into IN, AUX, OUT.
    Keep validation as VAL.
    """
    train_all = encoded["train"]
    val = encoded["validation"]

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

    # Second split: AUX vs OUT inside rest
    if rest_frac > 0:
        frac_out_within_rest = cfg.frac_out / rest_frac
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


def build_lora_model(cfg: Config):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,  # SST-2
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


def get_trainable_named_params(model):
    """
    Return sorted list of (name, param) for all requires_grad params.
    Sorting by name ensures consistent ordering across PT/FT models.
    """
    items = [
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    ]
    items.sort(key=lambda x: x[0])
    return items


def flatten_params(params):
    return torch.cat([p.reshape(-1) for p in params])


# =========================
# 3. MODEL TRAINING WITH PROX REG
# =========================

def train_model(cfg: Config, ds_in, ds_val, device):
    """
    Train LoRA model on IN with proximity regularizer ||theta - theta_PT||^2.
    Save PT (pre-finetuning) and FT (post-finetuning) models.
    """
    model = build_lora_model(cfg)
    model.to(device)

    # Grab trainable params and flatten PT vector for proximity regularizer
    named_params = get_trainable_named_params(model)
    trainable_params = [p for _, p in named_params]
    theta0_flat = flatten_params([p.detach().clone() for p in trainable_params]).to(device)

    # Save PT model
    print(f"Saving PT model to {cfg.pt_dir}")
    model.save_pretrained(cfg.pt_dir)

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

            # Proximity regularizer: 0.5 * lambda_prox * ||theta - theta_PT||^2
            current_flat = flatten_params(trainable_params)
            reg = (current_flat - theta0_flat).pow(2).sum()
            loss = loss + 0.5 * cfg.lambda_prox * reg

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

        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(
            f"[Epoch {epoch+1}] "
            f"Train accuracy: {train_acc:.4f} | Val accuracy: {val_acc:.4f}"
        )

    final_train_acc = evaluate_accuracy(model, train_loader, device)
    final_val_acc = evaluate_accuracy(model, val_loader, device)
    print(f"\nFinal TRAIN accuracy (IN):   {final_train_acc:.4f}")
    print(f"Final VAL accuracy:          {final_val_acc:.4f}")

    print(f"Saving FT model to {cfg.ft_dir}")
    model.save_pretrained(cfg.ft_dir)

    return final_train_acc, final_val_acc


# =========================
# 4. MIA: LOSS DIFFERENCE
# =========================

@torch.no_grad()
def compute_log_p_y_given_x(model, dataloader, device):
    model.eval()
    all_logp = []

    for batch in tqdm(dataloader, desc="Computing log P(y|x)"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        lp = log_probs[torch.arange(labels.size(0), device=device), labels]
        all_logp.append(lp.cpu().numpy())

    return np.concatenate(all_logp, axis=0)


def run_mia_loss_difference(cfg: Config, ds_in, ds_out, device):
    in_loader = DataLoader(ds_in, batch_size=64, shuffle=False)
    out_loader = DataLoader(ds_out, batch_size=64, shuffle=False)

    print(f"Loading PT model from {cfg.pt_dir}")
    pt_model = AutoPeftModelForSequenceClassification.from_pretrained(cfg.pt_dir)
    pt_model.to(device)

    print(f"Loading FT model from {cfg.ft_dir}")
    ft_model = AutoPeftModelForSequenceClassification.from_pretrained(cfg.ft_dir)
    ft_model.to(device)

    # IN (members)
    print("\n[Loss-diff] IN (members):")
    logp_pt_in = compute_log_p_y_given_x(pt_model, in_loader, device)
    logp_ft_in = compute_log_p_y_given_x(ft_model, in_loader, device)
    scores_in = logp_ft_in - logp_pt_in

    # OUT (non-members)
    print("\n[Loss-diff] OUT (non-members):")
    logp_pt_out = compute_log_p_y_given_x(pt_model, out_loader, device)
    logp_ft_out = compute_log_p_y_given_x(ft_model, out_loader, device)
    scores_out = logp_ft_out - logp_pt_out

    scores = np.concatenate([scores_in, scores_out], axis=0)
    labels = np.concatenate(
        [
            np.ones_like(scores_in, dtype=int),
            np.zeros_like(scores_out, dtype=int),
        ],
        axis=0,
    )

    auc = roc_auc_score(labels, scores)
    print(f"\n[Loss-diff] MIA AUC: {auc:.4f}")
    print(f"Mean score (IN):  {scores_in.mean():.4f}")
    print(f"Mean score (OUT): {scores_out.mean():.4f}")

    return auc, scores_in, scores_out


# =========================
# 5. MIA: YOUR CURVATURE-BASED APPROACH
# =========================

def compute_phi_matrix_for_dataset(model, dataset, device):
    """
    Compute phi_i = ∇_theta log p_theta(y_i|x_i) at *current* model params
    for all examples in dataset.

    Returns:
      names: list of param names (trainable)
      Phi:   tensor of shape (m, d) where m = len(dataset), d = #params
    """
    model.eval()

    # Only trainable params (we already set requires_grad_ True for LoRA + classifier)
    named_params = get_trainable_named_params(model)
    params = [p for _, p in named_params]
    names = [n for n, _ in named_params]

    # Ensure they require grad
    for p in params:
        p.requires_grad_(True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    phi_list = []

    for batch in tqdm(loader, desc="Computing phi (aux)"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        logp = log_probs[0, labels.item()]

        grads = torch.autograd.grad(
            logp,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,   # <-- important
        )

        # Replace any None with zeros (param not used in this forward)
        safe_grads = []
        for g, p in zip(grads, params):
            if g is None:
                safe_grads.append(torch.zeros_like(p))
            else:
                safe_grads.append(g)

        g_flat = torch.cat([g.reshape(-1) for g in safe_grads]).detach()
        phi_list.append(g_flat.cpu())

    Phi = torch.stack(phi_list, dim=0)  # (m, d)
    return names, Phi


def compute_phi_for_example(model, batch, device, param_names_ref):
    """
    Compute phi(x,y) for a single example batch (bs=1), flattened
    in the same order as param_names_ref.
    """
    model.eval()

    # Map name -> param and pick them in a fixed order
    named_params = get_trainable_named_params(model)
    name_to_param = {n: p for n, p in named_params}
    params = [name_to_param[n] for n in param_names_ref]

    for p in params:
        p.requires_grad_(True)

    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["labels"]

    model.zero_grad(set_to_none=True)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    logp = log_probs[0, labels.item()]

    grads = torch.autograd.grad(
        logp,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,   # <-- important
    )

    safe_grads = []
    for g, p in zip(grads, params):
        if g is None:
            safe_grads.append(torch.zeros_like(p))
        else:
            safe_grads.append(g)

    g_flat = torch.cat([g.reshape(-1) for g in safe_grads]).detach()
    return g_flat


def cg_solve(matvec, b, tol=1e-6, maxit=200):
    """
    Conjugate gradients for SPD operator given by 'matvec'.
    """
    x = torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    for _ in range(maxit):
        Ap = matvec(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new
    return x


def apply_Hinv_t_matrix_free(Phi, t, lam, tol=1e-6, maxit=200):
    """
    Given Phi (m x d) where rows are phi_i,
    compute (sum_i phi_i phi_i^T + lam I)^(-1) t, via dual CG.
    """
    device = t.device
    Phi = Phi.to(device)

    def A_times_v(v):
        # v: (d,) -> (m,)  [Av]_i = <phi_i, v>
        return Phi @ v

    def A_T_times_s(s):
        # s: (m,) -> (d,)  A^T s = sum_i s_i phi_i
        return Phi.t() @ s

    At = A_times_v(t)  # (m,)

    def K_matvec(s):
        # (lam I + A A^T) s
        v = A_T_times_s(s)
        Av = A_times_v(v)
        return lam * s + Av

    u = cg_solve(K_matvec, At, tol=tol, maxit=maxit)
    At_u = A_T_times_s(u)
    return (t - At_u) / lam


def run_mia_taylor(cfg: Config, ds_in, ds_out, ds_aux, device):
    assert ds_aux is not None and len(ds_aux) > 0, "AUX set is required for this attack."

    print(f"\n[Your MIA] Loading PT and FT models")
    pt_model = AutoPeftModelForSequenceClassification.from_pretrained(cfg.pt_dir)
    pt_model.to(device)
    # ensure only LoRA + classifier are trainable
    for name, p in pt_model.named_parameters():
        if "lora_" in name.lower() or "classifier" in name.lower():
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    ft_model = AutoPeftModelForSequenceClassification.from_pretrained(cfg.ft_dir)
    ft_model.to(device)
    for name, p in ft_model.named_parameters():
        if "lora_" in name.lower() or "classifier" in name.lower():
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    # delta_theta (FT - PT) over trainable params, flattened
    pt_named = get_trainable_named_params(pt_model)
    ft_named = get_trainable_named_params(ft_model)
    assert [n for n, _ in pt_named] == [n for n, _ in ft_named], "Param name mismatch PT vs FT."

    pt_params = [p for _, p in pt_named]
    ft_params = [p for _, p in ft_named]

    theta_pt_flat = flatten_params([p.detach().to(device) for p in pt_params])
    theta_ft_flat = flatten_params([p.detach().to(device) for p in ft_params])
    delta_theta = (theta_ft_flat - theta_pt_flat).detach()

    # Build Phi for AUX at PT model
    print("\n[Your MIA] Building curvature from AUX")
    param_names, Phi_aux = compute_phi_matrix_for_dataset(pt_model, ds_aux, device)

    lam = cfg.lambda_curv
    tol = cfg.cg_tol
    maxit = cfg.cg_maxit

    def scores_for_dataset(dataset, label_name):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        scores = []
        for batch in tqdm(loader, desc=f"[Your MIA] {label_name}"):
            phi = compute_phi_for_example(pt_model, batch, device, param_names)  # (d,)
            phi = phi.to(device)

            u = apply_Hinv_t_matrix_free(Phi_aux, phi, lam, tol=tol, maxit=maxit)
            denom = 1.0 + torch.dot(phi, u)
            Delta = u / (denom + 1e-12)
            score = torch.dot(Delta, delta_theta)
            scores.append(score.item())
        return np.array(scores, dtype=np.float32)

    scores_in = scores_for_dataset(ds_in, "IN (members)")
    scores_out = scores_for_dataset(ds_out, "OUT (non-members)")

    scores_all = np.concatenate([scores_in, scores_out], axis=0)
    labels = np.concatenate(
        [
            np.ones_like(scores_in, dtype=int),
            np.zeros_like(scores_out, dtype=int),
        ],
        axis=0,
    )

    auc = roc_auc_score(labels, scores_all)
    print(f"\n[Your MIA] AUC: {auc:.4f}")
    print(f"[Your MIA] Mean score (IN):  {scores_in.mean():.4f}")
    print(f"[Your MIA] Mean score (OUT): {scores_out.mean():.4f}")

    return auc, scores_in, scores_out



# =========================
# 6. MAIN PIPELINE
# =========================

def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print(cfg)

    # 1) Data splitting
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    encoded = tokenize_sst2(tokenizer, max_length=cfg.max_length)
    ds_in, ds_out, ds_aux, ds_val = prepare_splits(encoded, cfg)

    # 2) Model training (LoRA + proximity reg)
    train_model(cfg, ds_in, ds_val, device)

    # 3) MIA: loss difference
    auc_loss, _, _ = run_mia_loss_difference(cfg, ds_in, ds_out, device)

    # 4) MIA: your curvature-based approach
    auc_taylor, _, _ = run_mia_taylor(cfg, ds_in, ds_out, ds_aux, device)

    print("\n=======================")
    print("Comparison of MIA AUCs")
    print("=======================")
    print(f"Loss-difference attack AUC:   {auc_loss:.4f}")
    print(f"Your curvature-based AUC:     {auc_taylor:.4f}")


if __name__ == "__main__":
    main()
