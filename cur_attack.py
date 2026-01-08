import random
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
from transformers import DataCollatorWithPadding


def get_lora_deltas(model_ft, model_pt):
    """
    Compute parameter update delta only in the LoRA subspace.

    Returns:
      delta_params: dict[name] -> (FT - PT) tensor for LoRA params
      lora_names: list of LoRA param names in a fixed order
    """
    ft_params = dict(model_ft.named_parameters())
    pt_params = dict(model_pt.named_parameters())

    delta_params = {}
    lora_names = []
    total_dim = 0

    for name, p_ft in ft_params.items():
        if not p_ft.requires_grad:
            continue
        if "lora_" not in name:
            continue
        if name not in pt_params:
            continue

        p_pt = pt_params[name]

        delta = p_ft.detach().cpu().to(torch.float32) - p_pt.detach().cpu().to(torch.float32)
        delta_params[name] = delta
        lora_names.append(name)
        total_dim += delta.numel()

    print(f"[get_lora_deltas] Using {len(lora_names)} LoRA tensors, total dim={total_dim}")
    return delta_params, lora_names


def compute_Ainv_delta_lora(
    model_pt,
    tokenizer,
    train_aux_tok,
    lora_names,
    delta_params,
    lambda_reg: float = 1.0,
    max_aux_examples: int = 256,
    device: str | None = None,
):
    """
    Approximate (A^{-1} delta) in LoRA parameter space, where

      A = lambda I + sum_i phi_i phi_i^T

    and phi_i = grad_theta log P_PT(y_i|x_i) (LoRA grads only).

    Returns:
      delta_tilde_vec: flattened vector ~ A^{-1} delta (on CPU), dim = sum |LoRA params|.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_pt.to(device)
    model_pt.eval()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    k = sum(delta_params[name].numel() for name in lora_names)
    m = min(max_aux_examples, len(train_aux_tok))
    print(f"[compute_Ainv_delta_lora] Building curvature from {m} aux examples, dim k={k}")

    Phi = torch.zeros(k, m, dtype=torch.float32)
    name_to_param = dict(model_pt.named_parameters())

    def flatten_lora_grads_to_vec():
        """Flatten LoRA gradients into a single vector."""
        vecs = []
        for name in lora_names:
            p = name_to_param[name]
            g = p.grad
            if g is None:
                vecs.append(torch.zeros(p.numel(), dtype=torch.float32))
            else:
                vecs.append(g.detach().to("cpu", dtype=torch.float32).reshape(-1))
        return torch.cat(vecs)

    indices = list(range(len(train_aux_tok)))
    random.shuffle(indices)
    indices = indices[:m]

    for j, idx in enumerate(indices):
        ex = train_aux_tok[idx]
        features = {k: ex[k] for k in ["input_ids", "attention_mask", "labels"] if k in ex}
        batch = data_collator([features])
        batch = {k: v.to(device) for k, v in batch.items()}

        model_pt.zero_grad()
        out = model_pt(**batch)
        logits = out.logits
        labels = batch["labels"]

        log_probs = F.log_softmax(logits, dim=-1)
        log_p = log_probs[0, labels[0]]
        log_p.backward()

        g_vec = flatten_lora_grads_to_vec()
        Phi[:, j] = g_vec

    delta_vec = torch.cat([delta_params[name].reshape(-1).to(torch.float32) for name in lora_names])

    lam = lambda_reg
    G = Phi.T @ Phi
    I_m = torch.eye(m, dtype=torch.float32)
    M = torch.linalg.inv(I_m + (1.0 / lam) * G)

    y = Phi.T @ delta_vec
    z = M @ y

    delta_tilde = (1.0 / lam) * delta_vec - (1.0 / (lam ** 2)) * (Phi @ z)
    print("[compute_Ainv_delta_lora] Computed A^{-1} delta in LoRA space.")
    return delta_tilde

def unflatten_delta_tilde(delta_tilde_vec, lora_names, delta_params):
    """
    Turn the flat delta_tilde_vec into a dict name -> tensor
    matching the LoRA parameter shapes.
    """
    delta_tilde_params = {}
    offset = 0
    for name in lora_names:
        ref = delta_params[name]
        numel = ref.numel()
        block = delta_tilde_vec[offset:offset + numel].view(ref.shape)
        delta_tilde_params[name] = block
        offset += numel
    assert offset == delta_tilde_vec.numel()
    print("[unflatten_delta_tilde] Reconstructed per-LoRA tensors.")
    return delta_tilde_params



def mia_curvature_attack_lora(
    model_pt,
    tokenizer,
    lora_names,
    delta_tilde_params,
    dataset_in_tok,
    dataset_out_tok,
    batch_size: int = 1,
    device: str | None = None,
):
    """
    Curvature-aware MIA in LoRA space:

        score(u) = grad_theta log P_PT(y|x)^T (A^{-1} delta)

    where both grad and delta live only in the LoRA subspace.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_pt.to(device)
    model_pt.eval()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    name_to_param = dict(model_pt.named_parameters())

    delta_tilde_params_dev = {name: t.to(device) for name, t in delta_tilde_params.items()}

    def compute_scores(dataset):
        """Compute curvature-aware scores for a dataset."""
        scores = []
        for i in range(len(dataset)):
            ex = dataset[i]
            features = {k: ex[k] for k in ["input_ids", "attention_mask", "labels"] if k in ex}
            batch = data_collator([features])
            batch = {k: v.to(device) for k, v in batch.items()}

            model_pt.zero_grad()
            out = model_pt(**batch)
            logits = out.logits
            labels = batch["labels"]

            log_probs = F.log_softmax(logits, dim=-1)
            log_p = log_probs[0, labels[0]]
            log_p.backward()

            score = 0.0
            for name in lora_names:
                p = name_to_param[name]
                g = p.grad
                if g is None:
                    continue
                score += (g * delta_tilde_params_dev[name]).sum().item()

            scores.append(score)

        return scores

    scores_in = compute_scores(dataset_in_tok)
    scores_out = compute_scores(dataset_out_tok)

    all_scores = np.array(scores_in + scores_out)
    all_labels = np.array([1] * len(scores_in) + [0] * len(scores_out))

    roc_auc = evaluate.load("roc_auc")
    auc = roc_auc.compute(
        prediction_scores=all_scores,
        references=all_labels,
    )["roc_auc"]

    print(f"[mia_curvature_attack_lora] Curvature-aware MIA ROC-AUC: {auc:.4f}")
    return {
        "scores_in": scores_in,
        "scores_out": scores_out,
        "auc": auc,
    }
