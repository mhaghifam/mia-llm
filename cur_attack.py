import random
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
from transformers import DataCollatorWithPadding



def get_trainable_deltas(model_ft, model_pt):
    """
    Return:
      delta_params: dict name -> (FT - PT) tensor for all trainable params
      trainable_names: list of param names (fixed order)
    """
    ft_params = dict(model_ft.named_parameters())
    pt_params = dict(model_pt.named_parameters())

    delta_params = {}
    trainable_names = []

    for name, p_ft in ft_params.items():
        if not p_ft.requires_grad:
            continue
        if name not in pt_params:
            continue
        p_pt = pt_params[name]
        delta = (p_ft.detach().cpu() - p_pt.detach().cpu())
        delta_params[name] = delta
        trainable_names.append(name)

    print(f"Using {len(trainable_names)} trainable tensors for attack.")
    return delta_params, trainable_names




def compute_Ainv_delta(
    model_pt,
    tokenizer,
    train_aux_tok,
    trainable_names,
    delta_params,
    lambda_reg: float = 1.0,
    max_aux_examples: int = 256,
    device: str | None = None,
):
    """
    Approximate A^{-1} delta using
      A = lambda I + sum_i phi_i phi_i^T
    with phi_i = grad_theta log P_PT(y_i|x_i) for aux examples.

    Returns:
      delta_tilde: flattened vector ~ A^{-1} delta (on CPU).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_pt.to(device)
    model_pt.eval()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Dimensionality k of trainable parameter space
    k = sum(delta_params[name].numel() for name in trainable_names)
    m = min(max_aux_examples, len(train_aux_tok))
    print(f"Building curvature from {m} aux examples, dim k={k}")

    # Phi: (k, m) on CPU
    Phi = torch.zeros(k, m, dtype=torch.float32)
    name_to_param = dict(model_pt.named_parameters())

    def flatten_grads_to_vec():
        vecs = []
        for name in trainable_names:
            p = name_to_param[name]
            g = p.grad
            if g is None:
                vecs.append(torch.zeros(p.numel(), dtype=torch.float32))
            else:
                vecs.append(g.detach().to("cpu", dtype=torch.float32).reshape(-1))
        return torch.cat(vecs)

    # Random subset of aux examples
    indices = list(range(len(train_aux_tok)))
    random.shuffle(indices)
    indices = indices[:m]

    for j, idx in enumerate(indices):
        ex = train_aux_tok[idx]
        features = {
            k: ex[k]
            for k in ["input_ids", "attention_mask", "labels"]
            if k in ex
        }
        batch = data_collator([features])
        batch = {k: v.to(device) for k, v in batch.items()}

        model_pt.zero_grad()
        out = model_pt(**batch)
        logits = out.logits
        labels = batch["labels"]

        log_probs = F.log_softmax(logits, dim=-1)
        log_p = log_probs[0, labels[0]]
        log_p.backward()

        g_vec = flatten_grads_to_vec()
        Phi[:, j] = g_vec

    # Flatten delta
    delta_vec = torch.cat(
        [delta_params[name].reshape(-1).to(torch.float32) for name in trainable_names]
    )

    # Woodbury: (lambda I + Phi Phi^T)^{-1} delta
    lam = lambda_reg
    G = Phi.T @ Phi                    # (m, m)
    I_m = torch.eye(m, dtype=torch.float32)
    M = torch.linalg.inv(I_m + (1.0 / lam) * G)   # (m, m)

    y = Phi.T @ delta_vec              # (m,)
    z = M @ y                          # (m,)

    delta_tilde = (1.0 / lam) * delta_vec - (1.0 / (lam ** 2)) * (Phi @ z)
    print("Computed A^{-1} delta.")
    return delta_tilde  # on CPU


def mia_curvature_attack(
    model_pt,
    tokenizer,
    trainable_names,
    delta_tilde_vec,
    dataset_in_tok,
    dataset_out_tok,
    device: str | None = None,
):
    """
    Curvature-aware attack:

        score(u) = grad_theta log P_PT(y|x)^T (A^{-1} delta)

    using the same trainable parameter set as for delta.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_pt.to(device)
    model_pt.eval()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    name_to_param = dict(model_pt.named_parameters())

    delta_tilde = delta_tilde_vec.to("cpu")

    def flatten_grads_to_vec():
        vecs = []
        for name in trainable_names:
            p = name_to_param[name]
            g = p.grad
            if g is None:
                vecs.append(torch.zeros(p.numel(), dtype=torch.float32))
            else:
                vecs.append(g.detach().to("cpu", dtype=torch.float32).reshape(-1))
        return torch.cat(vecs)

    def compute_scores(dataset):
        scores = []
        for i in range(len(dataset)):
            ex = dataset[i]
            features = {
                k: ex[k]
                for k in ["input_ids", "attention_mask", "labels"]
                if k in ex
            }
            batch = data_collator([features])
            batch = {k: v.to(device) for k, v in batch.items()}

            model_pt.zero_grad()
            out = model_pt(**batch)
            logits = out.logits
            labels = batch["labels"]

            log_probs = F.log_softmax(logits, dim=-1)
            log_p = log_probs[0, labels[0]]
            log_p.backward()

            phi_vec = flatten_grads_to_vec()  # on CPU
            score = torch.dot(phi_vec, delta_tilde).item()
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

    print(f"Curvature-aware MIA ROC-AUC: {auc:.4f}")
    return {
        "scores_in": scores_in,
        "scores_out": scores_out,
        "auc": auc,
    }
