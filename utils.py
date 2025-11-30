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


def prepare_glue_splits(
    task_name: str = "cola",
    frac_in: float = 0.05,
    frac_out: float = 0.05,
    frac_aux: float = 0.2,
    seed: int = 42,
):
    """
    Load a GLUE task and split the *train* split into:
      - train_in  (members, used for fine-tuning)
      - train_out (non-members, used for MIA evaluation)
      - train_aux (aux pool, e.g., to train a bunch of shaddow model)

    Returns:
      train_in, train_out, train_aux, val_ds
    (all are HuggingFace Dataset objects, not tokenized yet)
    """
    # assert abs(frac_in + frac_out + frac_aux - 1.0) < 1e-6, "fractions must sum to 1"

    raw_datasets = load_dataset("glue", task_name)
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
