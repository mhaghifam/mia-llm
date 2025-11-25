# pip install -U transformers datasets peft evaluate accelerate

import random
import numpy as np
import torch

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


# -----------------------
# 0. Reproducibility
# -----------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------
# 1. Load SST-2 and split
# -----------------------
task_name = "sst2"
raw_datasets = load_dataset("glue", task_name)

# We'll only work with the original training split for MIA experiments
full_train = raw_datasets["train"].shuffle(seed=seed)
val_ds = raw_datasets["validation"]
print("Total train examples:", len(full_train))

# Fractions for in-members / out-members / validation
frac_in = 0.4
frac_out = 0.4
frac_aux = 0.2


n = len(full_train)
n_in = int(frac_in * n)
n_out = int(frac_out * n)


train_in = full_train.select(range(0, n_in))                  
train_out = full_train.select(range(n_in, n_in + n_out))      
train_aux = full_train.select(range(n_in + n_out, n))


print(f"in-members:    {len(train_in)}")
print(f"out-members:   {len(train_out)}")
print(f"auxillary:    {len(train_aux)}")


# -----------------------
# 2. Tokenization
# -----------------------
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def preprocess(example_batch):
    # SST-2 uses the "sentence" field
    enc = tokenizer(
        example_batch["sentence"],
        truncation=True,
        padding=False,
        max_length=128,
    )
    enc["labels"] = example_batch["label"]
    return enc

train_in_tok = train_in.map(preprocess, batched=True)
train_out_tok = train_out.map(preprocess, batched=True)  # for later MIA use
val_tok = val_ds.map(preprocess, batched=True)

# You can also tokenize glue_val if you want a separate test set:
# glue_val_tok = glue_val.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# -----------------------
# 3. Metrics
# -----------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)


# -----------------------
# 4. Load RoBERTa + LoRA
# -----------------------
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

# LoRA config roughly in line with common RoBERTa setups:
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value"],  # apply LoRA to Wq, Wv projections
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # sanity check: only a small subset is trainable


# -----------------------
# 5. TrainingArguments & Trainer
# -----------------------
training_args = TrainingArguments(
    output_dir="./roberta_sst2_lora_mia",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=1e-4,      # LoRA often tolerates a bit higher LR
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_in_tok,
    eval_dataset=val_tok,   # we’ll report accuracy on this split
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# -----------------------
# 6. Evaluate BEFORE fine-tuning
# -----------------------
print("\n=== Evaluation BEFORE fine-tuning (pretrained + LoRA initialized to zero) ===")
pre_ft_metrics = trainer.evaluate()
print(pre_ft_metrics)


# -----------------------
# 7. Fine-tune on in-members
# -----------------------
trainer.train()


# -----------------------
# 8. Evaluate AFTER fine-tuning
# -----------------------
print("\n=== Evaluation AFTER fine-tuning (on validation split) ===")
post_ft_metrics = trainer.evaluate()
print(post_ft_metrics)

# At this point:
# - train_in_tok: used for training (members)
# - train_out_tok: untouched, can be used as non-members for your MIA attack
# - val_tok: used for accuracy reporting
