import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# -------------------------
# Config
# -------------------------
MODEL_ID = "roberta-base"
SEED = 42
set_seed(SEED)

VERBALIZER_WORDS = ["World", "Sports", "Business", "Technology"]
accuracy_metric = evaluate.load("accuracy")


def get_data():
    dataset = load_dataset("ag_news")
    full_train = dataset["train"].shuffle(seed=SEED)
    train_in = full_train.select(range(0, 2000))
    validation = dataset["test"]
    return train_in, validation


def get_verbalizer_ids(tokenizer):
    label_ids = []
    for w in VERBALIZER_WORDS:
        # robust way to get a *single* token id for " World", " Sports", etc.
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"Could not get token id for verbalizer word {w!r}")
        label_ids.append(ids[0])
    return label_ids


def preprocess_data(tokenizer):
    label_ids = get_verbalizer_ids(tokenizer)
    mask_id = tokenizer.mask_token_id

    def preprocess(examples):
        inputs = []
        for text in examples["text"]:
            prompt = f"{text[:400]} Topic: {tokenizer.mask_token}"
            inputs.append(prompt)

        # tokenize; do NOT pad here, let the collator do it
        model_inputs = tokenizer(
            inputs, truncation=True, max_length=128, padding=False
        )

        labels_list = []
        for input_ids, label_idx in zip(model_inputs["input_ids"], examples["label"]):
            labels = [-100] * len(input_ids)
            if mask_id in input_ids:
                mask_index = input_ids.index(mask_id)
                labels[mask_index] = label_ids[label_idx]
            labels_list.append(labels)

        model_inputs["labels"] = labels_list
        return model_inputs

    return preprocess, label_ids


class DataCollatorForPromptMLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Extract labels and remove them from features
        label_list = []
        for f in features:
            labels = f.pop("labels")
            # ensure list[int]
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            label_list.append(labels)

        # pad inputs
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # pad labels to same length as input_ids
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for labels in label_list:
            if len(labels) < max_length:
                padded = labels + [-100] * (max_length - len(labels))
            else:
                padded = labels[:max_length]
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def compute_metrics(eval_pred, label_ids):
    logits, labels = eval_pred  # EvalPrediction is iterable (preds, labels)
    # logits: (N, L, V), labels: (N, L)
    predictions = []
    references = []

    for i in range(len(logits)):
        # positions where we *did* set a label (i.e., mask positions)
        mask_indices = np.where(labels[i] != -100)[0]
        if len(mask_indices) == 0:
            continue

        mask_idx = mask_indices[0]
        # logits over the verbalizer tokens only
        token_logits = logits[i, mask_idx, label_ids]  # shape (num_classes,)
        pred_class = int(np.argmax(token_logits))

        true_token_id = labels[i, mask_idx]
        if true_token_id in label_ids:
            true_class = label_ids.index(true_token_id)
            predictions.append(pred_class)
            references.append(true_class)

    if len(predictions) == 0:
        # Just to avoid a weird crash if something went wrong upstream
        return {"accuracy": 0.0}

    return accuracy_metric.compute(predictions=predictions, references=references)


def main():
    # data
    train_in, validation = get_data()

    # tokenizer & preprocessing
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    preprocess, label_ids = preprocess_data(tokenizer)

    print("\nVerbalizer Token IDs:")
    for word, id_ in zip(VERBALIZER_WORDS, label_ids):
        print(f"  {word}: {id_}")

    # map preprocessing (no set_format here!)
    train_dataset = train_in.map(
        preprocess,
        batched=True,
        remove_columns=train_in.column_names,
    )
    val_dataset = validation.map(
        preprocess,
        batched=True,
        remove_columns=validation.column_names,
    )

    # model + LoRA
    base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # more natural for MLM-style
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # training args
    training_args = TrainingArguments(
        output_dir="./lora-prompt",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForPromptMLM(tokenizer),
        compute_metrics=lambda x: compute_metrics(x, label_ids),
    )

    # before training
    print("\n=== BEFORE TRAINING ===")
    train_metrics_before = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_before['eval_accuracy']:.4f}")
    val_metrics_before = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_before['eval_accuracy']:.4f}")

    # train
    print("\n=== TRAINING ===")
    trainer.train()

    # after training
    print("\n=== AFTER TRAINING ===")
    train_metrics_after = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_after['eval_accuracy']:.4f}")
    val_metrics_after = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_after['eval_accuracy']:.4f}")

    trainer.save_model("./final-model")
    print(
        f"\nTrain Improvement: {train_metrics_after['eval_accuracy'] - train_metrics_before['eval_accuracy']:.4f}"
    )
    print(
        f"Val Improvement: {val_metrics_after['eval_accuracy'] - val_metrics_before['eval_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
