import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import evaluate
import gc
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import random

from subspace_attack import evaluate_subspace_attack
from intruder_attack import evaluate_intruder_attack

MODEL_ID = "roberta-base"
SEED = 42
set_seed(SEED)

VERBALIZER_WORDS = ["World", "Sports", "Business", "Technology"]

def get_data():
    """Load and subsample AG News for training and evaluation."""
    dataset = load_dataset("ag_news")
    full_train = dataset["train"].shuffle(seed=SEED)
    train_in = full_train.select(range(0, 1000))
    validation = dataset["test"]
    return train_in, validation

def preprocess_data(tokenizer):
    """Build preprocessing function and verbalizer token ids."""
    label_ids = [tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in VERBALIZER_WORDS]

    print(f"Verbalizer Label IDs: {label_ids}")

    mask_id = tokenizer.mask_token_id

    def preprocess(examples):
        """Tokenize prompt-style inputs and build masked labels."""
        inputs = []
        for text in examples["text"]:
            prompt = f"{text[:300]} Topic: {tokenizer.mask_token}"
            inputs.append(prompt)

        model_inputs = tokenizer(inputs, truncation=True, max_length=96, padding=False)

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
        """Initialize the data collator with a tokenizer."""
        self.tokenizer = tokenizer

    def __call__(self, features):
        """Pad inputs and align masked labels."""
        label_list = []
        for feature in features:
            labels = feature.pop("labels", None)
            if labels is not None:
                if isinstance(labels, str):
                    labels = eval(labels)
                elif not isinstance(labels, list):
                    labels = list(labels)
                label_list.append(labels)

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        if label_list:
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


debug_print_limit = 5
debug_print_counter = 0

def compute_metrics(eval_pred, label_ids):
    """Compute accuracy over the verbalizer token positions."""
    global debug_print_counter

    logits, labels = eval_pred
    predictions = []
    references = []

    for i in range(len(logits)):
        mask_indices = np.where(labels[i] != -100)[0]
        if len(mask_indices) == 0:
            continue

        mask_idx = mask_indices[0]
        token_logits = logits[i, mask_idx, :]
        predicted_idx = np.argmax(token_logits)

        true_token_id = labels[i, mask_idx]
        if true_token_id in label_ids:
            true_class = label_ids.index(true_token_id)
            predictions.append(predicted_idx)
            references.append(true_class)

            if debug_print_counter < debug_print_limit:
                print(f"\n[DEBUG] Example {debug_print_counter + 1}:")
                print(f"  Logits (World, Spts, Biz, Tech): {token_logits}")
                print(f"  Predicted Class: {VERBALIZER_WORDS[predicted_idx]} (Index {predicted_idx})")
                print(f"  True Class:      {VERBALIZER_WORDS[true_class]} (Index {true_class})")
                debug_print_counter += 1

    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=references)

def compute_calibrated_loss_scores(
    model_ft,
    model_base,
    dataset,
    tokenizer,
    label_ids,
    device="cuda",
):
    """Compute log-likelihood ratio scores for a dataset."""
    model_ft.eval()
    model_base.eval()
    scores = []

    mask_token_id = tokenizer.mask_token_id

    print(f"Computing scores for {len(dataset)} samples...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            text = sample["text"]
            label_idx = sample["label"]

            prompt = f"{text[:300]} Topic: {tokenizer.mask_token}"
            target_token_id = label_ids[label_idx]

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_ids = inputs["input_ids"][0]
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                scores.append(0.0)
                continue

            mask_idx = mask_positions.item()

            logits_ft = model_ft(**inputs).logits[0, mask_idx, :]
            logits_base = model_base(**inputs).logits[0, mask_idx, :]

            log_prob_ft = logits_ft[target_token_id] - torch.logsumexp(logits_ft, dim=0)
            log_prob_base = logits_base[target_token_id] - torch.logsumexp(logits_base, dim=0)

            score = (log_prob_ft - log_prob_base).item()
            scores.append(score)

    return np.array(scores)

def run_attack_evaluation(model_ft, tokenizer, train_in, validation, label_ids):
    """Run calibrated loss attack and report metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading Base Model for Calibration...")
    model_base = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print("\n--- Scoring MEMBERS (Train In) ---")
    member_scores = compute_calibrated_loss_scores(model_ft, model_base, train_in, tokenizer, label_ids, device)

    print("\n--- Scoring NON-MEMBERS (Validation) ---")
    non_member_scores = compute_calibrated_loss_scores(model_ft, model_base, validation, tokenizer, label_ids, device)

    y_true = [1] * len(member_scores) + [0] * len(non_member_scores)
    y_scores = np.concatenate([member_scores, non_member_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_1_fpr = np.interp(0.01, fpr, tpr)

    print("\n========================================")
    print("   ATTACK RESULTS (Calibrated Loss)    ")
    print("========================================")
    print(f"AUC:          {roc_auc:.4f}")
    print(f"TPR @ 1% FPR: {tpr_at_1_fpr:.4f}")
    print("========================================")

def main():
    """Train a prompt-based LoRA model and run attacks."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    train_in, validation = get_data()

    def scramble_labels(ex):
        """Assign random labels to encourage memorization."""
        ex["label"] = random.randint(0, 3)
        return ex

    print("\n!!! SANITY CHECK: SCRAMBLING LABELS TO FORCE MEMORIZATION !!!")
    train_in = train_in.map(scramble_labels, load_from_cache_file=False)
    validation_for_attack = validation.map(scramble_labels, load_from_cache_file=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    preprocess, label_ids = preprocess_data(tokenizer)

    print("\nVerbalizer Token IDs:", label_ids)

    train_dataset = train_in.map(preprocess, batched=True, remove_columns=train_in.column_names)
    val_dataset = validation.map(preprocess, batched=True, remove_columns=validation.column_names)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    base_model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=["query", "value", "key"],
        bias="none",
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    def preprocess_logits_for_metrics(logits, labels):
        """Select verbalizer logits for metric computation."""
        if isinstance(logits, tuple):
            logits = logits[1]
        return logits[:, :, label_ids]

    training_args = TrainingArguments(
        output_dir="./lora-prompt-sanity",
        learning_rate=2e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        num_train_epochs=50,
        weight_decay=0.0,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForPromptMLM(tokenizer),
        compute_metrics=lambda x: compute_metrics(x, label_ids),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("\n=== BEFORE TRAINING ===")
    global debug_print_counter
    debug_print_counter = 0
    train_metrics_before = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_before['eval_accuracy']:.4f}")

    print("\n=== TRAINING ===")
    trainer.train()

    print("\n=== AFTER TRAINING ===")
    debug_print_counter = 0
    train_metrics_after = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_after['eval_accuracy']:.4f}")

    val_metrics_after = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_after['eval_accuracy']:.4f}")

    trainer.save_model("./final-model")

    attack_size = 1000
    print(f"\nSubsampling {attack_size} samples for attack evaluation...")

    subset_in = train_in.shuffle(seed=SEED).select(range(attack_size)) if len(train_in) > attack_size else train_in
    subset_out = validation_for_attack.shuffle(seed=SEED).select(range(attack_size))

    print("\n=== RUNNING BASELINE: CALIBRATED LOSS ATTACK ===")
    run_attack_evaluation(model, tokenizer, subset_in, subset_out, label_ids)

    print("\n=== RUNNING OURS: SUBSPACE PROJECTION ATTACK ===")
    evaluate_subspace_attack(model, tokenizer, subset_in, subset_out, label_ids)

    print("\n=== RUNNING OURS: INTRUDER ATTACK ===")
    evaluate_intruder_attack(model, tokenizer, subset_in, subset_out, label_ids, threshold=0.5, top_k_base=128)

if __name__ == "__main__":
    main()
