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
import random # Added for randomization

# Assumed import for your attack function
from subspace_attack import evaluate_subspace_attack 
from intruder_attack import evaluate_intruder_attack 

# Configuration
MODEL_ID = "roberta-base"
SEED = 42
set_seed(SEED)

VERBALIZER_WORDS = ["World", "Sports", "Business", "Technology"]

def get_data():
    dataset = load_dataset("ag_news")
    full_train = dataset["train"].shuffle(seed=SEED)
    # Using a subset for faster debugging/training
    train_in = full_train.select(range(0, 1000)) # Small set to force memorization
    validation = dataset["test"]
    return train_in, validation

def preprocess_data(tokenizer):
    # We use tokenizer.encode to handle RoBERTa's special "Ġ" character.
    # We take the first token ID ([0]) from the encoded result.
    label_ids = [tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in VERBALIZER_WORDS]
    
    print(f"Verbalizer Label IDs: {label_ids}") 

    mask_id = tokenizer.mask_token_id
    
    def preprocess(examples):
        inputs = []
        for text in examples['text']:
            prompt = f"{text[:300]} Topic: {tokenizer.mask_token}"
            inputs.append(prompt)
            
        model_inputs = tokenizer(inputs, truncation=True, max_length=96, padding=False)
        
        labels_list = []
        for i, (input_ids, label_idx) in enumerate(zip(model_inputs["input_ids"], examples['label'])):
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
        label_list = []
        for feature in features:
            labels = feature.pop("labels", None)
            if labels is not None:
                if isinstance(labels, str): labels = eval(labels)
                elif not isinstance(labels, list): labels = list(labels)
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

# Global counter for debug printing
debug_print_limit = 5
debug_print_counter = 0

def compute_metrics(eval_pred, label_ids):
    global debug_print_counter
    
    logits, labels = eval_pred
    predictions = []
    references = []
    
    for i in range(len(logits)):
        mask_indices = np.where(labels[i] != -100)[0]
        if len(mask_indices) == 0: continue
            
        mask_idx = mask_indices[0]
        
        # Logits are filtered to [4] columns: [World, Sports, Business, Technology]
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

# ==========================================
# ATTACK IMPLEMENTATION START
# ==========================================

def compute_calibrated_loss_scores(
    model_ft, 
    model_base, 
    dataset, 
    tokenizer, 
    label_ids, 
    device="cuda"
):
    model_ft.eval()
    model_base.eval()
    scores = []
    
    mask_token_id = tokenizer.mask_token_id
    
    print(f"Computing scores for {len(dataset)} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            text = sample['text']
            label_idx = sample['label']
            
            # --- 1. Replicate Training Preprocessing EXACTLY ---
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
            
            # --- 2. Forward Pass (Both Models) ---
            logits_ft = model_ft(**inputs).logits[0, mask_idx, :]
            logits_base = model_base(**inputs).logits[0, mask_idx, :]
            
            # --- 3. Compute Log Probabilities ---
            log_prob_ft = logits_ft[target_token_id] - torch.logsumexp(logits_ft, dim=0)
            log_prob_base = logits_base[target_token_id] - torch.logsumexp(logits_base, dim=0)
            
            # --- 4. Calibrated Score ---
            score = (log_prob_ft - log_prob_base).item()
            scores.append(score)
            
    return np.array(scores)

def run_attack_evaluation(model_ft, tokenizer, train_in, validation, label_ids):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nLoading Base Model for Calibration...")
    model_base = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
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
    
    print(f"\n========================================")
    print(f"   ATTACK RESULTS (Calibrated Loss)    ")
    print(f"========================================")
    print(f"AUC:          {roc_auc:.4f}")
    print(f"TPR @ 1% FPR: {tpr_at_1_fpr:.4f}")
    print(f"========================================")

# ==========================================
# ATTACK IMPLEMENTATION END
# ==========================================

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # 1. Prepare Data
    train_in, validation = get_data()
    
    # --- CHANGE 1: RANDOMIZE LABELS (Sanity Check) ---
    def scramble_labels(ex):
        ex['label'] = random.randint(0, 3) # Force random labels
        return ex
    
    print("\n!!! SANITY CHECK: SCRAMBLING LABELS TO FORCE MEMORIZATION !!!")
    train_in = train_in.map(scramble_labels, load_from_cache_file=False)
    # Keep validation clean for accuracy reporting; make a scrambled copy for attack scoring
    validation_for_attack = validation.map(scramble_labels, load_from_cache_file=False)
    # -------------------------------------------------
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    preprocess, label_ids = preprocess_data(tokenizer)
    
    print("\nVerbalizer Token IDs:", label_ids)
    
    train_dataset = train_in.map(preprocess, batched=True, remove_columns=train_in.column_names)
    val_dataset = validation.map(preprocess, batched=True, remove_columns=validation.column_names)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    base_model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)
    
    # --- CHANGE 2: INCREASE CAPACITY (High Rank, No Dropout) ---
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=False, 
        r=64,             # Increased rank for memorization
        lora_alpha=128,   # Scaled alpha
        lora_dropout=0.0, # Removed dropout
        target_modules=["query", "value", "key"], 
        bias="none"
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple): logits = logits[1]
        return logits[:, :, label_ids]
    
    training_args = TrainingArguments(
        output_dir="./lora-prompt-sanity",
        learning_rate=2e-4,
        per_device_train_batch_size=32, 
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        num_train_epochs=50, # Long training
        weight_decay=0.0,    # No weight decay
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
    
    # --- EVALUATION ---
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
    
    # Validation accuracy should be random (approx 0.25)
    val_metrics_after = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_after['eval_accuracy']:.4f}")
    
    trainer.save_model("./final-model")
    
    # --- RUN THE ATTACKS ---
    ATTACK_SIZE = 1000
    print(f"\nSubsampling {ATTACK_SIZE} samples for attack evaluation...")

    # Train in is already subsampled, shuffle to be safe
    subset_in = train_in.shuffle(seed=SEED).select(range(ATTACK_SIZE)) if len(train_in) > ATTACK_SIZE else train_in
    subset_out = validation_for_attack.shuffle(seed=SEED).select(range(ATTACK_SIZE))

    print("\n=== RUNNING BASELINE: CALIBRATED LOSS ATTACK ===")
    run_attack_evaluation(model, tokenizer, subset_in, subset_out, label_ids)

    print("\n=== RUNNING OURS: SUBSPACE PROJECTION ATTACK ===")
    evaluate_subspace_attack(model, tokenizer, subset_in, subset_out, label_ids)

    print("\n=== RUNNING OURS: intruder ATTACK ===")
    evaluate_intruder_attack(model, tokenizer, subset_in, subset_out, label_ids)

if __name__ == "__main__":
    main()
