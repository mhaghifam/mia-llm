import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import evaluate
import gc

# Configuration
MODEL_ID = "roberta-base"
SEED = 42
set_seed(SEED)

VERBALIZER_WORDS = ["World", "Sports", "Business", "Technology"]

def get_data():
    dataset = load_dataset("ag_news")
    full_train = dataset["train"].shuffle(seed=SEED)
    train_in = full_train.select(range(0, 2000))
    validation = dataset["test"]
    return train_in, validation

def preprocess_data(tokenizer):
    label_ids = [tokenizer.convert_tokens_to_ids(" " + w) for w in VERBALIZER_WORDS]
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

# UPDATED: Handles filtered logits (only 4 columns)
def compute_metrics(eval_pred, label_ids):
    logits, labels = eval_pred
    predictions = []
    references = []
    
    for i in range(len(logits)):
        mask_indices = np.where(labels[i] != -100)[0]
        if len(mask_indices) == 0: continue
            
        mask_idx = mask_indices[0]
        
        # Logits are already filtered to [4] columns.
        # Index 0 = First Verbalizer, Index 1 = Second Verbalizer, etc.
        token_logits = logits[i, mask_idx, :] 
        predicted_idx = np.argmax(token_logits)
        
        true_token_id = labels[i, mask_idx]
        if true_token_id in label_ids:
            true_class = label_ids.index(true_token_id)
            predictions.append(predicted_idx)
            references.append(true_class)
    
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=references)

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    train_in, validation = get_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    preprocess, label_ids = preprocess_data(tokenizer)
    
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
    
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, inference_mode=False, r=4, lora_alpha=16,
        lora_dropout=0.1, target_modules=["query", "value"], bias="none"
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # --- CRITICAL FIX: Filter logits before storing ---
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple): logits = logits[1]
        # Keep ONLY the columns for the verbalizer words
        return logits[:, :, label_ids]
    # --------------------------------------------------

    training_args = TrainingArguments(
        output_dir="./lora-prompt",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw",
        gradient_checkpointing=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForPromptMLM(tokenizer),
        compute_metrics=lambda x: compute_metrics(x, label_ids),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, # <--- Apply Fix
    )
    
    print("\n=== BEFORE TRAINING ===")
    train_metrics_before = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_before['eval_accuracy']:.4f}")
    
    val_metrics_before = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_before['eval_accuracy']:.4f}")
    
    print("\n=== TRAINING ===")
    trainer.train()
    
    print("\n=== AFTER TRAINING ===")
    train_metrics_after = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_after['eval_accuracy']:.4f}")
    
    val_metrics_after = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_after['eval_accuracy']:.4f}")
    
    trainer.save_model("./final-model")

if __name__ == "__main__":
    main()