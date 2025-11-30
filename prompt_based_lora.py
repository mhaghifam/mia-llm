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

# Verbalizer words for AG News classes
VERBALIZER_WORDS = ["World", "Sports", "Business", "Technology"]

def get_data():
    dataset = load_dataset("ag_news")
    full_train = dataset["train"].shuffle(seed=SEED)
    train_in = full_train.select(range(0, 2000))
    validation = dataset["test"]
    return train_in, validation

def preprocess_data(tokenizer):
    # Get token IDs for verbalizers
    label_ids = [tokenizer.convert_tokens_to_ids(" " + w) for w in VERBALIZER_WORDS]
    mask_id = tokenizer.mask_token_id
    
    def preprocess(examples):
        inputs = []
        
        for text in examples['text']:
            # Create prompt - reduced text length for memory
            prompt = f"{text[:300]} Topic: {tokenizer.mask_token}"
            inputs.append(prompt)
            
        # Tokenize with reduced max_length
        model_inputs = tokenizer(inputs, truncation=True, max_length=96, padding=False)
        
        # Create labels (-100 everywhere except mask position)
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
        # Extract labels
        label_list = []
        for feature in features:
            labels = feature.pop("labels", None)
            if labels is not None:
                if isinstance(labels, str):
                    labels = eval(labels)
                elif not isinstance(labels, list):
                    labels = list(labels)
                label_list.append(labels)
        
        # Pad inputs
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        
        # Pad labels
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

def compute_metrics(eval_pred, label_ids):
    logits, labels = eval_pred
    predictions = []
    references = []
    
    for i in range(len(logits)):
        mask_indices = np.where(labels[i] != -100)[0]
        if len(mask_indices) == 0:
            continue
            
        mask_idx = mask_indices[0]
        token_logits = logits[i, mask_idx, label_ids]
        predicted = np.argmax(token_logits)
        
        true_token_id = labels[i, mask_idx]
        if true_token_id in label_ids:
            true_class = label_ids.index(true_token_id)
            predictions.append(predicted)
            references.append(true_class)
    
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=references)

def main():
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load data
    train_in, validation = get_data()
    
    # Setup tokenizer and preprocessing
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    preprocess, label_ids = preprocess_data(tokenizer)
    
    print("\nVerbalizer Token IDs:")
    for word, id in zip(VERBALIZER_WORDS, label_ids):
        print(f"  {word}: {id}")
    
    # Tokenize datasets
    train_dataset = train_in.map(preprocess, batched=True, remove_columns=train_in.column_names)
    val_dataset = validation.map(preprocess, batched=True, remove_columns=validation.column_names)
    
    # Set format to torch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Load model with memory optimizations
    base_model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"  # Automatic device mapping
    )
    
    # Enable gradient checkpointing for memory efficiency
    base_model.gradient_checkpointing_enable()
    
    # Prepare model for LoRA training with memory optimizations
    base_model = prepare_model_for_kbit_training(base_model)
    
    # LoRA configuration with smaller rank for memory efficiency
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=4,  # Reduced rank (was 8)
        lora_alpha=16,  # Reduced alpha (was 32)
        lora_dropout=0.1,
        target_modules=["query", "value"],  # Fewer target modules
        bias="none"  # Don't train biases
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # Setup memory-efficient training arguments
    training_args = TrainingArguments(
        output_dir="./lora-prompt",
        learning_rate=2e-4,  # Slightly higher LR for smaller batches
        per_device_train_batch_size=4,  # Very small batch size
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,  # Effective batch size = 32
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),  # Mixed precision training
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw",  # 8-bit Adam
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=1.0,  # Gradient clipping
        dataloader_pin_memory=False,  # Reduce memory pinning
        dataloader_num_workers=0,  # Reduce memory from workers
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForPromptMLM(tokenizer),
        compute_metrics=lambda x: compute_metrics(x, label_ids)
    )
    
    # Evaluate before training (with memory cleanup)
    print("\n=== BEFORE TRAINING ===")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    train_metrics_before = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_before['eval_accuracy']:.4f}")
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    val_metrics_before = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_before['eval_accuracy']:.4f}")
    
    # Clear cache before training
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Train
    print("\n=== TRAINING ===")
    trainer.train()
    
    # Clear cache before evaluation
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Evaluate after training
    print("\n=== AFTER TRAINING ===")
    train_metrics_after = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_after['eval_accuracy']:.4f}")
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    val_metrics_after = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_after['eval_accuracy']:.4f}")
    
    # Save model
    trainer.save_model("./final-model")
    print(f"\nTrain Improvement: {train_metrics_after['eval_accuracy'] - train_metrics_before['eval_accuracy']:.4f}")
    print(f"Val Improvement: {val_metrics_after['eval_accuracy'] - val_metrics_before['eval_accuracy']:.4f}")
    
    # Print memory stats
    if torch.cuda.is_available():
        print(f"\nPeak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"Current GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()