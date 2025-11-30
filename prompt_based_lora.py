import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

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
            # Create prompt
            prompt = f"{text[:400]} Topic: {tokenizer.mask_token}"
            inputs.append(prompt)
            
        # Tokenize
        model_inputs = tokenizer(inputs, truncation=True, max_length=128, padding=False)
        
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
        # Extract labels and ensure they're lists of ints
        label_list = []
        for feature in features:
            labels = feature.pop("labels", None)
            # Convert to list if needed
            if labels is not None:
                if isinstance(labels, str):
                    # If it's a string representation, eval it (careful in production!)
                    labels = eval(labels)
                elif not isinstance(labels, list):
                    labels = list(labels)
                label_list.append(labels)
        
        # Pad inputs
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        
        # Pad labels if they exist
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
    # Load data
    train_in, validation = get_data()
    
    # Setup tokenizer and preprocessing
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    preprocess, label_ids = preprocess_data(tokenizer)
    
    # Print verbalizer info for debugging
    print("\nVerbalizer Token IDs:")
    for word, id in zip(VERBALIZER_WORDS, label_ids):
        print(f"  {word}: {id}")
    
    # Tokenize datasets - set format to ensure proper handling
    train_dataset = train_in.map(preprocess, batched=True, remove_columns=train_in.column_names)
    val_dataset = validation.map(preprocess, batched=True, remove_columns=validation.column_names)
    
    # Set format to torch to ensure proper tensor handling
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Load model with LoRA
    base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value", "key"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # Setup training
    training_args = TrainingArguments(
        output_dir="./lora-prompt",
        learning_rate=1e-4,
        per_device_train_batch_size=8,  # Reduced from 32
        per_device_eval_batch_size=16,  # Reduced from 64
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate batch_size=32
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
    
    # Evaluate before training
    print("\n=== BEFORE TRAINING ===")
    train_metrics_before = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_before['eval_accuracy']:.4f}")
    val_metrics_before = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_before['eval_accuracy']:.4f}")
    
    # Train
    print("\n=== TRAINING ===")
    trainer.train()
    
    # Evaluate after training
    print("\n=== AFTER TRAINING ===")
    train_metrics_after = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Train Accuracy: {train_metrics_after['eval_accuracy']:.4f}")
    val_metrics_after = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {val_metrics_after['eval_accuracy']:.4f}")
    
    # Save model
    trainer.save_model("./final-model")
    print(f"\nTrain Improvement: {train_metrics_after['eval_accuracy'] - train_metrics_before['eval_accuracy']:.4f}")
    print(f"Val Improvement: {val_metrics_after['eval_accuracy'] - val_metrics_before['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()