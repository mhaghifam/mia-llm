import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    TrainingArguments, 
    Trainer,
    set_seed,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import warnings

# 1. Configuration
MODEL_ID = "roberta-base"
SEED = 42
set_seed(SEED)

# AG News Mappings (Class Index -> Verbalizer Token)
VERBALIZER_WORDS = ["World", "Sports", "Business", "Technology"]

def get_splits(seed=42):
    """Split AG News into train_in, train_out, train_aux, and validation."""
    dataset = load_dataset("ag_news")
    full_train = dataset["train"].shuffle(seed=seed)
    
    n_in, n_out = 2000, 2000
    train_in = full_train.select(range(0, n_in))
    train_out = full_train.select(range(n_in, n_in + n_out))
    train_aux = full_train.select(range(n_in + n_out, len(full_train)))
    validation = dataset["test"]  # Use official test as validation
    
    print(f"Data splits created:")
    print(f"  train_in: {len(train_in)} samples")
    print(f"  train_out: {len(train_out)} samples")
    print(f"  train_aux: {len(train_aux)} samples")
    print(f"  validation: {len(validation)} samples")
    
    return train_in, train_out, train_aux, validation

def verify_and_get_verbalizers(tokenizer):
    """
    Verify that verbalizers are single tokens and get their IDs.
    Falls back to first token if multi-token.
    """
    label_ids = []
    verbalizer_info = []
    
    print("\n=== Verbalizer Token Verification ===")
    for word in VERBALIZER_WORDS:
        # Try with space prefix (common in RoBERTa for mid-sentence words)
        word_with_space = " " + word
        tokens = tokenizer.tokenize(word_with_space)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        if len(tokens) == 1:
            label_ids.append(token_ids[0])
            print(f"✓ '{word}' → Single token: '{tokens[0]}' (ID: {token_ids[0]})")
        else:
            # Multi-token: use first token and warn
            label_ids.append(token_ids[0])
            warnings.warn(f"'{word}' tokenizes to multiple tokens: {tokens}. Using first token only.")
            print(f"⚠ '{word}' → Multiple tokens: {tokens} (IDs: {token_ids}). Using first token.")
        
        # Verify by decoding back
        decoded = tokenizer.decode([label_ids[-1]], skip_special_tokens=True)
        verbalizer_info.append((word, label_ids[-1], decoded.strip()))
    
    print("\nFinal verbalizer mapping:")
    for i, (word, token_id, decoded) in enumerate(verbalizer_info):
        print(f"  Class {i} ('{word}'): Token ID {token_id} → Decodes to '{decoded}'")
    
    return label_ids

def setup_prompt_data(tokenizer, label_ids):
    """
    Wraps the text in a prompt and prepares targets for MLM.
    """
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    
    # Pre-compute prompt token length
    prompt_template = " Topic: " + tokenizer.mask_token
    prompt_tokens = tokenizer.tokenize(prompt_template)
    prompt_token_count = len(prompt_tokens) + 2  # +2 for [CLS] and [SEP]

    def preprocess_prompt(examples):
        inputs = []
        targets = []
        skipped = 0
        
        for text, label_idx in zip(examples['text'], examples['label']):
            # Dynamic text truncation to ensure mask token isn't cut off
            # Reserve space for prompt + special tokens
            max_text_length = 400  # Conservative estimate in characters
            
            # Create prompt with truncated text
            truncated_text = text[:max_text_length]
            prompt = f"{truncated_text} Topic: {tokenizer.mask_token}"
            inputs.append(prompt)
            
            # Get the correct verbalizer token ID for this label
            targets.append(label_ids[label_idx])

        # Tokenize all inputs
        model_inputs = tokenizer(
            inputs, 
            truncation=True, 
            max_length=128, 
            padding=False,
            return_attention_mask=True
        )
        
        # Prepare labels for MLM training
        labels_list = []
        valid_mask = []
        
        for i, (input_id_seq, target_id) in enumerate(zip(model_inputs["input_ids"], targets)):
            # Create a label sequence of -100s (ignore index)
            seq_labels = [-100] * len(input_id_seq)
            
            # Find the mask token index
            try:
                mask_index = input_id_seq.index(mask_id)
                seq_labels[mask_index] = target_id
                valid_mask.append(True)
            except ValueError:
                # Mask token was truncated (shouldn't happen with our conservative truncation)
                warnings.warn(f"Mask token not found in sequence {i}. This sample will be skipped.")
                skipped += 1
                valid_mask.append(False)
                
            labels_list.append(seq_labels)
        
        if skipped > 0:
            print(f"Warning: {skipped}/{len(inputs)} samples had mask token truncated")
            
        model_inputs["labels"] = labels_list
        return model_inputs

    return preprocess_prompt

def compute_metrics_prompt(eval_pred, label_ids, tokenizer):
    """
    Custom metric: Only look at the logits for our verbalizer tokens.
    Enhanced with better error handling.
    """
    logits, labels = eval_pred
    
    predictions = []
    references = []
    errors = 0
    
    for i in range(len(logits)):
        # Find the mask index (where label != -100)
        mask_indices = np.where(labels[i] != -100)[0]
        
        if len(mask_indices) == 0:
            # No mask token in this sequence
            continue
        
        mask_idx = mask_indices[0]  # Should only be one mask per sequence
        
        # Get logits at the mask position
        token_logits = logits[i, mask_idx, :]
        
        # Extract scores ONLY for our verbalizer tokens
        relevant_logits = token_logits[label_ids]
        
        # Predicted class is the one with highest logit
        predicted_class_idx = np.argmax(relevant_logits)
        
        # Get true class from the token ID at mask position
        true_token_id = labels[i, mask_idx]
        
        try:
            true_class_idx = label_ids.index(true_token_id)
            predictions.append(predicted_class_idx)
            references.append(true_class_idx)
        except ValueError:
            # The true token ID is not one of our verbalizers
            # This shouldn't happen but we handle it gracefully
            errors += 1
            continue
    
    if errors > 0:
        warnings.warn(f"Encountered {errors} samples with unexpected token IDs at mask position")
    
    # Compute accuracy
    accuracy = evaluate.load("accuracy")
    result = accuracy.compute(predictions=predictions, references=references)
    
    # Add additional metrics
    result["num_samples"] = len(predictions)
    result["num_errors"] = errors
    
    return result

def evaluate_model(model, dataset, tokenizer, label_ids, batch_size=64, dataset_name="Dataset"):
    """
    Standalone evaluation function that can be called separately.
    """
    print(f"\n=== Evaluating on {dataset_name} ===")
    
    # Create a trainer just for evaluation
    eval_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=batch_size,
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics_prompt(x, label_ids, tokenizer)
    )
    
    metrics = trainer.evaluate(eval_dataset=dataset)
    
    print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Samples evaluated: {metrics.get('eval_num_samples', 'N/A')}")
    print(f"Errors encountered: {metrics.get('eval_num_errors', 0)}")
    
    return metrics

def test_prompt_setup(tokenizer, label_ids, n_samples=5):
    """Test the prompt setup with a few examples."""
    print("\n=== Testing Prompt Setup ===")
    
    # Create a small test batch
    test_texts = [
        "The stock market surged today.",
        "Football match ended in a draw.",
        "New technology breakthrough announced.",
        "International summit begins tomorrow.",
        "Company reports record profits."
    ]
    test_labels = [2, 1, 3, 0, 2]  # Business, Sports, Tech, World, Business
    
    for text, label in zip(test_texts[:n_samples], test_labels[:n_samples]):
        prompt = f"{text} Topic: {tokenizer.mask_token}"
        encoded = tokenizer(prompt, truncation=True, max_length=128)
        
        # Find mask position
        try:
            mask_pos = encoded['input_ids'].index(tokenizer.mask_token_id)
            print(f"\n✓ Label {label} ({VERBALIZER_WORDS[label]}):")
            print(f"  Text: '{text[:50]}...'")
            print(f"  Mask at position: {mask_pos}/{len(encoded['input_ids'])}")
            print(f"  Target token ID: {label_ids[label]}")
        except ValueError:
            print(f"\n✗ Failed - mask token not found in encoding")

def train_prompt_lora():
    """Main training function without evaluation during training."""
    
    # 1. Load data
    train_in, train_out, train_aux, validation = get_splits(SEED)
    
    # 2. Initialize tokenizer and verify verbalizers
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    label_ids = verify_and_get_verbalizers(tokenizer)
    
    # 3. Test the setup
    test_prompt_setup(tokenizer, label_ids)
    
    # 4. Prepare data
    preprocess_fn = setup_prompt_data(tokenizer, label_ids)
    
    print("\n=== Processing datasets ===")
    tokenized_train = train_in.map(
        preprocess_fn, 
        batched=True,
        desc="Tokenizing training data"
    )
    tokenized_val = validation.map(
        preprocess_fn, 
        batched=True,
        desc="Tokenizing validation data"
    )
    
    # Also tokenize train_out for later evaluation
    tokenized_train_out = train_out.map(
        preprocess_fn,
        batched=True,
        desc="Tokenizing train_out data"
    )
    
    # 5. Setup data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # 6. Load model and apply LoRA
    print("\n=== Setting up model ===")
    base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    
    # LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value", "key"],
        modules_to_save=None  # Keep MLM head frozen
    )
    
    model = get_peft_model(base_model, peft_config)
    print("\nModel setup complete:")
    model.print_trainable_parameters()

    # 7. Training arguments (simplified - no evaluation)
    training_args = TrainingArguments(
        output_dir="./roberta-prompt-lora-improved",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",  # Don't save during training
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,  # Critical for custom labels
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )

    # 8. Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. Evaluate base model BEFORE training (zero-shot)
    print("\n" + "="*50)
    print("ZERO-SHOT EVALUATION (Before Training)")
    print("="*50)
    
    base_metrics = {}
    base_metrics['train_in'] = evaluate_model(model, tokenized_train, tokenizer, label_ids, dataset_name="Train_in (Zero-shot)")
    base_metrics['train_out'] = evaluate_model(model, tokenized_train_out, tokenizer, label_ids, dataset_name="Train_out (Zero-shot)")
    base_metrics['validation'] = evaluate_model(model, tokenized_val, tokenizer, label_ids, dataset_name="Validation (Zero-shot)")

    # 10. Train
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    train_result = trainer.train()
    print(f"\nTraining completed. Loss: {train_result.training_loss:.4f}")
    
    # 11. Evaluate fine-tuned model AFTER training
    print("\n" + "="*50)
    print("POST-TRAINING EVALUATION")
    print("="*50)
    
    final_metrics = {}
    final_metrics['train_in'] = evaluate_model(model, tokenized_train, tokenizer, label_ids, dataset_name="Train_in (Fine-tuned)")
    final_metrics['train_out'] = evaluate_model(model, tokenized_train_out, tokenizer, label_ids, dataset_name="Train_out (Fine-tuned)")
    final_metrics['validation'] = evaluate_model(model, tokenized_val, tokenizer, label_ids, dataset_name="Validation (Fine-tuned)")
    
    # 12. Print comparison
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    for dataset in ['train_in', 'train_out', 'validation']:
        before = base_metrics[dataset]['eval_accuracy']
        after = final_metrics[dataset]['eval_accuracy']
        improvement = after - before
        print(f"{dataset:12} | Before: {before:.4f} | After: {after:.4f} | Δ: {improvement:+.4f}")
    
    # 13. Save the model
    trainer.save_model("./roberta-prompt-lora-final")
    print("\n✓ Model saved to ./roberta-prompt-lora-final")
    
    return model, trainer, base_metrics, final_metrics

if __name__ == "__main__":
    model, trainer, base_metrics, final_metrics = train_prompt_lora()
    
    print("\n=== Training and Evaluation Complete ===")
    
    # Print membership inference attack relevant metrics
    print("\n=== MIA-Relevant Metrics ===")
    train_in_acc = final_metrics['train_in']['eval_accuracy']
    train_out_acc = final_metrics['train_out']['eval_accuracy']
    gap = train_in_acc - train_out_acc
    
    print(f"Train_in accuracy:  {train_in_acc:.4f}")
    print(f"Train_out accuracy: {train_out_acc:.4f}")
    print(f"Memorization gap:   {gap:.4f}")
    
    if gap > 0.05:
        print("⚠ Significant memorization detected (gap > 0.05)")
    else:
        print("✓ Low memorization (gap ≤ 0.05)")