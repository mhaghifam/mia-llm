import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    TrainingArguments, 
    Trainer, 
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import evaluate
import gc
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "roberta-base"
SEED = 42
set_seed(SEED)

VERBALIZER_WORDS = ["World", "Sports", "Business", "Technology"]

# ==========================================
# 2. DATA & PREPROCESSING
# ==========================================
def get_data():
    dataset = load_dataset("ag_news")
    full_train = dataset["train"].shuffle(seed=SEED)
    
    # --- UPDATED: Training on 10,000 samples ---
    train_in = full_train.select(range(0, 10000))
    
    # Official Test set as Non-Member proxy
    validation = dataset["test"] 
    return train_in, validation

def preprocess_data(tokenizer):
    # Corrected Label IDs for RoBERTa (handling the special leading space)
    label_ids = [tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in VERBALIZER_WORDS]
    print(f"Verbalizer Token IDs: {label_ids}") 

    mask_id = tokenizer.mask_token_id
    
    def preprocess(examples):
        inputs = []
        for text in examples['text']:
            # Prompt construction
            prompt = f"{text[:300]} Topic: {tokenizer.mask_token}"
            inputs.append(prompt)
            
        model_inputs = tokenizer(inputs, truncation=True, max_length=96, padding=False)
        
        labels_list = []
        for i, (input_ids, label_idx) in enumerate(zip(model_inputs["input_ids"], examples['label'])):
            # Create label sequence: -100 everywhere except the MASK position
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

# ==========================================
# 3. METRICS
# ==========================================
def compute_metrics(eval_pred, label_ids):
    logits, labels = eval_pred
    predictions = []
    references = []
    
    for i in range(len(logits)):
        mask_indices = np.where(labels[i] != -100)[0]
        if len(mask_indices) == 0: continue
        mask_idx = mask_indices[0]
        
        # Logits are pre-filtered to just the 4 class columns
        token_logits = logits[i, mask_idx, :] 
        predicted_idx = np.argmax(token_logits)
        
        true_token_id = labels[i, mask_idx]
        if true_token_id in label_ids:
            true_class = label_ids.index(true_token_id)
            predictions.append(predicted_idx)
            references.append(true_class)

    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=references)

# ==========================================
# 4. ATTACK: SUBSPACE PROJECTION (OURS)
# ==========================================
class SubspaceProjectionAttack:
    def __init__(self, base_model, lora_model, weighting_strategy="linear"):
        self.base_model = base_model
        self.device = base_model.device
        self.layer_subspaces = {} 
        self.weights = {}         
        self.target_layer_indices = []
        
        print("Initializing Subspace Attack: Decomposing Adapters...")
        
        if hasattr(lora_model.base_model, "roberta"):
            encoder = lora_model.base_model.roberta.encoder
        else:
            encoder = lora_model.roberta.encoder
            
        num_layers = len(encoder.layer)

        for i in range(num_layers):
            try:
                peft_layer_q = encoder.layer[i].attention.self.query
                peft_layer_v = encoder.layer[i].attention.self.value
            except AttributeError:
                continue

            if hasattr(peft_layer_q, "lora_A") and hasattr(peft_layer_v, "lora_A"):
                self.target_layer_indices.append(i)
                self.layer_subspaces[i] = {
                    'q': self._get_qr(peft_layer_q),
                    'v': self._get_qr(peft_layer_v)
                }
        
        self._setup_weights(weighting_strategy)
        print(f"Attack initialized on {len(self.target_layer_indices)} layers.")

    def _get_qr(self, lora_layer):
        A = lora_layer.lora_A.default.weight.detach().float()
        B = lora_layer.lora_B.default.weight.detach().float()
        
        # QR Decomposition
        Q_B, _ = torch.linalg.qr(B)   
        Q_A, _ = torch.linalg.qr(A.T) 
        
        return Q_B.to(self.device), Q_A.to(self.device)

    def _setup_weights(self, strategy):
        for i in self.target_layer_indices:
            if strategy == "linear":
                self.weights[i] = (i + 1)
            else:
                self.weights[i] = 1.0

    def compute_score(self, input_ids, labels):
        target_params = []
        for i in self.target_layer_indices:
            layer = self.base_model.roberta.encoder.layer[i].attention.self
            target_params.append(layer.query.weight)
            target_params.append(layer.value.weight)
            
        outputs = self.base_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        grads = torch.autograd.grad(loss, target_params, retain_graph=False, create_graph=False)
        
        total_score = 0.0
        grad_iter = iter(grads)
        
        for i in self.target_layer_indices:
            grad_q = next(grad_iter)
            grad_v = next(grad_iter)
            
            def get_proj_norm(grad, subspace):
                Q_B, Q_A = subspace
                proj = torch.matmul(Q_B.T, grad)
                proj = torch.matmul(proj, Q_A)
                return torch.norm(proj) / (torch.norm(grad) + 1e-9)

            score_q = get_proj_norm(grad_q, self.layer_subspaces[i]['q'])
            score_v = get_proj_norm(grad_v, self.layer_subspaces[i]['v'])
            
            total_score += ((score_q + score_v) / 2.0) * self.weights[i]
            
        return total_score.item()

def run_subspace_attack(model_ft, tokenizer, train_in, validation, label_ids):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nLoading Base Model for Gradient Computation...")
    base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    base_model.to(device)
    base_model.eval()
    
    attacker = SubspaceProjectionAttack(base_model, model_ft, weighting_strategy="linear")
    mask_token_id = tokenizer.mask_token_id
    
    def score_dataset(dataset):
        scores = []
        for i in tqdm(range(len(dataset)), desc="Subspace Scoring"):
            sample = dataset[i]
            prompt = f"{sample['text'][:300]} Topic: {tokenizer.mask_token}"
            target_id = label_ids[sample['label']]
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96).to(device)
            
            input_ids = inputs["input_ids"][0]
            labels = torch.full_like(input_ids, -100)
            mask_pos = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_pos) == 0:
                scores.append(0.0)
                continue
            
            labels[mask_pos.item()] = target_id
            
            base_model.zero_grad()
            score = attacker.compute_score(inputs["input_ids"], labels.unsqueeze(0))
            scores.append(score)
        return np.array(scores)

    print("Scoring Members...")
    scores_in = score_dataset(train_in)
    print("Scoring Non-Members...")
    scores_out = score_dataset(validation)
    
    evaluate_roc(scores_in, scores_out, "Subspace Projection")

# ==========================================
# 5. ATTACK: CALIBRATED LOSS (BASELINE)
# ==========================================
def compute_calibrated_loss_scores(model_ft, model_base, dataset, tokenizer, label_ids, device):
    model_ft.eval()
    model_base.eval()
    scores = []
    mask_token_id = tokenizer.mask_token_id
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Loss Scoring"):
            sample = dataset[i]
            prompt = f"{sample['text'][:300]} Topic: {tokenizer.mask_token}"
            target_id = label_ids[sample['label']]
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            input_ids = inputs["input_ids"][0]
            mask_pos = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_pos) == 0:
                scores.append(0.0)
                continue
            mask_idx = mask_pos.item()
            
            logits_ft = model_ft(**inputs).logits[0, mask_idx, :]
            logits_base = model_base(**inputs).logits[0, mask_idx, :]
            
            lp_ft = logits_ft[target_id] - torch.logsumexp(logits_ft, dim=0)
            lp_base = logits_base[target_id] - torch.logsumexp(logits_base, dim=0)
            
            scores.append((lp_ft - lp_base).item())
            
    return np.array(scores)

def run_calibrated_loss_attack(model_ft, tokenizer, train_in, validation, label_ids):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nLoading Base Model for Loss Calibration...")
    base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    
    print("Scoring Members...")
    scores_in = compute_calibrated_loss_scores(model_ft, base_model, train_in, tokenizer, label_ids, device)
    print("Scoring Non-Members...")
    scores_out = compute_calibrated_loss_scores(model_ft, base_model, validation, tokenizer, label_ids, device)
    
    evaluate_roc(scores_in, scores_out, "Calibrated Loss")

def evaluate_roc(scores_in, scores_out, name):
    y_true = [1] * len(scores_in) + [0] * len(scores_out)
    y_scores = np.concatenate([scores_in, scores_out])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_1 = np.interp(0.01, fpr, tpr)
    
    print(f"\n========================================")
    print(f"   {name.upper()} RESULTS")
    print(f"========================================")
    print(f"AUC:          {roc_auc:.4f}")
    print(f"TPR @ 1% FPR: {tpr_at_1:.4f}")
    print(f"========================================")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    train_in, validation = get_data()
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
    
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=16,
        lora_dropout=0.1, target_modules=["query", "value"], bias="none"
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple): logits = logits[1]
        return logits[:, :, label_ids]
    
    training_args = TrainingArguments(
        output_dir="./lora-prompt",
        learning_rate=2e-4,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        gradient_checkpointing=False
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
    
    print("\n=== STARTING TRAINING ===")
    trainer.train()
    
    trainer.save_model("./final-model")

    # --- START ATTACK EVALUATION ---
    # Subsample to 1,000 samples for fast evaluation
    ATTACK_SIZE = 1000
    print(f"\nSubsampling {ATTACK_SIZE} samples for attack evaluation...")
    
    subset_in = train_in.shuffle(seed=SEED).select(range(ATTACK_SIZE))
    subset_out = validation.shuffle(seed=SEED).select(range(ATTACK_SIZE))

    print("\n=== RUNNING BASELINE: CALIBRATED LOSS ATTACK ===")
    run_calibrated_loss_attack(model, tokenizer, subset_in, subset_out, label_ids)
    
    print("\n=== RUNNING OURS: SUBSPACE PROJECTION ATTACK ===")
    run_subspace_attack(model, tokenizer, subset_in, subset_out, label_ids)

if __name__ == "__main__":
    main()