import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from transformers import AutoModelForMaskedLM

class SubspaceProjectionAttack:
    def __init__(self, base_model, lora_model, weighting_strategy="linear"):
        """
        Implements the Subspace Projection Attack targeting Logit Gradients.
        """
        self.base_model = base_model
        self.device = base_model.device
        self.layer_subspaces = {} 
        self.weights = {}         
        self.target_layer_indices = []
        
        print("Initializing Subspace Attack: Decomposing Adapters...")
        
        # 1. Identify Architecture & Layers (RoBERTa specific)
        if hasattr(lora_model.base_model, "roberta"):
            encoder = lora_model.base_model.roberta.encoder
        elif hasattr(lora_model, "roberta"):
            encoder = lora_model.roberta.encoder
        else:
            # Fallback/Generic
            encoder = lora_model.base_model.model.encoder
            
        num_layers = len(encoder.layer)

        # 2. Extract Subspaces (Q_B, Q_A) for every layer
        for i in range(num_layers):
            try:
                # Path to attention weights in RoBERTa
                peft_layer_q = encoder.layer[i].attention.self.query
                peft_layer_v = encoder.layer[i].attention.self.value
            except AttributeError:
                continue

            # Check if LoRA is active on this layer
            if hasattr(peft_layer_q, "lora_A") and hasattr(peft_layer_v, "lora_A"):
                self.target_layer_indices.append(i)
                self.layer_subspaces[i] = {
                    'q': self._get_qr(peft_layer_q),
                    'v': self._get_qr(peft_layer_v)
                }
        
        # 3. Setup Layer Weights
        self._setup_weights(weighting_strategy)
        print(f"Attack initialized on {len(self.target_layer_indices)} layers.")

    def _get_qr(self, lora_layer):
        # Extract LoRA weights: W_delta = B @ A
        A = lora_layer.lora_A.default.weight.detach().float()
        B = lora_layer.lora_B.default.weight.detach().float()
        
        # QR Decomposition
        # Q_B spans the Error space (Columns of B)
        # Q_A spans the Activation space (Rows of A -> Columns of A.T)
        Q_B, _ = torch.linalg.qr(B)   
        Q_A, _ = torch.linalg.qr(A.T) 
        
        return Q_B.to(self.device), Q_A.to(self.device)

    def _setup_weights(self, strategy):
        for i in self.target_layer_indices:
            if strategy == "linear":
                self.weights[i] = (i + 1) # Layer 12 gets 12x weight of Layer 1
            else:
                self.weights[i] = 1.0

    def compute_score(self, input_ids, mask_idx, target_token_id):
        """
        Computes score using gradient of the TARGET LOGIT (not Loss).
        """
        # 1. Enable Gradients on specific Base Model weights
        target_params = []
        for i in self.target_layer_indices:
            layer = self.base_model.roberta.encoder.layer[i].attention.self
            target_params.append(layer.query.weight)
            target_params.append(layer.value.weight)
            
        # 2. Forward Pass
        # We need gradients, so we allow grad accumulation
        outputs = self.base_model(input_ids=input_ids)
        
        # --- CRITICAL CHANGE: TARGET THE LOGIT DIRECTLY ---
        # We compute gradient of the score for the correct verbalizer.
        # This vector represents "The features that increase probability of this class".
        target_logit = outputs.logits[0, mask_idx, target_token_id]
        
        # 3. Compute Gradients
        grads = torch.autograd.grad(target_logit, target_params, retain_graph=False, create_graph=False)
        
        # 4. Project and Score
        total_score = 0.0
        grad_iter = iter(grads)
        
        for i in self.target_layer_indices:
            grad_q = next(grad_iter)
            grad_v = next(grad_iter)
            
            def get_proj_energy(grad, subspace):
                Q_B, Q_A = subspace
                # Project: P = Qb.T @ G @ Qa
                proj = torch.matmul(Q_B.T, grad)
                proj = torch.matmul(proj, Q_A)
                
                # --- RAW ENERGY (Unnormalized) ---
                # We want the magnitude of alignment. 
                # Stronger "feature strength" in the subspace = Member.
                return torch.norm(proj)

            score_q = get_proj_energy(grad_q, self.layer_subspaces[i]['q'])
            score_v = get_proj_energy(grad_v, self.layer_subspaces[i]['v'])
            
            # Weighted average for this layer
            total_score += ((score_q + score_v) / 2.0) * self.weights[i]
            
        return total_score.item()

def evaluate_subspace_attack(model_ft, tokenizer, train_in, validation, label_ids, model_id="roberta-base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load FRESH Base Model (W_0)
    # Crucial: Use float32 for stable gradient calculation
    print("\n[Attack] Loading Base Model for Gradient Computation...")
    base_model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.float32)
    base_model.to(device)
    base_model.eval()
    
    # 2. Initialize Attacker
    attacker = SubspaceProjectionAttack(base_model, model_ft, weighting_strategy="linear")
    mask_token_id = tokenizer.mask_token_id
    
    def score_dataset(dataset, desc):
        scores = []
        for i in tqdm(range(len(dataset)), desc=desc):
            sample = dataset[i]
            
            # Preprocess (Same as training)
            prompt = f"{sample['text'][:300]} Topic: {tokenizer.mask_token}"
            target_id = label_ids[sample['label']]
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96).to(device)
            input_ids = inputs["input_ids"]
            
            # Find mask position
            mask_pos = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_pos) == 0:
                scores.append(0.0)
                continue
            
            mask_idx = mask_pos.item()
            
            # Compute Gradient Projection Score
            base_model.zero_grad()
            # Pass indices instead of labels tensor
            score = attacker.compute_score(input_ids, mask_idx, target_id)
            scores.append(score)
            
        return np.array(scores)

    # 3. Score Datasets
    print("\n[Attack] Scoring Members...")
    scores_in = score_dataset(train_in, "Members")
    
    print("\n[Attack] Scoring Non-Members...")
    scores_out = score_dataset(validation, "Non-Members")
    
    # 4. Diagnostics
    print("\n--- DIAGNOSTICS ---")
    print(f"Members:     Mean={np.mean(scores_in):.4e}, Std={np.std(scores_in):.4e}")
    print(f"Non-Members: Mean={np.mean(scores_out):.4e}, Std={np.std(scores_out):.4e}")

    # 5. Calculate Metrics
    y_true = [1] * len(scores_in) + [0] * len(scores_out)
    y_scores = np.concatenate([scores_in, scores_out])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_1 = np.interp(0.01, fpr, tpr)
    
    print(f"\n>>> SUBSPACE ATTACK RESULTS <<<")
    print(f"AUC:          {roc_auc:.4f}")
    print(f"TPR @ 1% FPR: {tpr_at_1:.4f}")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    del base_model
    del attacker
    torch.cuda.empty_cache()
    
    return roc_auc