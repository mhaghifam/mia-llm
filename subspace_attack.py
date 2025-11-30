import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

class SubspaceProjectionAttack:
    def __init__(self, base_model, lora_model, weighting_strategy="linear"):
        """
        Implements the Gradient Subspace Projection Attack.
        
        Args:
            base_model: The frozen base model (W_0) used to compute 'trajectory' gradients.
            lora_model: The fine-tuned PEFT model containing the adapters (A, B).
            weighting_strategy: 'linear' (default) gives higher weight to later layers.
        """
        self.base_model = base_model
        self.device = base_model.device
        self.layer_subspaces = {} 
        self.weights = {}         
        self.target_layer_indices = []
        
        print("Initializing Subspace Attack: Decomposing Adapters...")
        
        # 1. Identify Architecture & Layers
        if hasattr(lora_model.base_model, "roberta"):
            encoder = lora_model.base_model.roberta.encoder
        elif hasattr(lora_model, "roberta"):
            encoder = lora_model.roberta.encoder
        else:
            # Fallback for other HF models (like BERT)
            encoder = lora_model.base_model.model.encoder
            
        num_layers = len(encoder.layer)

        # 2. Extract Subspaces (Q_B, Q_A) for every layer
        for i in range(num_layers):
            try:
                # Path to attention weights in standard RoBERTa/BERT
                peft_layer_q = encoder.layer[i].attention.self.query
                peft_layer_v = encoder.layer[i].attention.self.value
            except AttributeError:
                continue

            # Check if LoRA is active on this layer
            if hasattr(peft_layer_q, "lora_A") and hasattr(peft_layer_v, "lora_A"):
                self.target_layer_indices.append(i)
                # Store orthonormal bases for Query and Value projections
                self.layer_subspaces[i] = {
                    'q': self._get_qr(peft_layer_q),
                    'v': self._get_qr(peft_layer_v)
                }
        
        # 3. Setup Layer Weights
        self._setup_weights(weighting_strategy)
        print(f"Attack initialized on {len(self.target_layer_indices)} layers.")

    def _get_qr(self, lora_layer):
        """
        Extracts orthonormal basis from LoRA weights: W_delta = B @ A
        """
        # lora_A is (r, in), lora_B is (out, r)
        A = lora_layer.lora_A.default.weight.detach().float()
        B = lora_layer.lora_B.default.weight.detach().float()
        
        # QR Decomposition
        # Q_B spans the Error space (Columns of B)
        # Q_A spans the Activation space (Row space of A -> Cols of A.T)
        Q_B, _ = torch.linalg.qr(B)   
        Q_A, _ = torch.linalg.qr(A.T) 
        
        return Q_B.to(self.device), Q_A.to(self.device)

    def _setup_weights(self, strategy):
        """
        Assigns weights to layers based on the hypothesis that higher layers 
        contain more semantic/memorized information.
        """
        for i in self.target_layer_indices:
            if strategy == "linear":
                self.weights[i] = (i + 1) # Layer 12 gets 12x weight of Layer 1
            elif strategy == "exponential":
                self.weights[i] = np.exp(i / 12.0)
            else:
                self.weights[i] = 1.0

    def compute_score(self, input_ids, labels):
        """
        Computes the weighted projection score for a single sample.
        Score = Sum( w_l * || Qb_l^T * Grad_l * Qa_l || / || Grad_l || )
        """
        # 1. Enable Gradients ONLY on specific Base Model weights to save memory
        target_params = []
        for i in self.target_layer_indices:
            # Warning: This logic assumes RoBERTa architecture
            layer = self.base_model.roberta.encoder.layer[i].attention.self
            target_params.append(layer.query.weight)
            target_params.append(layer.value.weight)
            
        # 2. Forward Pass
        # We need gradients, so we ensure the model is in a differentiable state
        # even if it's technically 'eval' mode
        outputs = self.base_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # 3. Compute Gradients
        # autograd.grad computes dL/dW for the listed params
        grads = torch.autograd.grad(loss, target_params, retain_graph=False, create_graph=False)
        
        # 4. Project and Score
        total_score = 0.0
        grad_iter = iter(grads)
        
        for i in self.target_layer_indices:
            # Gradients come in pairs (Query, Value) because of how we appended them
            grad_q = next(grad_iter)
            grad_v = next(grad_iter)
            
            def get_proj_norm(grad, subspace):
                Q_B, Q_A = subspace
                # Project: P = Qb.T @ G @ Qa
                # Dimensions: (r, d_out) @ (d_out, d_in) @ (d_in, r) -> (r, r)
                proj = torch.matmul(Q_B.T, grad)
                proj = torch.matmul(proj, Q_A)
                
                # Normalized Score: Energy in subspace / Total Gradient Energy
                return torch.norm(proj) / (torch.norm(grad) + 1e-9)

            score_q = get_proj_norm(grad_q, self.layer_subspaces[i]['q'])
            score_v = get_proj_norm(grad_v, self.layer_subspaces[i]['v'])
            
            # Average score for this layer * Layer Weight
            total_score += ((score_q + score_v) / 2.0) * self.weights[i]
            
        return total_score.item()

def evaluate_subspace_attack(model_ft, tokenizer, train_in, validation, label_ids, model_id="roberta-base"):
    """
    Orchestrator function to run the attack evaluation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Base Model (W_0)
    # Crucial: Use float32 for stable gradient calculation
    from transformers import AutoModelForMaskedLM
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
            # Reconstruct Prompt exactly as training
            # Note: Ensure text slicing matches training exactly (e.g., [:300])
            prompt = f"{sample['text'][:300]} Topic: {tokenizer.mask_token}"
            target_id = label_ids[sample['label']]
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96).to(device)
            
            # Construct Labels (-100 everywhere except mask)
            input_ids = inputs["input_ids"][0]
            labels = torch.full_like(input_ids, -100)
            
            # Find mask position
            mask_pos = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_pos) == 0:
                scores.append(0.0)
                continue
            
            labels[mask_pos.item()] = target_id
            
            # Compute Gradient Projection Score
            base_model.zero_grad()
            score = attacker.compute_score(inputs["input_ids"], labels.unsqueeze(0))
            scores.append(score)
            
        return np.array(scores)

    # 3. Score Datasets
    print("\n[Attack] Scoring Members...")
    scores_in = score_dataset(train_in, "Members")
    
    print("\n[Attack] Scoring Non-Members...")
    scores_out = score_dataset(validation, "Non-Members")
    
    # 4. Calculate Metrics
    y_true = [1] * len(scores_in) + [0] * len(scores_out)
    y_scores = np.concatenate([scores_in, scores_out])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_1 = np.interp(0.01, fpr, tpr)
    
    print(f"\n>>> SUBSPACE ATTACK RESULTS <<<")
    print(f"AUC:          {roc_auc:.4f}")
    print(f"TPR @ 1% FPR: {tpr_at_1:.4f}")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    # Clean up memory
    del base_model
    del attacker
    torch.cuda.empty_cache()
    
    return roc_auc, tpr_at_1