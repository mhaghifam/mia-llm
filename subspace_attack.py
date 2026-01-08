import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

class SubspaceProjectionAttack:
    def __init__(self, lora_model, weighting_strategy="top_5"):
        """
        Implements the Subspace Projection Attack using logit gradients and cosine similarity.
        Gradients are taken w.r.t. the *fine-tuned* (LoRA) model.
        """
        self.model = lora_model
        self.device = next(lora_model.parameters()).device
        self.layer_subspaces = {}
        self.weights = {}
        self.target_layer_indices = []

        print(f"Initializing Subspace Attack (Strategy: {weighting_strategy})...")

        if hasattr(lora_model.base_model, "roberta"):
            encoder = lora_model.base_model.roberta.encoder
        elif hasattr(lora_model, "roberta"):
            encoder = lora_model.roberta.encoder
        else:
            encoder = lora_model.base_model.model.encoder

        num_layers = len(encoder.layer)
        self.encoder = encoder

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

        self.active_layers = [i for i in self.target_layer_indices if self.weights[i] > 0]
        print(f"Active Layers for Gradient Computation: {self.active_layers}")

    def _get_qr(self, lora_layer):
        """Return QR bases for LoRA B and A subspaces."""
        A = lora_layer.lora_A.default.weight.detach().float()
        B = lora_layer.lora_B.default.weight.detach().float()
        
        Q_B, _ = torch.linalg.qr(B)   
        Q_A, _ = torch.linalg.qr(A.T) 
        
        return Q_B.to(self.device), Q_A.to(self.device)

    def _setup_weights(self, strategy):
        """Assign per-layer weights for aggregation."""
        for i in self.target_layer_indices:
            self.weights[i] = 0.0
            
        if strategy == "top_5":
            top_indices = sorted(self.target_layer_indices)[-5:]
            for i in top_indices:
                self.weights[i] = 1.0 
                
        elif strategy == "linear":
            for i in self.target_layer_indices:
                self.weights[i] = (i + 1)
                
        else:
            for i in self.target_layer_indices:
                self.weights[i] = 1.0

    def compute_score(self, input_ids, mask_idx, target_token_id):
        """
        Computes the alignment score using Logit Gradients.
        """
        if not self.active_layers:
            return 0.0

        target_params = []
        for i in self.active_layers:
            layer = self.encoder.layer[i].attention.self
            layer.query.weight.requires_grad_(True)
            layer.value.weight.requires_grad_(True)
            target_params.append(layer.query.weight)
            target_params.append(layer.value.weight)
            
        if not target_params:
            return 0.0

        with torch.enable_grad():
            outputs = self.model(input_ids=input_ids)
        
        target_logit = outputs.logits[0, mask_idx, target_token_id]
        
        grads = torch.autograd.grad(target_logit, target_params, retain_graph=False, create_graph=False)
        
        for p in target_params:
            p.requires_grad_(False)

        total_score = 0.0
        grad_iter = iter(grads)
        
        for i in self.active_layers:
            grad_q = next(grad_iter)
            grad_v = next(grad_iter)
            
            def get_cosine_similarity(grad, subspace):
                Q_B, Q_A = subspace
                proj = torch.matmul(Q_B.T, grad)
                proj = torch.matmul(proj, Q_A)
                
                energy_subspace = torch.norm(proj)
                energy_total = torch.norm(grad)
                
                return energy_subspace / (energy_total + 1e-9)

            score_q = get_cosine_similarity(grad_q, self.layer_subspaces[i]['q'])
            score_v = get_cosine_similarity(grad_v, self.layer_subspaces[i]['v'])
            
            total_score += (score_q + score_v) / 2.0
            
        return (total_score / len(self.active_layers)).item()

def evaluate_subspace_attack(model_ft, tokenizer, train_in, validation, label_ids):
    """Evaluate the subspace attack and return ROC-AUC."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[Attack] Using fine-tuned model for gradient computation...")
    model_ft.to(device)
    model_ft.eval()

    attacker = SubspaceProjectionAttack(model_ft, weighting_strategy="top_5")
    mask_token_id = tokenizer.mask_token_id
    
    def score_dataset(dataset, desc, attacker_obj):
        """Score a dataset with the subspace attack."""
        scores = []
        for i in tqdm(range(len(dataset)), desc=desc):
            sample = dataset[i]
            
            prompt = f"{sample['text'][:300]} Topic: {tokenizer.mask_token}"
            target_id = label_ids[sample['label']]
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96).to(device)
            input_ids = inputs["input_ids"]
            
            mask_pos = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_pos) == 0:
                scores.append(0.0)
                continue
            
            mask_idx = mask_pos.item()
            
            attacker_obj.model.zero_grad()
            score = attacker_obj.compute_score(input_ids, mask_idx, target_id)
            scores.append(score)
            
        return np.array(scores)

    print("\n[Attack] Scoring Members...")
    scores_in = score_dataset(train_in, "Members", attacker)
    
    print("\n[Attack] Scoring Non-Members...")
    scores_out = score_dataset(validation, "Non-Members", attacker)
    
    print("\n--- DIAGNOSTICS ---")
    print(f"Members:     Mean={np.mean(scores_in):.4f}, Std={np.std(scores_in):.4f}")
    print(f"Non-Members: Mean={np.mean(scores_out):.4f}, Std={np.std(scores_out):.4f}")

    y_true = [1] * len(scores_in) + [0] * len(scores_out)
    y_scores = np.concatenate([scores_in, scores_out])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_1 = np.interp(0.01, fpr, tpr)
    
    print("\n>>> SUBSPACE ATTACK RESULTS (Logit Gradient + Cosine Sim) <<<")
    print(f"AUC:          {roc_auc:.4f}")
    print(f"TPR @ 1% FPR: {tpr_at_1:.4f}")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    del attacker
    torch.cuda.empty_cache()
    
    return roc_auc
