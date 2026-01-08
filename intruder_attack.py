import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from transformers import AutoModelForMaskedLM

class SpectralIntruderAttack:
    def __init__(self, base_model, lora_model, threshold=0.1, top_k_base=64):
        """
        Implements the Spectral Intruder Attack.
        
        Args:
            base_model: The frozen pre-trained model (W_0).
            lora_model: The fine-tuned PEFT model (W_FT).
            threshold: Cosine similarity cutoff (0.1 means <10% alignment is an intruder).
            top_k_base: Number of principal components of W_0 to check against.
        """
        self.base_model = base_model
        self.model_ft = lora_model
        self.device = next(lora_model.parameters()).device
        self.intruder_subspaces = {} 
        self.target_layer_indices = []
        
        self.base_model.eval()
        
        print(f"Initializing Intruder Attack (Threshold < {threshold})...")
        
        if hasattr(lora_model.base_model, "roberta"):
            self.ft_encoder = lora_model.base_model.roberta.encoder
            self.base_encoder = base_model.roberta.encoder
        elif hasattr(lora_model.base_model, "model"):
            self.ft_encoder = lora_model.base_model.model.encoder
            self.base_encoder = base_model.model.encoder
        else:
            self.ft_encoder = lora_model.base_model.encoder
            self.base_encoder = base_model.encoder
            
        num_layers = len(self.ft_encoder.layer)

        for i in tqdm(range(num_layers), desc="Spectral Analysis"):
            try:
                peft_layer_q = self.ft_encoder.layer[i].attention.self.query
                peft_layer_v = self.ft_encoder.layer[i].attention.self.value
                
                W0_q = self.base_encoder.layer[i].attention.self.query.weight.detach()
                W0_v = self.base_encoder.layer[i].attention.self.value.weight.detach()
            except AttributeError:
                continue

            if hasattr(peft_layer_q, "lora_A"):
                self.target_layer_indices.append(i)
                
                intruders_q = self._find_intruders(peft_layer_q, W0_q, threshold, top_k_base)
                intruders_v = self._find_intruders(peft_layer_v, W0_v, threshold, top_k_base)
                
                self.intruder_subspaces[i] = {
                    'q': intruders_q,
                    'v': intruders_v
                }
        
        self.active_layers = [i for i in self.target_layer_indices 
                              if self.intruder_subspaces[i]['q'] is not None 
                              or self.intruder_subspaces[i]['v'] is not None]
        
        print(f"Intruder Dimensions found in {len(self.active_layers)}/{num_layers} layers.")

    def _find_intruders(self, lora_layer, W_pre, threshold, top_k):
        """Return intruder directions in the LoRA-updated weight matrix."""
        A = lora_layer.lora_A.default.weight.detach().cpu().float()
        B = lora_layer.lora_B.default.weight.detach().cpu().float()
        W_pre = W_pre.cpu().float()
        
        W_FT = W_pre + torch.matmul(B, A)
        
        try:
            U_pre, _, _ = torch.linalg.svd(W_pre, full_matrices=False)
        except RuntimeError: return None
        
        valid_k = min(top_k, U_pre.shape[1])
        U_safe = U_pre[:, :valid_k]
        
        try:
            U_ft, _, _ = torch.linalg.svd(W_FT, full_matrices=False)
        except RuntimeError: return None
        
        check_depth = min(64, U_ft.shape[1])
        U_ft_check = U_ft[:, :check_depth]
        
        intruders = []
        
        for j in range(U_ft_check.shape[1]):
            u_vec = U_ft_check[:, j]
            
            projection = torch.matmul(U_safe.T, u_vec)
            alignment = torch.norm(projection)
            
            if alignment < threshold:
                intruders.append(u_vec)
        
        if not intruders:
            return None
            
        return torch.stack(intruders, dim=1).to(self.device)

    def compute_score(self, input_ids, mask_idx, target_token_id):
        """Compute an intruder projection score for one example."""
        if not self.active_layers:
            return 0.0

        target_params = []
        
        for i in self.active_layers:
            layer = self.ft_encoder.layer[i].attention.self
            
            layer.query.weight.requires_grad_(True)
            layer.value.weight.requires_grad_(True)
            
            target_params.append(layer.query.weight)
            target_params.append(layer.value.weight)

        if not target_params:
            return 0.0
            
        with torch.enable_grad():
            outputs = self.model_ft(input_ids=input_ids)
            target_logit = outputs.logits[0, mask_idx, target_token_id]
            
            grads = torch.autograd.grad(target_logit, target_params, retain_graph=False, create_graph=False)
        
        for p in target_params:
            p.requires_grad_(False)

        total_score = 0.0
        grad_iter = iter(grads)
        valid_counts = 0
        
        for i in self.active_layers:
            grad_q = next(grad_iter)
            grad_v = next(grad_iter)
            
            def score(grad, Q_int):
                if Q_int is None: return 0.0
                proj = torch.matmul(Q_int.T, grad)
                return torch.norm(proj) / (torch.norm(grad) + 1e-9)

            s_q = score(grad_q, self.intruder_subspaces[i]['q'])
            s_v = score(grad_v, self.intruder_subspaces[i]['v'])
            
            if self.intruder_subspaces[i]['q'] is not None:
                total_score += s_q
                valid_counts += 1
            if self.intruder_subspaces[i]['v'] is not None:
                total_score += s_v
                valid_counts += 1
                
        if valid_counts == 0: return 0.0
        return (total_score / valid_counts).item()

def evaluate_intruder_attack(model_ft, tokenizer, train_in, validation, label_ids, threshold=0.1, top_k_base=64):
    """Evaluate the intruder attack and return ROC-AUC."""
    device = next(model_ft.parameters()).device
    
    print("\n[Attack] Loading Base Model for Spectral Comparison...")
    base_model = AutoModelForMaskedLM.from_pretrained("roberta-base", torch_dtype=torch.float32).to("cpu")
    base_model.eval()
    
    attacker = SpectralIntruderAttack(base_model, model_ft, threshold=threshold, top_k_base=top_k_base)
    
    mask_token_id = tokenizer.mask_token_id
    
    def score_dataset(dataset, desc, attacker_obj):
        """Score a dataset with the intruder attack."""
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
            
            model_ft.zero_grad()
            
            score = attacker_obj.compute_score(input_ids, mask_idx, target_id)
            scores.append(score)
        return np.array(scores)

    print("Scoring Members...")
    scores_in = score_dataset(train_in, "Members", attacker)
    print("Scoring Non-Members...")
    scores_out = score_dataset(validation, "Non-Members", attacker)
    
    print(f"\nMembers Mean: {np.mean(scores_in):.4f}")
    print(f"Non-Members Mean: {np.mean(scores_out):.4f}")
    
    y_true = [1]*len(scores_in) + [0]*len(scores_out)
    y_scores = np.concatenate([scores_in, scores_out])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n>>> INTRUDER ATTACK (T={threshold}) AUC: {roc_auc:.4f} <<<")
    
    del base_model
    del attacker
    torch.cuda.empty_cache()
    
    return roc_auc
