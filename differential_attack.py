import torch
import numpy as np

class DifferentialSubspaceAttack:
    def __init__(self, base_model, target_model, shadow_model, weighting_strategy="top_5"):
        """
        Implements Differential Subspace Attack.
        Projections are made onto (Target_Subspace - Shadow_Subspace).
        """
        self.base_model = base_model
        self.device = next(base_model.parameters()).device
        self.private_subspaces = {} 
        self.weights = {}         
        self.target_layer_indices = []
        
        print("Initializing Differential Subspace Attack...")
        
        if hasattr(target_model, "base_model") and hasattr(target_model.base_model, "roberta"):
            encoder = target_model.base_model.roberta.encoder
            shadow_encoder = shadow_model.base_model.roberta.encoder
        elif hasattr(target_model, "base_model") and hasattr(target_model.base_model, "model"):
            encoder = target_model.base_model.model.encoder
            shadow_encoder = shadow_model.base_model.model.encoder
        elif hasattr(target_model, "roberta"):
            encoder = target_model.roberta.encoder
            shadow_encoder = shadow_model.roberta.encoder
        elif hasattr(target_model, "model"):
            encoder = target_model.model.encoder
            shadow_encoder = shadow_model.model.encoder
        else:
            encoder = target_model.encoder
            shadow_encoder = shadow_model.encoder

        if hasattr(base_model, "base_model") and hasattr(base_model.base_model, "roberta"):
            self.base_encoder = base_model.base_model.roberta.encoder
        elif hasattr(base_model, "base_model") and hasattr(base_model.base_model, "model"):
            self.base_encoder = base_model.base_model.model.encoder
        elif hasattr(base_model, "roberta"):
            self.base_encoder = base_model.roberta.encoder
        elif hasattr(base_model, "model"):
            self.base_encoder = base_model.model.encoder
        else:
            self.base_encoder = base_model.encoder
            
        num_layers = len(encoder.layer)

        for i in range(num_layers):
            try:
                tgt_q = encoder.layer[i].attention.self.query
                tgt_v = encoder.layer[i].attention.self.value
                
                shd_q = shadow_encoder.layer[i].attention.self.query
                shd_v = shadow_encoder.layer[i].attention.self.value
            except AttributeError: continue

            if hasattr(tgt_q, "lora_A") and hasattr(shd_q, "lora_A"):
                self.target_layer_indices.append(i)
                
                private_q = self._get_private_subspace(tgt_q, shd_q)
                private_v = self._get_private_subspace(tgt_v, shd_v)
                
                self.private_subspaces[i] = {
                    'q': private_q,
                    'v': private_v
                }
        
        self._setup_weights(weighting_strategy)
        self.active_layers = [i for i in self.target_layer_indices if self.weights[i] > 0]

    def _get_private_subspace(self, target_layer, shadow_layer):
        """
        Returns the subspace of Target B that is ORTHOGONAL to Shadow B.
        """
        B_tgt = target_layer.lora_B.default.weight.detach().float()
        B_shd = shadow_layer.lora_B.default.weight.detach().float()
        
        Q_shd, _ = torch.linalg.qr(B_shd)
        
        Proj = torch.matmul(Q_shd, torch.matmul(Q_shd.T, B_tgt))
        
        Residual = B_tgt - Proj
        
        Q_private, _ = torch.linalg.qr(Residual)
        
        A_tgt = target_layer.lora_A.default.weight.detach().float()
        Q_A, _ = torch.linalg.qr(A_tgt.T)
        
        return Q_private.to(self.device), Q_A.to(self.device)

    def _setup_weights(self, strategy):
        """Assign per-layer weights for aggregation."""
        for i in self.target_layer_indices: self.weights[i] = 0.0
        if strategy == "top_5":
            for i in sorted(self.target_layer_indices)[-5:]: self.weights[i] = 1.0
        else:
            for i in self.target_layer_indices: self.weights[i] = 1.0

    def compute_score(self, input_ids, mask_idx, target_token_id):
        """Compute a private-subspace projection score for one example."""
        if not self.active_layers:
            return 0.0

        target_params = []
        for i in self.active_layers:
            layer = self.base_encoder.layer[i].attention.self
            layer.query.weight.requires_grad_(True)
            layer.value.weight.requires_grad_(True)
            target_params.append(layer.query.weight)
            target_params.append(layer.value.weight)
            
        if not target_params:
            return 0.0

        self.base_model.zero_grad()
        with torch.enable_grad():
            outputs = self.base_model(input_ids=input_ids)
            target_logit = outputs.logits[0, mask_idx, target_token_id]
            grads = torch.autograd.grad(target_logit, target_params, retain_graph=False, create_graph=False)

        for p in target_params:
            p.requires_grad_(False)
        
        total_score = 0.0
        grad_iter = iter(grads)
        
        for i in self.active_layers:
            grad_q = next(grad_iter)
            grad_v = next(grad_iter)
            
            def get_score(grad, subspace):
                Q_B, Q_A = subspace
                proj = torch.matmul(Q_B.T, grad)
                proj = torch.matmul(proj, Q_A)
                return torch.norm(proj) / (torch.norm(grad) + 1e-9)

            score_q = get_score(grad_q, self.private_subspaces[i]['q'])
            score_v = get_score(grad_v, self.private_subspaces[i]['v'])
            
            total_score += ((score_q + score_v) / 2.0) * self.weights[i]
            
        return (total_score / len(self.active_layers)).item()
