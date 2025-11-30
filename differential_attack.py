import torch
import numpy as np

class DifferentialSubspaceAttack:
    def __init__(self, base_model, target_model, shadow_model, weighting_strategy="top_5"):
        """
        Implements Differential Subspace Attack.
        Projections are made onto (Target_Subspace - Shadow_Subspace).
        """
        self.base_model = base_model
        self.device = base_model.device
        self.private_subspaces = {} 
        self.weights = {}         
        self.target_layer_indices = []
        
        print("Initializing Differential Subspace Attack...")
        
        # 1. Identify Layers
        # (Assuming RoBERTa structure for both)
        if hasattr(target_model.base_model, "roberta"):
            encoder = target_model.base_model.roberta.encoder
            shadow_encoder = shadow_model.base_model.roberta.encoder
        else:
            encoder = target_model.roberta.encoder
            shadow_encoder = shadow_model.roberta.encoder
            
        num_layers = len(encoder.layer)

        # 2. Extract and Subtract Subspaces
        for i in range(num_layers):
            try:
                # Target Layers
                tgt_q = encoder.layer[i].attention.self.query
                tgt_v = encoder.layer[i].attention.self.value
                
                # Shadow Layers
                shd_q = shadow_encoder.layer[i].attention.self.query
                shd_v = shadow_encoder.layer[i].attention.self.value
            except AttributeError: continue

            if hasattr(tgt_q, "lora_A") and hasattr(shd_q, "lora_A"):
                self.target_layer_indices.append(i)
                
                # Compute Private Subspaces
                # We filter Target B against Shadow B
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
        # 1. Get Matrices
        B_tgt = target_layer.lora_B.default.weight.detach().float()
        B_shd = shadow_layer.lora_B.default.weight.detach().float()
        
        # 2. Orthogonalize Target B w.r.t Shadow B
        # We want the component of B_tgt that cannot be explained by B_shd.
        
        # First, get an orthonormal basis for Shadow
        # Q_shd: (d_out, r)
        Q_shd, _ = torch.linalg.qr(B_shd)
        
        # Project Target onto Shadow
        # Proj = Q_shd @ (Q_shd.T @ B_tgt)
        Proj = torch.matmul(Q_shd, torch.matmul(Q_shd.T, B_tgt))
        
        # Subtract to get Residual (The Private Component)
        Residual = B_tgt - Proj
        
        # 3. Get Basis for the Residual
        # Q_private spans the "New" directions found only in Target
        Q_private, _ = torch.linalg.qr(Residual)
        
        # For A (Input side), we usually just take the Target's A
        # or we could do the same subtraction logic. 
        # For simplicity, let's trust the Output Space (B) filtering is enough.
        A_tgt = target_layer.lora_A.default.weight.detach().float()
        Q_A, _ = torch.linalg.qr(A_tgt.T)
        
        return Q_private.to(self.device), Q_A.to(self.device)

    def _setup_weights(self, strategy):
        # (Same weighting logic as before)
        for i in self.target_layer_indices: self.weights[i] = 0.0
        if strategy == "top_5":
            for i in sorted(self.target_layer_indices)[-5:]: self.weights[i] = 1.0
        else:
            for i in self.target_layer_indices: self.weights[i] = 1.0

    def compute_score(self, input_ids, mask_idx, target_token_id):
        # (Same gradient computation logic as before, but using self.private_subspaces)
        target_params = []
        for i in self.active_layers:
            layer = self.base_model.roberta.encoder.layer[i].attention.self
            target_params.append(layer.query.weight)
            target_params.append(layer.value.weight)
            
        outputs = self.base_model(input_ids=input_ids)
        target_logit = outputs.logits[0, mask_idx, target_token_id]
        grads = torch.autograd.grad(target_logit, target_params, retain_graph=False, create_graph=False)
        
        total_score = 0.0
        grad_iter = iter(grads)
        
        for i in self.active_layers:
            grad_q = next(grad_iter)
            grad_v = next(grad_iter)
            
            def get_score(grad, subspace):
                Q_B, Q_A = subspace
                # Project onto PRIVATE subspace
                proj = torch.matmul(Q_B.T, grad)
                proj = torch.matmul(proj, Q_A)
                return torch.norm(proj) / (torch.norm(grad) + 1e-9)

            score_q = get_score(grad_q, self.private_subspaces[i]['q'])
            score_v = get_score(grad_v, self.private_subspaces[i]['v'])
            
            total_score += ((score_q + score_v) / 2.0) * self.weights[i]
            
        return (total_score / len(self.active_layers)).item()