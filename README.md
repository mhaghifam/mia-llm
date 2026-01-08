This repo explores white-box membership inference attacks on LoRA-finetuned models. In standard settings, the usual baseline using the log-likelihood ratio $\log P_{\text{FT}}(y|x) - \log P_{\text{PT}}(y|x)$ often fails because LoRA acts as a strong regularizer: loss gaps between train and test points become very small.

Our goal is to go beyond scalar loss and exploit the geometry of the LoRA update itself and design white-box attacks. This repository contains code and experiments for testing these attacks against the baseline on simple setups (e.g., RoBERTa + LoRA on SST-2).

## Attacks

- Baseline (calibrated loss / log-ratio): compute $\log P_{\text{FT}}(y|x) - \log P_{\text{PT}}(y|x)$ using the fine-tuned and pre-trained models. Implemented in `prompt_based_lora.py` and `mia_lora.py`.
- Subspace Projection Attack (`subspace_attack.py`): compute gradients of the target logit w.r.t. base weights and score how much gradient energy lies in the LoRA update subspace (QR of LoRA B/A).
- Intruder Attack (`intruder_attack.py`): find intruder singular vectors in $W_{\text{FT}} = W_0 + BA$ that are weakly aligned with top singular vectors of $W_0$; score examples by projecting logit gradients onto these intruder directions.
- Differential Subspace Attack (`differential_attack.py`): define a private subspace by orthogonalizing target LoRA B against shadow LoRA B, then score gradient alignment with this target-only subspace.
- Curvature-Aware LoRA Attack (`cur_attack.py`): approximate $A^{-1}\Delta$ in LoRA parameter space from aux data and score each example by $\nabla \log P(y|x)^\top (A^{-1}\Delta)$.
