from typing import Dict

import numpy as np
import torch


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    baseline: torch.Tensor,
    verification: torch.Tensor,
    passes: int = 10,
    base_tamper_threshold: float = 0.98,
    wear_gate_alpha: float = 0.08,
    uncertainty_threshold: float = 0.02,
    device: str = "cpu",
) -> Dict[str, float]:
    model = model.to(device)
    baseline = baseline.to(device)
    verification = verification.to(device)

    model.train()

    tamper_probs = []
    wear_probs = []
    for _ in range(passes):
        out = model(baseline, verification)
        tamper_probs.append(torch.sigmoid(out["tamper_logits"]).detach().cpu().numpy())
        wear_probs.append(torch.sigmoid(out["wear_logits"]).detach().cpu().numpy())

    t = np.stack(tamper_probs, axis=0)
    w = np.stack(wear_probs, axis=0)

    tamper_mean = float(t.mean())
    wear_mean = float(w.mean())
    tamper_var = float(t.var())

    gated_threshold = min(0.999, base_tamper_threshold + wear_gate_alpha * wear_mean)

    if tamper_var > uncertainty_threshold:
        decision = "inconclusive"
    else:
        decision = "tampered" if tamper_mean >= gated_threshold else "authentic"

    model.eval()

    return {
        "tamper_prob": tamper_mean,
        "wear_prob": wear_mean,
        "tamper_variance": tamper_var,
        "gated_threshold": gated_threshold,
        "decision": decision,
    }
