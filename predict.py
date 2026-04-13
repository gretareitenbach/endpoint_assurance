import argparse
import os
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from puf_model.inference import mc_dropout_predict
from puf_model.model import SiameseAttentionalPUF


def resolve_existing_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    root_candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_candidate):
        return root_candidate
    return os.path.abspath(path)


def prep_image(path: str, image_size: int) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = (img - img.mean()) / max(1.0, img.std())
    img = np.clip(img, -4.0, 4.0)
    ten = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    return ten


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict authenticity with MC-dropout uncertainty")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mc-passes", type=int, default=10)
    parser.add_argument("--wear-gate-alpha", type=float, default=0.08)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.02)
    args = parser.parse_args()

    args.checkpoint = resolve_existing_path(args.checkpoint)
    args.baseline = resolve_existing_path(args.baseline)
    args.verification = resolve_existing_path(args.verification)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = SiameseAttentionalPUF(in_channels=1, base_channels=32, dropout=0.2)
    model.load_state_dict(ckpt["model"])

    baseline = prep_image(args.baseline, args.image_size)
    verification = prep_image(args.verification, args.image_size)

    out = mc_dropout_predict(
        model=model,
        baseline=baseline,
        verification=verification,
        passes=args.mc_passes,
        base_tamper_threshold=float(ckpt.get("threshold", 0.98)),
        wear_gate_alpha=args.wear_gate_alpha,
        uncertainty_threshold=args.uncertainty_threshold,
        device=device,
    )

    print("Decision:", out["decision"])
    print(f"Tamper probability: {out['tamper_prob']:.4f}")
    print(f"Wear probability: {out['wear_prob']:.4f}")
    print(f"Tamper variance (MC): {out['tamper_variance']:.6f}")
    print(f"Gated threshold: {out['gated_threshold']:.4f}")


if __name__ == "__main__":
    main()
