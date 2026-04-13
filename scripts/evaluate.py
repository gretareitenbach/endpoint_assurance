import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from puf_model.data import PairDataset, build_pair_records, split_records_by_master
from puf_model.metrics import binary_metrics
from puf_model.model import SiameseAttentionalPUF


def resolve_existing_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    root_candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_candidate):
        return root_candidate
    return os.path.abspath(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained PUF model")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--paired-dir", default="paired_dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--wear-gate-alpha", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.checkpoint = resolve_existing_path(args.checkpoint)
    args.dataset_dir = resolve_existing_path(args.dataset_dir)
    args.paired_dir = resolve_existing_path(args.paired_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    threshold = float(ckpt.get("threshold", 0.98))

    model = SiameseAttentionalPUF(in_channels=1, base_channels=32, dropout=0.2).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    records = build_pair_records(args.dataset_dir, args.paired_dir)
    _, val_records = split_records_by_master(records, train_ratio=0.8, seed=args.seed)
    ds = PairDataset(val_records, image_size=(args.image_size, args.image_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    y_true = []
    y_prob = []

    with torch.no_grad():
        for batch in loader:
            baseline = batch["baseline"].to(device)
            verification = batch["verification"].to(device)
            target = batch["tamper_label"].numpy()

            out = model(baseline, verification)
            tamper = torch.sigmoid(out["tamper_logits"])
            wear = torch.sigmoid(out["wear_logits"])
            gated = torch.clamp(tamper - args.wear_gate_alpha * wear, 0.0, 1.0)

            y_true.append(target)
            y_prob.append(gated.cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0)
    y_prob_np = np.concatenate(y_prob, axis=0)

    metrics = binary_metrics(y_true_np, y_prob_np, threshold)
    metrics["threshold"] = threshold

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
