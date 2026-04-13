import argparse
import json
import os
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from puf_model.data import PairDataset, build_pair_records, split_records_by_master
from puf_model.losses import MultiHeadAsymmetricLoss
from puf_model.metrics import binary_metrics, sweep_thresholds
from puf_model.model import SiameseAttentionalPUF


def resolve_existing_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    root_candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_candidate):
        return root_candidate
    return os.path.abspath(path)


def resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiHeadAsymmetricLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    train: bool,
    base_threshold: float,
    wear_gate_alpha: float,
    epoch_idx: int,
    total_epochs: int,
    show_progress: bool,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.train(mode=train)

    losses = []
    tamper_losses = []
    wear_losses = []
    y_true = []
    y_prob = []

    desc = f"Epoch {epoch_idx}/{total_epochs} {'train' if train else 'val'}"
    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False)

    for batch in iterator:
        baseline = batch["baseline"].to(device)
        verification = batch["verification"].to(device)
        tamper_target = batch["tamper_label"].to(device)
        wear_target = batch["wear_label"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        out = model(baseline, verification)
        loss_dict = criterion(
            out["tamper_logits"],
            out["wear_logits"],
            tamper_target,
            wear_target,
        )

        if train:
            loss_dict["loss"].backward()
            optimizer.step()

        tamper_prob = torch.sigmoid(out["tamper_logits"])
        wear_prob = torch.sigmoid(out["wear_logits"])
        gated_prob = tamper_prob - wear_gate_alpha * wear_prob
        gated_prob = torch.clamp(gated_prob, 0.0, 1.0)

        losses.append(float(loss_dict["loss"].item()))
        tamper_losses.append(float(loss_dict["tamper_loss"].item()))
        wear_losses.append(float(loss_dict["wear_loss"].item()))
        y_true.append(tamper_target.detach().cpu().numpy())
        y_prob.append(gated_prob.detach().cpu().numpy())

        if show_progress and tqdm is not None:
            iterator.set_postfix(
                total_loss=f"{np.mean(losses):.4f}",
                tamper_loss=f"{np.mean(tamper_losses):.4f}",
                wear_loss=f"{np.mean(wear_losses):.4f}",
            )

    y_true_np = np.concatenate(y_true, axis=0)
    y_prob_np = np.concatenate(y_prob, axis=0)

    metrics = binary_metrics(y_true_np, y_prob_np, threshold=base_threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["tamper_loss"] = float(np.mean(tamper_losses)) if tamper_losses else 0.0
    metrics["wear_loss"] = float(np.mean(wear_losses)) if wear_losses else 0.0

    return metrics, y_true_np, y_prob_np


def main() -> None:
    parser = argparse.ArgumentParser(description="Train preliminary Siamese-Attentional PUF model")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--paired-dir", default="paired_dataset")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fp-weight", type=float, default=50.0)
    parser.add_argument("--wear-weight", type=float, default=0.5)
    parser.add_argument("--wear-gate-alpha", type=float, default=0.08)
    parser.add_argument("--base-threshold", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    args.dataset_dir = resolve_existing_path(args.dataset_dir)
    args.paired_dir = resolve_existing_path(args.paired_dir)
    args.output_dir = resolve_output_path(args.output_dir)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    records = build_pair_records(args.dataset_dir, args.paired_dir)
    if not records:
        raise RuntimeError("No pair records found. Check dataset and paired directories.")

    train_records, val_records = split_records_by_master(records, train_ratio=0.8, seed=args.seed)
    print(f"Records: total={len(records)}, train={len(train_records)}, val={len(val_records)}")

    train_ds = PairDataset(train_records, image_size=(args.image_size, args.image_size))
    val_ds = PairDataset(val_records, image_size=(args.image_size, args.image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SiameseAttentionalPUF(in_channels=1, base_channels=32, dropout=0.2).to(device)
    criterion = MultiHeadAsymmetricLoss(fp_weight=args.fp_weight, wear_weight=args.wear_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_f01 = -1.0
    best_thr = args.base_threshold
    history = []
    show_progress = not args.no_progress

    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            base_threshold=args.base_threshold,
            wear_gate_alpha=args.wear_gate_alpha,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            show_progress=show_progress,
        )

        with torch.no_grad():
            val_metrics_fixed, y_true, y_prob = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                train=False,
                base_threshold=args.base_threshold,
                wear_gate_alpha=args.wear_gate_alpha,
                epoch_idx=epoch,
                total_epochs=args.epochs,
                show_progress=show_progress,
            )

        sweep_thr, sweep_metrics = sweep_thresholds(y_true, y_prob)

        epoch_log = {
            "epoch": epoch,
            "train": train_metrics,
            "val_fixed": val_metrics_fixed,
            "val_sweep": sweep_metrics,
            "val_sweep_threshold": sweep_thr,
        }
        history.append(epoch_log)

        if sweep_metrics["f01"] > best_f01:
            best_f01 = sweep_metrics["f01"]
            best_thr = sweep_thr
            best_state = {
                "model": model.state_dict(),
                "config": vars(args),
                "threshold": best_thr,
            }

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(tamper={train_metrics['tamper_loss']:.4f}, wear={train_metrics['wear_loss']:.4f}) "
            f"val_f01@0.98={val_metrics_fixed['f01']:.4f} "
            f"val_f01@best={sweep_metrics['f01']:.4f} "
            f"best_thr={sweep_thr:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint.")

    ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save(best_state, ckpt_path)

    hist_path = os.path.join(args.output_dir, "train_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved best model: {ckpt_path}")
    print(f"Saved history: {hist_path}")
    print(f"Best F0.1: {best_f01:.4f} at threshold={best_thr:.4f}")


if __name__ == "__main__":
    main()
