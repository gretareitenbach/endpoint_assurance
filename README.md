# endpoint_assurance

Tamper detection and authenticity verification for adhesive-tape Physical Unclonable Functions (PUFs) using a Siamese deep learning model.

This repository contains:
- Synthetic PUF data generation
- Pair construction for authentic vs tampered comparisons
- A Siamese attentional model with pairwise alignment (STN)
- Training, evaluation, and prediction scripts
- Saved artifacts (`best_model.pt`, training history)

## Overview

The system compares:
- A baseline (master) scan of a tape strip
- A verification scan of the same claimed strip

The model outputs:
- Tamper probability
- Wear probability
- Final authenticity/tamper decision with uncertainty handling

The design prioritizes low false positives by:
- Using an asymmetric tamper loss that heavily penalizes false alarms
- Applying a wear-aware gate to avoid misclassifying natural aging as tampering

## Repository Layout

```text
endpoint_assurance/
	artifacts/
		best_model.pt
		train_history.json
	data_generation/
		create_puf.py
		augment_puf.py
	dataset/
	paired_dataset/
	puf_model/
		data.py
		model.py
		losses.py
		metrics.py
		inference.py
	scripts/
		train.py
		evaluate.py
		predict.py
	requirements.txt
	model_architecture.md
```

## Environment Setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Data Pipeline

### Step 1: Generate master PUF images

```bash
python data_generation/create_puf.py
```

Creates grayscale master images in `dataset/` with names like:

```text
puf_master_0000.png
```

### Step 2: Generate paired verification samples

```bash
python data_generation/augment_puf.py
```

Creates verification samples in `paired_dataset/` with label-encoded filenames:
- `*_positive_*.png` -> authentic, no wear (`tamper=0`, `wear=0`)
- `*_positive_heat_*.png` -> authentic, worn (`tamper=0`, `wear=1`)
- `*_thermal_tamper_*.png` -> tampered (`tamper=1`, `wear=0`)
- `*_cut_*.png` -> tampered (`tamper=1`, `wear=0`)

These names are parsed directly by `puf_model/data.py` to produce labels.

## Training

Run training from the repository root:

```bash
python scripts/train.py
```

Useful options:

```bash
python scripts/train.py \
	--dataset-dir dataset \
	--paired-dir paired_dataset \
	--output-dir artifacts \
	--epochs 30 \
	--batch-size 16 \
	--lr 1e-3 \
	--fp-weight 50.0 \
	--wear-weight 0.5 \
	--wear-gate-alpha 0.08 \
	--base-threshold 0.98
```

Outputs:
- `artifacts/best_model.pt` (best checkpoint by validation $F_{0.1}$ after threshold sweep)
- `artifacts/train_history.json` (epoch metrics and sweep results)

## Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py \
	--checkpoint artifacts/best_model.pt \
	--dataset-dir dataset \
	--paired-dir paired_dataset
```

The script prints JSON metrics including precision, recall, accuracy, and $F_{0.1}$ at the saved threshold.

## Prediction (Single Pair)

Predict authenticity for one baseline/verification pair:

```bash
python scripts/predict.py \
	--checkpoint artifacts/best_model.pt \
	--baseline dataset/puf_master_0000.png \
	--verification paired_dataset/puf_master_0000_positive_00.png
```

Optional uncertainty controls:

```bash
python scripts/predict.py \
	--checkpoint artifacts/best_model.pt \
	--baseline <path_to_master> \
	--verification <path_to_query> \
	--mc-passes 10 \
	--wear-gate-alpha 0.08 \
	--uncertainty-threshold 0.02
```

Decision logic:
- Compute mean tamper probability via MC-dropout
- Compute mean wear probability
- Shift threshold with wear gate: `gated_threshold = base_threshold + wear_gate_alpha * wear_prob`
- If variance is high, return `inconclusive`; otherwise classify as `authentic` or `tampered`

## Current Artifacts

The repository already includes:
- `artifacts/best_model.pt`
- `artifacts/train_history.json`

You can run `scripts/evaluate.py` and `scripts/predict.py` immediately with those files.

## Model Notes

The implemented model (`SiameseAttentionalPUF`) includes:
- Pairwise affine alignment module (STN) for verification scan normalization
- Shared residual-dilated encoder with squeeze-and-excitation attention
- Differential fusion of baseline and verification features
- Dual heads for tamper and wear logits

See `model_architecture.md` for a higher-level design discussion.

## Troubleshooting

- `No pair records found`: ensure both `dataset/` and `paired_dataset/` exist and contain `puf_master_*.png` files.
- OpenCV read failures: verify file paths and image extensions are `.png`.
- CUDA issues: training and inference fall back to CPU automatically.
- PowerShell activation blocked: run `Set-ExecutionPolicy -Scope Process Bypass` in the current shell.

## License

This project is distributed under the terms in `LICENSE`.