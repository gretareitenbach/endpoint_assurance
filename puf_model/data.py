import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PairRecord:
    baseline_path: str
    verification_path: str
    tamper_label: int
    wear_label: int
    master_id: str


def _parse_master_id(filename: str) -> str:
    stem = os.path.basename(filename).replace(".png", "")
    parts = stem.split("_")
    return "_".join(parts[:3])


def _labels_from_name(filename: str) -> Tuple[int, int]:
    name = os.path.basename(filename)
    if "_thermal_tamper_" in name or "_cut_" in name:
        return 1, 0
    if "_positive_heat_" in name:
        return 0, 1
    if "_positive_" in name:
        return 0, 0
    raise ValueError(f"Unrecognized sample type in filename: {filename}")


def build_pair_records(dataset_dir: str, paired_dir: str) -> List[PairRecord]:
    baseline_map = {
        os.path.basename(path).replace(".png", ""): path
        for path in glob.glob(os.path.join(dataset_dir, "puf_master_*.png"))
    }

    pair_files = glob.glob(os.path.join(paired_dir, "puf_master_*.png"))
    records: List[PairRecord] = []

    for p in pair_files:
        master_id = _parse_master_id(p)
        baseline_path = baseline_map.get(master_id)
        if baseline_path is None:
            continue

        tamper_label, wear_label = _labels_from_name(p)
        records.append(
            PairRecord(
                baseline_path=baseline_path,
                verification_path=p,
                tamper_label=tamper_label,
                wear_label=wear_label,
                master_id=master_id,
            )
        )

    return records


def split_records_by_master(
    records: Sequence[PairRecord],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[PairRecord], List[PairRecord]]:
    unique_ids = sorted({r.master_id for r in records})
    rnd = random.Random(seed)
    rnd.shuffle(unique_ids)

    cutoff = max(1, int(len(unique_ids) * train_ratio))
    train_ids = set(unique_ids[:cutoff])

    train = [r for r in records if r.master_id in train_ids]
    val = [r for r in records if r.master_id not in train_ids]

    if not val:
        val = train[-max(1, len(train) // 5) :]
        train = train[: -len(val)]

    return train, val


class PairDataset(Dataset):
    def __init__(self, records: Sequence[PairRecord], image_size: Tuple[int, int] = (224, 224)):
        self.records = list(records)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_gray(path: str, image_size: Tuple[int, int]) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.resize(img, image_size[::-1], interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)

    @staticmethod
    def _adaptive_zscore(img: np.ndarray) -> np.ndarray:
        mean = float(img.mean())
        std = float(img.std())
        std = max(std, 1.0)
        norm = (img - mean) / std
        return np.clip(norm, -4.0, 4.0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        dep = self._read_gray(rec.baseline_path, self.image_size)
        ver = self._read_gray(rec.verification_path, self.image_size)

        dep = self._adaptive_zscore(dep)
        ver = self._adaptive_zscore(ver)

        dep_t = torch.from_numpy(dep).unsqueeze(0).float()
        ver_t = torch.from_numpy(ver).unsqueeze(0).float()

        return {
            "baseline": dep_t,
            "verification": ver_t,
            "tamper_label": torch.tensor(rec.tamper_label, dtype=torch.float32),
            "wear_label": torch.tensor(rec.wear_label, dtype=torch.float32),
            "master_id": rec.master_id,
            "verification_path": rec.verification_path,
        }
