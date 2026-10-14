"""Synthetic data and a small, explicit NPZ/CSV dataset contract."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def generate_synthetic(output, samples=150, seed=42, rgb_size=32, bands=16, hsi_size=8):
    if samples < 15 or rgb_size < 32 or bands < 4 or hsi_size < 4:
        raise ValueError("Need >=15 samples, RGB size >=32, >=4 bands, HSI size >=4")
    output = Path(output)
    if (output / "manifest.csv").exists():
        raise FileExistsError(f"Dataset already exists: {output}. Use a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "arrays"
    arrays.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    wavelength = np.linspace(0, 1, bands, dtype=np.float32)
    yy, xx = np.mgrid[-1:1:complex(rgb_size), -1:1:complex(rgb_size)]
    leaf_mask = ((xx * 0.75) ** 2 + yy**2) < 0.65
    for index in range(samples):
        label = index % 5
        stage = int(rng.integers(0, 3))
        spad = float(np.clip(15 + label * 9 + rng.normal(0, 1.3), 5, 65))
        signal = (spad - 15) / 40
        rgb = rng.normal(0.18, 0.025, (3, rgb_size, rgb_size)).astype(np.float32)
        colors = [0.43 - 0.18 * signal, 0.40 + 0.30 * signal, 0.20 - 0.09 * signal]
        for channel in range(3):
            rgb[channel, leaf_mask] = colors[channel] + rng.normal(0, 0.025, leaf_mask.sum())
        rgb += (stage - 1) * 0.015
        spectrum = 0.20 + 0.15 * wavelength + signal * (0.15 + 0.25 / (1 + np.exp(-20 * (wavelength - 0.65))))
        spectrum += (stage - 1) * 0.025 * np.sin(wavelength * np.pi)
        hsi = spectrum[:, None, None] + rng.normal(0, 0.015, (bands, hsi_size, hsi_size))
        sample_id = f"synthetic-{index:05d}"
        relative = f"arrays/{sample_id}.npz"
        np.savez_compressed(output / relative, rgb=np.clip(rgb, 0, 1), hsi=np.clip(hsi, 0, 1).astype(np.float32))
        records.append({"sample_id": sample_id, "plant_id": sample_id, "array_path": relative,
                        "growth_stage": stage, "nutrition_class": label, "spad": round(spad, 4)})
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    info = {"source": "synthetic", "seed": seed, "samples": samples, "rgb_size": rgb_size,
            "bands": bands, "hsi_size": hsi_size, "warning": "Not field measurements; metrics do not validate agronomic performance."}
    (output / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return output / "manifest.csv"


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "plant_id", "array_path", "growth_stage", "nutrition_class", "spad"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain records and columns: {sorted(required)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("sample_id must be unique")
    seen_paths = set()
    for row in rows:
        row["growth_stage"] = int(row["growth_stage"])
        row["nutrition_class"] = int(row["nutrition_class"])
        row["spad"] = float(row["spad"])
        if not row["plant_id"] or not 0 <= row["growth_stage"] < 3 or not 0 <= row["nutrition_class"] < 5 or not np.isfinite(row["spad"]):
            raise ValueError(f"Invalid target or plant_id for {row['sample_id']}")
        array_path = (path.parent / row["array_path"]).resolve()
        if array_path in seen_paths:
            raise ValueError("Each record must refer to a distinct NPZ file")
        seen_paths.add(array_path)
        row["absolute_path"] = str(array_path)
    return rows


def split_records(records, seed=42):
    # Group first so repeated visits of one plant cannot cross dataset boundaries.
    groups = defaultdict(list)
    for row in records:
        groups[row["plant_id"]].append(row)
    strata = defaultdict(list)
    for group_id, rows in groups.items():
        counts = Counter(row["nutrition_class"] for row in rows)
        dominant = min(counts, key=lambda item: (-counts[item], item))
        strata[dominant].append(group_id)
    rng = np.random.default_rng(seed)
    result = {"train": [], "val": [], "test": []}
    for label in sorted(strata):
        group_ids = sorted(strata[label])
        if len(group_ids) < 3:
            raise ValueError(f"Class {label} needs at least 3 independent plant groups for a 3-way split")
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_val = max(1, int(round(n * 0.1)))
        n_test = max(1, int(round(n * 0.2)))
        n_train = n - n_val - n_test
        for name, ids in (("train", group_ids[:n_train]), ("val", group_ids[n_train:n_train+n_val]), ("test", group_ids[n_train+n_val:])):
            result[name].extend(row for group in ids for row in groups[group])
    return result


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_digest(manifest, records):
    """Fingerprint both metadata and sample bytes so test inputs cannot silently change."""
    digest = hashlib.sha256(Path(manifest).read_bytes())
    for row in sorted(records, key=lambda item: item["sample_id"]):
        digest.update(row["sample_id"].encode("utf-8"))
        with Path(row["absolute_path"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class MultimodalDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with np.load(row["absolute_path"], allow_pickle=False) as payload:
            rgb = np.array(payload["rgb"], dtype=np.float32, copy=True)
            hsi = np.array(payload["hsi"], dtype=np.float32, copy=True)
        if rgb.ndim != 3 or rgb.shape[0] != 3 or min(rgb.shape[1:]) < 32 or hsi.ndim != 3 or min(hsi.shape) < 4:
            raise ValueError(f"Expected RGB [3,H>=32,W>=32], HSI [bands>=4,H>=4,W>=4]: {row['sample_id']}")
        if not np.isfinite(rgb).all() or not np.isfinite(hsi).all() or rgb.min() < 0 or rgb.max() > 1 or hsi.min() < 0 or hsi.max() > 1:
            raise ValueError("Input arrays must be finite and calibrated to [0,1]")
        return {"rgb": (torch.from_numpy(rgb) - IMAGENET_MEAN) / IMAGENET_STD,
                "hsi": torch.from_numpy(hsi).unsqueeze(0), "stage": torch.tensor(row["growth_stage"]),
                "label": torch.tensor(row["nutrition_class"]), "spad": torch.tensor(row["spad"], dtype=torch.float32),
                "sample_id": row["sample_id"]}
