"""Training and evaluation helpers with train-only target normalization."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import MultimodalDataset, dataset_digest, manifest_digest, split_records
from .model import GrowthAwareMultiTaskNet, ModelConfig


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(records, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    drop_last = shuffle and len(records) % batch_size == 1
    return DataLoader(MultimodalDataset(records), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, drop_last=drop_last, num_workers=0)


def _epoch(model, loader, device, optimizer=None, target_mean=None, target_std=None):
    training = optimizer is not None
    model.train(training)
    classification_loss = nn.CrossEntropyLoss()
    regression_loss = nn.SmoothL1Loss()
    total_loss = 0.0
    all_labels, all_preds, all_spad, all_spad_preds, all_gates = [], [], [], [], []
    ids = []
    for batch in loader:
        rgb, hsi = batch["rgb"].to(device), batch["hsi"].to(device)
        stage = batch["stage"].to(device)
        labels = batch["label"].to(device)
        spad = batch["spad"].to(device)
        normalized_target = (spad - target_mean) / target_std
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(rgb, hsi, stage)
            loss_cls = classification_loss(output["logits"], labels)
            loss_reg = regression_loss(output["spad_normalized"], normalized_target)
            loss = loss_cls + loss_reg
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        size = labels.shape[0]
        total_loss += float(loss.detach()) * size
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(output["logits"].argmax(dim=-1).detach().cpu().tolist())
        all_spad.extend(spad.cpu().tolist())
        all_spad_preds.extend((output["spad_normalized"] * target_std + target_mean).detach().cpu().tolist())
        all_gates.extend(output["gates"].detach().cpu().tolist())
        ids.extend(batch["sample_id"])
    labels = np.asarray(all_labels)
    predictions = np.asarray(all_preds)
    truth = np.asarray(all_spad)
    predicted = np.asarray(all_spad_preds)
    mae = float(np.mean(np.abs(truth - predicted)))
    rmse = float(np.sqrt(np.mean(np.square(truth - predicted))))
    denominator = float(np.square(truth - truth.mean()).sum())
    r2 = float(1.0 - np.square(truth - predicted).sum() / denominator) if denominator else 0.0
    return {"loss": total_loss / len(labels), "accuracy": float(np.mean(labels == predictions)),
            "spad_mae": mae, "spad_rmse": rmse, "spad_r2": r2,
            "mean_rgb_gate": float(np.mean(np.asarray(all_gates)[:, 0])),
            "count": len(labels), "sample_ids": ids, "labels": labels.tolist(),
            "predictions": predictions.tolist(), "spad_true": truth.tolist(), "spad_pred": predicted.tolist()}


def _write_predictions(path, metrics):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "nutrition_class_true", "nutrition_class_pred", "spad_true", "spad_pred"])
        writer.writerows(zip(metrics["sample_ids"], metrics["labels"], metrics["predictions"], metrics["spad_true"], metrics["spad_pred"]))


def train(manifest, output_dir, epochs=5, batch_size=16, learning_rate=1e-3, seed=42, device="auto", feature_dim=64, pretrained_rgb=False):
    if epochs < 1 or batch_size < 2:
        raise ValueError("epochs must be >=1 and batch_size >=2")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    seed_everything(seed)
    from .data import load_manifest
    rows = load_manifest(manifest)
    data_fingerprint = dataset_digest(manifest, rows)
    splits = split_records(rows, seed=seed)
    plant_sets = {key: {row["plant_id"] for row in values} for key, values in splits.items()}
    if plant_sets["train"] & plant_sets["val"] or plant_sets["train"] & plant_sets["test"] or plant_sets["val"] & plant_sets["test"]:
        raise RuntimeError("Group leakage detected between dataset splits")
    if len(splits["train"]) < 2 or not splits["val"] or not splits["test"]:
        raise ValueError("Need enough independent plant groups to populate train/validation/test")
    train_spad = np.asarray([row["spad"] for row in splits["train"]], dtype=np.float32)
    target_mean = torch.tensor(float(train_spad.mean()), device=device)
    target_std = torch.tensor(max(float(train_spad.std()), 1e-6), device=device)
    config = ModelConfig(feature_dim=feature_dim, attention_heads=4, pretrained_rgb=pretrained_rgb)
    model = GrowthAwareMultiTaskNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_loader = _loader(splits["train"], batch_size, True, seed)
    val_loader = _loader(splits["val"], batch_size, False, seed)
    history = []
    best_score = -float("inf")
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, device, optimizer, target_mean, target_std)
        val_metrics = _epoch(model, val_loader, device, target_mean=target_mean, target_std=target_std)
        score = val_metrics["accuracy"] - val_metrics["spad_mae"] / max(float(target_std), 1.0) * 0.1
        entry = {"epoch": epoch, "train": {key: train_metrics[key] for key in ("loss", "accuracy", "spad_mae", "spad_r2")},
                 "validation": {key: val_metrics[key] for key in ("loss", "accuracy", "spad_mae", "spad_r2")}}
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "model_config": config.to_dict(),
                        "target_mean": float(target_mean), "target_std": float(target_std),
                        "manifest_sha256": manifest_digest(manifest), "split_seed": seed,
                        "dataset_sha256": data_fingerprint,
                        "split_ids": {key: [row["sample_id"] for row in values] for key, values in splits.items()},
                        "split_plant_ids": {key: sorted(value) for key, value in plant_sets.items()},
                        "history": history, "data_source": _dataset_source(manifest)}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = _epoch(model, _loader(splits["test"], batch_size, False, seed), device,
                          target_mean=target_mean, target_std=target_std)
    summary = {"data_source": _dataset_source(manifest), "manifest_sha256": checkpoint["manifest_sha256"],
               "dataset_sha256": data_fingerprint,
               "seed": seed, "device": str(device), "split_counts": {key: len(value) for key, value in splits.items()},
               "test_metrics": {key: test_metrics[key] for key in ("loss", "accuracy", "spad_mae", "spad_rmse", "spad_r2", "mean_rgb_gate", "count")},
               "interpretation": "Synthetic-data metrics are a software smoke check and do not establish field performance." if _dataset_source(manifest) == "synthetic" else "Validate independently before making scientific or deployment claims."}
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_predictions(output_dir / "test_predictions.csv", test_metrics)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _dataset_source(manifest):
    info_path = Path(manifest).resolve().parent / "dataset_info.json"
    if info_path.exists():
        try:
            return json.loads(info_path.read_text(encoding="utf-8")).get("source", "external")
        except (ValueError, OSError):
            pass
    return "external"


def evaluate(manifest, checkpoint_path, output_dir, batch_size=16, seed=42, device="auto"):
    from .data import load_manifest
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["manifest_sha256"] != manifest_digest(manifest):
        raise ValueError("Dataset manifest differs from the one used to create this checkpoint")
    config = ModelConfig(**checkpoint["model_config"])
    config.pretrained_rgb = False  # All weights are in the checkpoint; no download is needed for evaluation.
    model = GrowthAwareMultiTaskNet(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    rows = load_manifest(manifest)
    if checkpoint["dataset_sha256"] != dataset_digest(manifest, rows):
        raise ValueError("Dataset sample bytes differ from the ones used to create this checkpoint")
    ids = set(checkpoint["split_ids"]["test"])
    test_rows = [row for row in rows if row["sample_id"] in ids]
    if len(test_rows) != len(ids):
        raise ValueError("Checkpoint test IDs are missing from this manifest")
    metrics = _epoch(model, _loader(test_rows, batch_size, False, seed), device,
                     target_mean=torch.tensor(checkpoint["target_mean"], device=device),
                     target_std=torch.tensor(checkpoint["target_std"], device=device))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"data_source": _dataset_source(manifest), "manifest_sha256": checkpoint["manifest_sha256"],
               "dataset_sha256": checkpoint["dataset_sha256"],
               "device": str(device), "test_metrics": {key: metrics[key] for key in ("loss", "accuracy", "spad_mae", "spad_rmse", "spad_r2", "mean_rgb_gate", "count")}}
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_predictions(output_dir / "test_predictions.csv", metrics)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
