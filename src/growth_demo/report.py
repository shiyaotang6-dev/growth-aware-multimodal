"""Small experiment plots; all titles preserve data provenance."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_manifest


def make_report(manifest, run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    with (run / "test_predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    row = rows[0]
    with np.load(row["absolute_path"], allow_pickle=False) as sample:
        axes[0].imshow(sample["rgb"].transpose(1, 2, 0))
        axes[0].set_title("Synthetic RGB input" if metrics["data_source"] == "synthetic" else "RGB input")
        axes[0].axis("off")
        axes[1].plot(sample["hsi"].mean(axis=(1, 2)), color="#227b6c", linewidth=2)
        axes[1].set_title("Mean hyperspectral signal")
        axes[1].set_xlabel("Band index")
        axes[1].set_ylabel("Normalized reflectance")
    counts = [sum(row["nutrition_class"] == label for row in rows) for label in range(5)]
    axes[2].bar(range(5), counts, color="#227b6c")
    axes[2].set_title("Nutrition class distribution")
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("Samples")
    fig.suptitle(f"Data source: {metrics['data_source']} — demonstration inputs", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "data_preview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    epochs = [item["epoch"] for item in history]
    for split, color in (("train", "#227b6c"), ("validation", "#e09142")):
        axes[0].plot(epochs, [item[split]["loss"] for item in history], label=split, color=color)
        axes[1].plot(epochs, [item[split]["accuracy"] for item in history], label=split, color=color)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].set_title("Classification accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    truth = [float(row["spad_true"]) for row in predictions]
    predicted = [float(row["spad_pred"]) for row in predictions]
    axes[2].scatter(truth, predicted, color="#227b6c", alpha=0.75)
    low, high = min(truth + predicted), max(truth + predicted)
    axes[2].plot([low, high], [low, high], "--", color="#777777")
    axes[2].set_title("Held-out SPAD regression")
    axes[2].set_xlabel("Target SPAD")
    axes[2].set_ylabel("Predicted SPAD")
    fig.suptitle(f"Data source: {metrics['data_source']} — software demonstration, not field validation", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "training_report.png", dpi=160)
    plt.close(fig)
    return output
