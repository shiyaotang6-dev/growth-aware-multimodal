import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np

from growth_demo.data import MultimodalDataset, generate_synthetic, load_manifest, split_records
from growth_demo.engine import evaluate, train
from growth_demo.model import GrowthAwareMultiTaskNet, ModelConfig


torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def test_fusion_is_differentiable_and_gates_are_probabilities(self):
        torch.manual_seed(42)
        model = GrowthAwareMultiTaskNet(ModelConfig(feature_dim=32, stage_dim=8))
        rgb = torch.rand(2, 3, 32, 32)
        hsi = torch.rand(2, 1, 8, 4, 4)
        out = model(rgb, hsi, torch.tensor([0, 2]))
        self.assertEqual(out["logits"].shape, (2, 5))
        self.assertEqual(out["spad_normalized"].shape, (2,))
        self.assertEqual(out["attention"].shape, (2, 2, 2))
        self.assertTrue(torch.allclose(out["gates"].sum(-1), torch.ones(2)))
        loss = out["logits"].square().mean() + out["spad_normalized"].square().mean()
        loss.backward()
        for parameter in (model.rgb_encoder.conv1.weight, model.hsi_encoder.features[0].weight,
                          model.stage_embedding.weight, model.graph.attention.in_proj_weight, model.gate[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_growth_stage_changes_fusion_for_identical_inputs(self):
        torch.manual_seed(1)
        model = GrowthAwareMultiTaskNet(ModelConfig(feature_dim=32, stage_dim=8)).eval()
        rgb = torch.rand(1, 3, 32, 32).repeat(2, 1, 1, 1)
        hsi = torch.rand(1, 1, 8, 4, 4).repeat(2, 1, 1, 1, 1)
        with torch.no_grad():
            out = model(rgb, hsi, torch.tensor([0, 2]))
        self.assertFalse(torch.allclose(out["gates"][0], out["gates"][1]))


class DataTests(unittest.TestCase):
    def test_synthetic_inputs_and_reproducible_grouped_split(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_synthetic(directory, samples=50, seed=42, bands=8, hsi_size=4)
            rows = load_manifest(manifest)
            splits = split_records(rows)
            self.assertEqual({key: len(value) for key, value in splits.items()}, {"train": 35, "val": 5, "test": 10})
            self.assertEqual(splits, split_records(rows))
            ids = [{row["sample_id"] for row in values} for values in splits.values()]
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            sample = MultimodalDataset(splits["train"])[0]
            self.assertEqual(sample["rgb"].shape, (3, 32, 32))
            self.assertEqual(sample["hsi"].shape, (1, 8, 4, 4))
            self.assertTrue(torch.isfinite(sample["rgb"]).all())

    def test_multiple_visits_of_same_plant_never_cross_splits(self):
        rows = [{"sample_id": f"{label}-{group}-{visit}", "plant_id": f"{label}-{group}", "nutrition_class": label}
                for label in range(5) for group in range(10) for visit in range(2)]
        splits = split_records(rows, seed=2)
        group_sets = [{row["plant_id"] for row in value} for value in splits.values()]
        self.assertFalse(group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
        self.assertEqual(sum(map(len, splits.values())), len(rows))

    def test_duplicate_sample_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_synthetic(directory, samples=15)
            lines = Path(manifest).read_text(encoding="utf-8").splitlines()
            lines.append(lines[1])
            Path(manifest).write_text("\n".join(lines), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_manifest(manifest)

    def test_small_training_run_and_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = generate_synthetic(base / "data", samples=30, bands=8, hsi_size=4)
            result = train(manifest, base / "run", epochs=1, batch_size=8, feature_dim=32, device="cpu")
            again = evaluate(manifest, base / "run" / "best_model.pt", base / "evaluation", device="cpu")
            self.assertEqual(result["data_source"], "synthetic")
            self.assertEqual(result["test_metrics"]["count"], 5)
            self.assertAlmostEqual(result["test_metrics"]["spad_mae"], again["test_metrics"]["spad_mae"], places=4)
            self.assertTrue((base / "run" / "test_predictions.csv").exists())
            sample_path = Path(load_manifest(manifest)[0]["absolute_path"])
            with np.load(sample_path) as sample:
                rgb, hsi = sample["rgb"].copy(), sample["hsi"].copy()
            rgb[0, 0, 0] = 0.99
            np.savez_compressed(sample_path, rgb=rgb, hsi=hsi)
            with self.assertRaisesRegex(ValueError, "sample bytes"):
                evaluate(manifest, base / "run" / "best_model.pt", base / "tampered", device="cpu")


if __name__ == "__main__":
    unittest.main()
