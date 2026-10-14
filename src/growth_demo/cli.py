import argparse
import json

import torch

from .data import generate_synthetic
from .engine import evaluate, train


def main():
    parser = argparse.ArgumentParser(prog="growth-demo", description="Growth-stage-aware RGB/HSI multi-task research demo")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count; use before the subcommand")
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("make-synthetic", help="Generate clearly labeled synthetic inputs and targets")
    synthetic.add_argument("--output", default="data/synthetic")
    synthetic.add_argument("--samples", type=int, default=150)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--rgb-size", type=int, default=32)
    synthetic.add_argument("--bands", type=int, default=16)
    synthetic.add_argument("--hsi-size", type=int, default=8)
    fit = commands.add_parser("train", help="Train with a plant-grouped 70/10/20 split")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--output", default="runs/demo")
    fit.add_argument("--epochs", type=int, default=5)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--learning-rate", type=float, default=1e-3)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string")
    fit.add_argument("--feature-dim", type=int, default=64)
    fit.add_argument("--pretrained-rgb", action="store_true", help="Use torchvision ImageNet ResNet18 weights; may download")
    test = commands.add_parser("evaluate", help="Re-score the checkpoint's frozen test plants")
    test.add_argument("--manifest", required=True)
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--output", default="runs/evaluation")
    test.add_argument("--batch-size", type=int, default=16)
    test.add_argument("--seed", type=int, default=42)
    test.add_argument("--device", default="auto")
    report = commands.add_parser("report", help="Plot inputs, learning curves, and held-out regression")
    report.add_argument("--manifest", required=True)
    report.add_argument("--run", required=True)
    report.add_argument("--output", default="runs/report")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >=1")
    torch.set_num_threads(args.threads)
    if args.command == "make-synthetic":
        path = generate_synthetic(args.output, args.samples, args.seed, args.rgb_size, args.bands, args.hsi_size)
        print(json.dumps({"manifest": str(path.resolve()), "samples": args.samples, "source": "synthetic"}, ensure_ascii=False))
    elif args.command == "train":
        train(args.manifest, args.output, args.epochs, args.batch_size, args.learning_rate,
              args.seed, args.device, args.feature_dim, args.pretrained_rgb)
    elif args.command == "evaluate":
        evaluate(args.manifest, args.checkpoint, args.output, args.batch_size, args.seed, args.device)
    else:
        from .report import make_report
        print(make_report(args.manifest, args.run, args.output).resolve())


if __name__ == "__main__":
    main()
