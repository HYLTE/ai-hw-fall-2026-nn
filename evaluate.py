"""Evaluate the best checkpoint once on the official 10k test set; write results to <out-dir>/results.md."""
import argparse
import os

import torch
from torchvision.utils import save_image

from data import MEAN, STD, load_mnist
from model import TransformerClassifier


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/best.pt")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt)
    model = TransformerClassifier(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test = load_mnist("data", train=False)
    x, y = test.tensors
    pred = torch.cat([model(xb).argmax(dim=1) for xb in x.split(1000)])
    errors = (pred != y).sum().item()
    # Confusion matrix: rows = true digit, columns = predicted digit.
    confusion = torch.bincount(y * 10 + pred, minlength=100).view(10, 10)

    os.makedirs(args.out_dir, exist_ok=True)
    wrong = (pred != y).nonzero().squeeze(1)[:64]
    save_image(x[wrong] * STD + MEAN, os.path.join(args.out_dir, "misclassified.png"),
               nrow=8, padding=2, pad_value=0.5)

    lines = [
        "# MNIST test results",
        "",
        f"- Checkpoint: `{args.ckpt}` (epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f})",
        "- Model: " + ", ".join(f"{k}={v}" for k, v in ckpt["config"].items()),
        f"- **Test accuracy: {1 - errors / len(y):.4f}** ({errors} errors / {len(y)})",
        "",
        "## Confusion matrix (rows = true, columns = predicted)",
        "",
        "| true \\ pred | " + " | ".join(map(str, range(10))) + " |",
        "|---" * 11 + "|",
        *(f"| **{d}** | " + " | ".join(map(str, confusion[d].tolist())) + " |" for d in range(10)),
        "",
        f"## First {len(wrong)} misclassified test digits",
        "",
        "![misclassified](misclassified.png)",
    ]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
