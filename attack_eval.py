"""Evaluate white-box L-inf attacks on the test set; write results to <out-dir>/attacks.md."""
import argparse
import os

import torch
from torch import nn
from torchvision.transforms import Normalize
from torchvision.utils import save_image

from attacks import fgsm, mifgsm, pgd
from data import MEAN, STD, load_mnist
from model import TransformerClassifier

EPSILONS = (0.1, 0.2, 0.3)  # L-inf budgets on the [0, 1] pixel scale
EXAMPLE_EPS = 0.2  # budget shown in the example figure
ATTACKS = {  # name: (settings, attack(model, x, y, eps) -> x_adv)
    "FGSM": ("1 step",
             lambda m, x, y, eps: fgsm(m, x, y, eps)),
    "I-FGSM": ("10 steps, α = ε/10",
               lambda m, x, y, eps: pgd(m, x, y, eps, alpha=eps / 10, steps=10, random_start=False)),
    "PGD": ("40 steps, α = 2.5ε/40, random start",
            lambda m, x, y, eps: pgd(m, x, y, eps, alpha=2.5 * eps / 40, steps=40)),
    "MI-FGSM": ("10 steps, α = ε/10, μ = 1.0",
                lambda m, x, y, eps: mifgsm(m, x, y, eps, alpha=eps / 10, steps=10, decay=1.0)),
}


@torch.no_grad()
def predict(model, x):
    return torch.cat([model(xb).argmax(dim=1) for xb in x.split(1000)])


def table(metric):
    """Markdown table of metric[attack, eps]: rows = eps, columns = attacks."""
    return ["| ε | " + " | ".join(ATTACKS) + " |",
            "|---" * (len(ATTACKS) + 1) + "|",
            *(f"| {eps} | " + " | ".join(f"{metric[name, eps]:.4f}" for name in ATTACKS) + " |"
              for eps in EPSILONS)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/best.pt")
    p.add_argument("--n", type=int, default=10000, help="attack the first n test images")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)  # PGD random start
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt)
    net = TransformerClassifier(**ckpt["config"])
    net.load_state_dict(ckpt["state_dict"])
    # Normalization becomes the first layer, so attacks work on [0, 1] pixels and eps is in pixel units.
    model = nn.Sequential(Normalize((MEAN,), (STD,)), net).eval().requires_grad_(False)

    x, y = load_mnist("data", train=False, normalize=False).tensors
    x, y = x[:args.n], y[:args.n]
    correct = predict(model, x) == y
    n_correct = correct.sum().item()
    print(f"recognition rate before attacks: {n_correct / len(y):.4f}", flush=True)

    asr, robust_acc, examples = {}, {}, {}
    for name, (_, attack) in ATTACKS.items():
        for eps in EPSILONS:
            x_adv = torch.cat([attack(model, xb, yb, eps)
                               for xb, yb in zip(x.split(args.batch_size), y.split(args.batch_size))])
            assert (x_adv - x).abs().max() <= eps + 1e-6 and x_adv.min() >= 0 and x_adv.max() <= 1
            adv_correct = predict(model, x_adv) == y
            # Success: a correctly classified image becomes misclassified.
            asr[name, eps] = (correct & ~adv_correct).sum().item() / n_correct
            robust_acc[name, eps] = (correct & adv_correct).sum().item() / len(y)
            if eps == EXAMPLE_EPS:
                examples[name] = x_adv[:8]
            print(f"{name:8s} eps {eps:.2f}  ASR {asr[name, eps]:.4f}  "
                  f"acc under attack {robust_acc[name, eps]:.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = {"clean": x[:8], **examples}  # figure rows, top to bottom
    save_image(torch.cat(list(rows.values())), os.path.join(args.out_dir, "adversarial_examples.png"),
               nrow=8, padding=2, pad_value=0.5)

    lines = [
        "# Adversarial attacks on the MNIST transformer",
        "",
        f"- Checkpoint: `{args.ckpt}`, first {len(y)} test images",
        "- Threat model: white-box, untargeted, ‖x_adv − x‖∞ ≤ ε on [0, 1] pixels",
        f"- **Recognition rate before attacks: {n_correct / len(y):.4f}** ({n_correct} / {len(y)})",
        "- Attack success rate (ASR): fraction of correctly classified images that the attack misclassifies",
        "- Accuracy under attack: recognition rate × (1 − ASR)",
        "",
        "| Attack | Settings |",
        "|---|---|",
        *(f"| {name} | {settings} |" for name, (settings, _) in ATTACKS.items()),
        "",
        "## Attack success rate",
        "",
        *table(asr),
        "",
        "## Accuracy under attack",
        "",
        *table(robust_acc),
        "",
        f"## Examples at ε = {EXAMPLE_EPS}",
        "",
        "![adversarial examples](adversarial_examples.png)",
        "",
        f"Image rows, top to bottom: {', '.join(rows)}. Cells are the model's predictions.",
        "",
        "| | " + " | ".join(f"#{i + 1}" for i in range(8)) + " |",
        "|---" * 9 + "|",
        "| true label | " + " | ".join(map(str, y[:8].tolist())) + " |",
        *(f"| {row} | " + " | ".join(map(str, predict(model, imgs).tolist())) + " |"
          for row, imgs in rows.items()),
    ]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "attacks.md"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
