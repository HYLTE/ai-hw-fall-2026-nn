"""Train the transformer encoder on MNIST with the paper's optimizer, LR schedule and label smoothing."""
import argparse
import os

import torch
from torch import nn

from data import get_loaders
from model import TransformerClassifier


def noam_lr(step, d_model, warmup):
    """lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)  (step counts from 1)."""
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def run_epoch(model, loader, loss_fn, optimizer=None, scheduler=None):
    """One pass over `loader`; trains if an optimizer is given. Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss = correct = n = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/best.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    train_loader, val_loader, _ = get_loaders(batch_size=args.batch_size, seed=args.seed)

    config = dict(patch=args.patch, d_model=args.d_model, h=args.heads, d_ff=args.d_ff,
                  n_layers=args.layers, dropout=args.dropout)
    model = TransformerClassifier(**config)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Adam(beta1=0.9, beta2=0.98, eps=1e-9); base lr 1.0 so the schedule sets the actual rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: noam_lr(s + 1, args.d_model, args.warmup))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, scheduler)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.2e}  "
              f"train loss {train_loss:.4f} acc {train_acc:.4f}  "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}", flush=True)
        if val_acc > best_acc:  # model selection on validation only; test is untouched
            best_acc = val_acc
            torch.save({"config": config, "state_dict": model.state_dict(),
                        "epoch": epoch, "val_acc": val_acc}, args.out)
    print(f"best val acc {best_acc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
