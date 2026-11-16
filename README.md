Sofia MSCS 2201 - Artificial Intelligence

## Transformer encoder for MNIST

A from-scratch transformer encoder (Vaswani et al., 2017) classifying MNIST digits.
Each image is split into 49 patches of 4x4 pixels, a [CLS] token is prepended, and its
final state is classified.

- `model.py`: attention, multi-head attention, FFN, encoder layer, positional encoding, classifier
- `data.py`: official 60k/10k split, 5k of train held out for validation
- `train.py`: Adam (0.9, 0.98, 1e-9), warmup + inverse-sqrt LR schedule, label smoothing 0.1
- `evaluate.py`: one-time test-set accuracy, confusion matrix, misclassified digits

```sh
uv sync
source .venv/bin/activate
python train.py      # best checkpoint by val accuracy -> runs/best.pt
python evaluate.py   # test accuracy, confusion matrix -> results/results.md
```

Results: [results/results.md](results/results.md)

## Adversarial attacks

White-box, untargeted L∞ attacks on the trained model, run on [0, 1] pixels for ε ∈ {0.1, 0.2, 0.3}.
Metrics: recognition rate before attacks, and attack success rate (ASR), the fraction of correctly
classified test images that the attack makes misclassified.

- `attacks.py`: FGSM, I-FGSM / PGD, MI-FGSM
- `attack_eval.py`: ε sweep over the test set, ASR and accuracy under attack, example images

```sh
python attack_eval.py   # ~40 min on CPU; --n 1000 for a quick run -> results/attacks.md
```

Results: [results/attacks.md](results/attacks.md)
