"""MNIST: official 60k train / 10k test split; 5k of train held out for validation."""
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

MEAN, STD = 0.1307, 0.3081  # pixel statistics of the MNIST training set


def load_mnist(root, train):
    """Whole split as normalized tensors: x (N, 1, 28, 28) float, y (N,) int64."""
    ds = datasets.MNIST(root, train=train, download=True)
    x = (ds.data.float() / 255 - MEAN) / STD
    return TensorDataset(x.unsqueeze(1), ds.targets)


def get_loaders(root="data", batch_size=128, val_size=5000, seed=0):
    full_train = load_mnist(root, train=True)
    train, val = random_split(full_train, [len(full_train) - val_size, val_size],
                              generator=torch.Generator().manual_seed(seed))
    test = load_mnist(root, train=False)
    return (DataLoader(train, batch_size, shuffle=True),
            DataLoader(val, 1000),
            DataLoader(test, 1000))
