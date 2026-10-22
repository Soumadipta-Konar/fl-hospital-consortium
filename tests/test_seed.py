import numpy as np
import torch
from src.seed import seed_everything

def test_seed_reproducibility():
    seed_everything(42)
    t1 = torch.randn(10, 10)
    n1 = np.random.randn(10)

    seed_everything(42)
    t2 = torch.randn(10, 10)
    n2 = np.random.randn(10)

    assert torch.equal(t1, t2), "PyTorch tensors with same seed must be strictly equal"
    assert np.array_equal(n1, n2), "NumPy arrays with same seed must be strictly equal"

def test_seed_differentiation():
    seed_everything(42)
    t1 = torch.randn(10, 10)

    seed_everything(43)
    t2 = torch.randn(10, 10)

    assert not torch.equal(t1, t2), "Different seeds must produce different tensor draws"
