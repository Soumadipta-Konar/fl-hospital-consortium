import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def check_seed_determinism(seed=42):
    seed_everything(seed)
    r1 = [random.random(), np.random.rand(), float(torch.rand(1).item())]
    seed_everything(seed)
    r2 = [random.random(), np.random.rand(), float(torch.rand(1).item())]
    return r1 == r2, r1, r2

if __name__ == "__main__":
    ok, r1, r2 = check_seed_determinism(123)
    print(f"seed check: {ok}")
    print(f"draw 1: {r1}")
    print(f"draw 2: {r2}")
