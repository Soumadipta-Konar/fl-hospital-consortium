import os
import warnings
import pandas as pd
import torch
SITE_NAMES = ["cleveland", "hungarian", "switzerland", "va"]
NUM_SITES = 4

def _get_fed_heart_disease_class():
    try:
        from flamby.datasets.fed_heart_disease import FedHeartDisease
        return FedHeartDisease
    except ImportError as e:
        raise ImportError(
            "FLamby is required to load FedHeartDisease data. "
            "Install it via requirements.txt: pip install -r requirements.txt"
        ) from e

def load_site_data(site, train=True):
    FedHeartDisease = _get_fed_heart_disease_class()
    data_dir = "data/fed_heart_disease" if os.path.exists("data/fed_heart_disease") else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        kwargs = {"center": site, "train": train, "pooled": False}
        if data_dir:
            kwargs["data_path"] = data_dir
        ds = FedHeartDisease(**kwargs)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([ds[i][1] for i in range(len(ds))])
    return x, y

def load_all_sites():
    data = {}
    for site in range(NUM_SITES):
        x_tr, y_tr = load_site_data(site, train=True)
        x_te, y_te = load_site_data(site, train=False)
        data[site] = {
            "train": (x_tr, y_tr),
            "test": (x_te, y_te)
        }
    return data

def load_pooled_data(train=True):
    FedHeartDisease = _get_fed_heart_disease_class()
    data_dir = "data/fed_heart_disease" if os.path.exists("data/fed_heart_disease") else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        kwargs = {"train": train, "pooled": True}
        if data_dir:
            kwargs["data_path"] = data_dir
        ds = FedHeartDisease(**kwargs)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([ds[i][1] for i in range(len(ds))])
    return x, y

def get_site_summary():
    rows = []
    for site in range(NUM_SITES):
        x_tr, y_tr = load_site_data(site, train=True)
        x_te, y_te = load_site_data(site, train=False)
        n_tr = len(y_tr)
        n_te = len(y_te)
        pos_tr = int(y_tr.sum().item())
        pos_te = int(y_te.sum().item())
        rows.append({
            "site": site,
            "name": SITE_NAMES[site],
            "n_train": n_tr,
            "n_test": n_te,
            "n_total": n_tr + n_te,
            "pos_train": pos_tr,
            "pos_test": pos_te,
            "prev_train": round(pos_tr / n_tr, 3),
            "prev_test": round(pos_te / n_te, 3),
            "prev_total": round((pos_tr + pos_te) / (n_tr + n_te), 3)
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    summary = get_site_summary()
    print(summary.to_string(index=False))
