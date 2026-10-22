import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import flwr as fl
from src.models import Baseline, get_loss_fn, evaluate_model
from src.data import load_site_data

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, site, lr=0.001, batch_size=4, local_epochs=1):
        self.site = site
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.model = Baseline()
        self.loss_fn = get_loss_fn()
        self.x_tr, self.y_tr = load_site_data(site, train=True)
        self.x_te, self.y_te = load_site_data(site, train=False)

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, w in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(w).float()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        indices = torch.randperm(len(self.x_tr))

        total_loss = 0.0
        for _ in range(self.local_epochs):
            for i in range(0, len(self.x_tr), self.batch_size):
                b_idx = indices[i:i + self.batch_size]
                xb, yb = self.x_tr[b_idx], self.y_tr[b_idx]
                optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(b_idx)

        avg_loss = total_loss / len(self.x_tr)
        return self.get_parameters(config), len(self.x_tr), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc, test_loss = evaluate_model(self.model, self.x_te, self.y_te, self.loss_fn)
        return test_loss, len(self.x_te), {"accuracy": acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=int, required=True)
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    args = parser.parse_args()

    client = HospitalClient(
        site=args.site,
        lr=args.lr,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs
    )
    print(f"Starting Flower client for site {args.site} connecting to {args.server_address}...")
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
        max_retries=1
    )

if __name__ == "__main__":
    main()
