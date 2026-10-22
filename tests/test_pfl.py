import torch
from src.pfl import (
    PersonalizedMLP,
    train_personalized_ditto,
    train_fedper_client,
    run_pfl,
)
from src.models import Baseline


def test_personalized_mlp_forward():
    model = PersonalizedMLP(input_dim=13, hidden_dim=8, output_dim=1)
    x = torch.randn(10, 13)
    out = model(x)
    assert out.shape == (10, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_train_personalized_ditto():
    torch.manual_seed(42)
    pers_model = Baseline()
    glob_model = Baseline()
    x = torch.randn(20, 13)
    y = torch.randint(0, 2, (20, 1)).float()

    sd_before = {k: v.clone() for k, v in pers_model.state_dict().items()}
    sd_after = train_personalized_ditto(pers_model, glob_model, x, y, epochs=2, lr=0.01, lam=0.5)

    # Weights must update
    assert not torch.equal(sd_before["linear.weight"], sd_after["linear.weight"])


def test_train_fedper():
    torch.manual_seed(42)
    model = PersonalizedMLP()
    x = torch.randn(16, 13)
    y = torch.randint(0, 2, (16, 1)).float()

    sd = train_fedper_client(model, x, y, epochs=1, lr=0.01)
    assert "base.0.weight" in sd
    assert "head.weight" in sd


def test_run_pfl_simulation():
    torch.manual_seed(42)
    mock_data = {}
    for sid in range(4):
        x_tr = torch.randn(16, 13)
        y_tr = torch.randint(0, 2, (16, 1)).float()
        x_te = torch.randn(8, 13)
        y_te = torch.randint(0, 2, (8, 1)).float()
        mock_data[sid] = {"train": (x_tr, y_tr), "test": (x_te, y_te)}
    mock_pooled_test = (torch.randn(16, 13), torch.randint(0, 2, (16, 1)).float())

    # 1. Test Ditto
    recs_ditto = run_pfl(mock_data, mock_pooled_test, seeds=[42], rounds=2, method="ditto", lam=0.1)
    assert len(recs_ditto) == 2 * (4 + 1)
    assert recs_ditto[0]["method"] == "ditto"

    # 2. Test FedPer
    recs_fedper = run_pfl(mock_data, mock_pooled_test, seeds=[42], rounds=2, method="fedper")
    assert len(recs_fedper) == 2 * (4 + 1)
    assert recs_fedper[0]["method"] == "fedper"

    # 3. Test Local Fine-Tuning
    recs_ft = run_pfl(mock_data, mock_pooled_test, seeds=[42], rounds=2, method="finetune", ft_epochs=1)
    assert len(recs_ft) == 2 * (4 + 1)
    assert recs_ft[0]["method"] == "finetune"
