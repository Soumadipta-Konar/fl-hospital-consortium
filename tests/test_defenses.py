import torch
from src.privacy import (
    aggregate_median,
    aggregate_trimmed_mean,
    aggregate_krum,
    aggregate_multi_krum,
    run_privacy_fl,
)


def test_aggregate_trimmed_mean():
    # 4 clients: 3 benign closely clustered around [1.0, 2.0], 1 extreme Byzantine outlier
    c0 = {"linear.weight": torch.tensor([[1.0, 2.0]]), "linear.bias": torch.tensor([0.1])}
    c1 = {"linear.weight": torch.tensor([[1.1, 2.1]]), "linear.bias": torch.tensor([0.12])}
    c2 = {"linear.weight": torch.tensor([[0.9, 1.9]]), "linear.bias": torch.tensor([0.08])}
    c3 = {"linear.weight": torch.tensor([[1000.0, -500.0]]), "linear.bias": torch.tensor([999.0])}

    aggregated = aggregate_trimmed_mean([c0, c1, c2, c3], beta=1)
    # The extreme values must be trimmed away completely
    assert torch.all(aggregated["linear.weight"] < 2.5)
    assert torch.all(aggregated["linear.weight"] > 0.5)
    assert torch.abs(aggregated["linear.bias"] - 0.1) < 0.05


def test_aggregate_krum():
    # 4 clients: 3 benign closely clustered, 1 Byzantine adversary
    c0 = {"w": torch.tensor([1.0, 2.0])}
    c1 = {"w": torch.tensor([1.05, 2.05])}
    c2 = {"w": torch.tensor([0.95, 1.95])}
    c3 = {"w": torch.tensor([500.0, -500.0])}

    selected = aggregate_krum([c0, c1, c2, c3], f=1)
    # Krum must select one of the benign clients (0, 1, or 2), never the adversary (3)
    assert torch.all(selected["w"] < 5.0)
    assert torch.all(selected["w"] > 0.0)


def test_aggregate_multi_krum():
    c0 = {"w": torch.tensor([1.0, 2.0])}
    c1 = {"w": torch.tensor([1.05, 2.05])}
    c2 = {"w": torch.tensor([0.95, 1.95])}
    c3 = {"w": torch.tensor([500.0, -500.0])}

    avg_selected = aggregate_multi_krum([c0, c1, c2, c3], f=1, m=2)
    assert torch.all(avg_selected["w"] < 5.0)
    assert torch.all(avg_selected["w"] > 0.0)


def test_aggregate_median():
    c0 = {"w": torch.tensor([1.0, 10.0])}
    c1 = {"w": torch.tensor([2.0, 20.0])}
    c2 = {"w": torch.tensor([3.0, 30.0])}

    med = aggregate_median([c0, c1, c2])
    assert torch.allclose(med["w"], torch.tensor([2.0, 20.0]))


def test_run_privacy_fl_with_defenses():
    # Run synthetic federated simulation under Byzantine garbage attack with defenses
    torch.manual_seed(42)
    mock_data = {}
    for sid in range(4):
        x_tr = torch.randn(16, 13)
        y_tr = torch.randint(0, 2, (16, 1)).float()
        x_te = torch.randn(8, 13)
        y_te = torch.randint(0, 2, (8, 1)).float()
        mock_data[sid] = {"train": (x_tr, y_tr), "test": (x_te, y_te)}
    mock_pooled_test = (torch.randn(16, 13), torch.randint(0, 2, (16, 1)).float())

    # Trimmed mean under attack
    recs_tm = run_privacy_fl(
        mock_data, pooled_test=mock_pooled_test, seeds=[42], rounds=2,
        attack_type="garbage", attack_site=0, garbage_scale=50.0,
        aggregation_rule="trimmed_mean", beta=1
    )
    assert len(recs_tm) == 2 * (4 + 1)  # 2 rounds * (4 sites + 1 pooled)

    # Krum under attack
    recs_krum = run_privacy_fl(
        mock_data, pooled_test=mock_pooled_test, seeds=[42], rounds=2,
        attack_type="garbage", attack_site=0, garbage_scale=50.0,
        aggregation_rule="krum", krum_f=1
    )
    assert len(recs_krum) == 2 * (4 + 1)
