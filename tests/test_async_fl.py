import torch
from src.async_fl import (
    staleness_polynomial,
    staleness_hinge,
    staleness_constant,
    compute_staleness_weight,
    async_server_update,
    run_async_fl,
)


def test_staleness_functions():
    # Fresh update (tau = 0) must have weight 1.0
    assert staleness_polynomial(0) == 1.0
    assert staleness_hinge(0) == 1.0
    assert staleness_constant(0) == 1.0

    # Stale updates must have strictly decreasing weights
    s1 = staleness_polynomial(1, alpha=0.5)
    s5 = staleness_polynomial(5, alpha=0.5)
    s20 = staleness_polynomial(20, alpha=0.5)
    assert 1.0 > s1 > s5 > s20 > 0.0

    # Hinge function: plateau for tau <= b, then decay
    assert staleness_hinge(2, a=1.0, b=4) == 1.0
    assert staleness_hinge(4, a=1.0, b=4) == 1.0
    assert staleness_hinge(5, a=1.0, b=4) < 1.0

    # Dispatcher
    assert compute_staleness_weight(0, func="poly") == 1.0
    assert compute_staleness_weight(0, func="hinge") == 1.0
    assert compute_staleness_weight(10, func="const") == 1.0


def test_async_server_update():
    w_server = {"w": torch.tensor([1.0, 2.0])}
    w_client = {"w": torch.tensor([3.0, 6.0])}

    # alpha = 0.5, s_weight = 1.0 -> effective_alpha = 0.5
    updated = async_server_update(w_server, w_client, staleness_weight=1.0, alpha=0.5)
    assert torch.allclose(updated["w"], torch.tensor([2.0, 4.0]))

    # Stale update with s_weight = 0.2 -> effective_alpha = 0.1
    updated_stale = async_server_update(w_server, w_client, staleness_weight=0.2, alpha=0.5)
    # (1 - 0.1)*1.0 + 0.1*3.0 = 1.2
    # (1 - 0.1)*2.0 + 0.1*6.0 = 2.4
    assert torch.allclose(updated_stale["w"], torch.tensor([1.2, 2.4]))


def test_run_async_fl_simulation():
    torch.manual_seed(42)
    mock_data = {}
    for sid in range(4):
        x_tr = torch.randn(16, 13)
        y_tr = torch.randint(0, 2, (16, 1)).float()
        x_te = torch.randn(8, 13)
        y_te = torch.randint(0, 2, (8, 1)).float()
        mock_data[sid] = {"train": (x_tr, y_tr), "test": (x_te, y_te)}
    mock_pooled_test = (torch.randn(16, 13), torch.randint(0, 2, (16, 1)).float())

    recs = run_async_fl(
        mock_data, mock_pooled_test, seeds=[42], total_updates=6,
        eval_interval=3, base_time=0.1,
    )
    # Evaluated at updates 3 and 6: 2 evaluation points * (4 sites + 1 pooled) = 10 records
    assert len(recs) == 10
    assert recs[0]["method"] == "fedasync"
    assert recs[-1]["sim_time_s"] > 0.0
