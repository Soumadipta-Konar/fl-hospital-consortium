import numpy as np
from flower.robust_strategy import RobustFedAvg


def test_flower_robust_median():
    w0 = [np.array([1.0, 10.0]), np.array([[2.0]])]
    w1 = [np.array([2.0, 20.0]), np.array([[3.0]])]
    w2 = [np.array([3.0, 30.0]), np.array([[4.0]])]

    med = RobustFedAvg._aggregate_median([w0, w1, w2])
    assert np.allclose(med[0], np.array([2.0, 20.0]))
    assert np.allclose(med[1], np.array([[3.0]]))


def test_flower_robust_trimmed_mean():
    # 4 clients: 3 benign, 1 poisoned
    w0 = [np.array([1.0, 2.0])]
    w1 = [np.array([1.1, 2.1])]
    w2 = [np.array([0.9, 1.9])]
    w3 = [np.array([1000.0, -999.0])]

    trimmed = RobustFedAvg._aggregate_trimmed_mean([w0, w1, w2, w3], beta=1)
    assert np.all(trimmed[0] < 2.5)
    assert np.all(trimmed[0] > 0.5)


def test_flower_robust_krum():
    w0 = [np.array([1.0, 2.0])]
    w1 = [np.array([1.05, 2.05])]
    w2 = [np.array([0.95, 1.95])]
    w3 = [np.array([500.0, -500.0])]

    selected = RobustFedAvg._aggregate_krum([w0, w1, w2, w3], f=1)
    assert np.all(selected[0] < 3.0)
    assert np.all(selected[0] > 0.0)
