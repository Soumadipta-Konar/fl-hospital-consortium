import math
from src.privacy_accountant import (
    compute_privacy_budget,
    compute_privacy_curve,
    validate_clinical_privacy,
    compute_gaussian_rdp,
    compute_subsampled_gaussian_rdp,
)


def test_zero_noise_infinite_epsilon():
    eps, delta = compute_privacy_budget(noise_multiplier=0.0, rounds=10)
    assert math.isinf(eps)
    assert delta == 1e-5


def test_compute_gaussian_rdp():
    # RDP(alpha) = alpha / (2 * sigma^2)
    # alpha = 2.0, sigma = 1.0 -> 2.0 / 2.0 = 1.0
    val = compute_gaussian_rdp(alpha=2.0, sigma=1.0)
    assert math.isclose(val, 1.0)
    assert math.isinf(compute_gaussian_rdp(alpha=2.0, sigma=0.0))


def test_monotonicity_with_rounds():
    # As communication rounds increase, privacy spend must strictly increase
    eps_10, _ = compute_privacy_budget(noise_multiplier=1.0, rounds=10)
    eps_25, _ = compute_privacy_budget(noise_multiplier=1.0, rounds=25)
    eps_50, _ = compute_privacy_budget(noise_multiplier=1.0, rounds=50)

    assert 0 < eps_10 < eps_25 < eps_50


def test_monotonicity_with_noise_multiplier():
    # Higher noise injection must strictly reduce epsilon (stronger privacy guarantee)
    eps_low_noise, _ = compute_privacy_budget(noise_multiplier=0.5, rounds=20)
    eps_med_noise, _ = compute_privacy_budget(noise_multiplier=1.0, rounds=20)
    eps_high_noise, _ = compute_privacy_budget(noise_multiplier=2.0, rounds=20)

    assert eps_low_noise > eps_med_noise > eps_high_noise > 0.0


def test_subsampling_amplification():
    # Sampling q < 1.0 (client dropout / Poisson subsampling) must provide strictly stronger privacy than q = 1.0
    rdp_full = compute_subsampled_gaussian_rdp(alpha=4.0, sigma=1.0, q=1.0)
    rdp_sub = compute_subsampled_gaussian_rdp(alpha=4.0, sigma=1.0, q=0.5)
    assert rdp_sub < rdp_full

    eps_full, _ = compute_privacy_budget(noise_multiplier=1.0, rounds=20, q=1.0)
    eps_sub, _ = compute_privacy_budget(noise_multiplier=1.0, rounds=20, q=0.5)
    assert eps_sub < eps_full


def test_privacy_curve_generation():
    curve = compute_privacy_curve(noise_multiplier=1.0, max_rounds=10)
    assert len(curve) == 10
    assert curve[0]["round"] == 1
    assert curve[-1]["round"] == 10
    assert curve[0]["epsilon"] < curve[-1]["epsilon"]


def test_clinical_validation_tiers():
    strict_res = validate_clinical_privacy(1.2)
    assert strict_res["tier"] == "strict"
    assert strict_res["compliant"] is True

    mod_res = validate_clinical_privacy(5.4)
    assert mod_res["tier"] == "moderate"
    assert mod_res["compliant"] is True

    non_comp = validate_clinical_privacy(float("inf"))
    assert non_comp["tier"] == "non-compliant"
    assert non_comp["compliant"] is False
