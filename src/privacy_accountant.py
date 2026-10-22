import math
from typing import List, Dict, Any, Tuple


DEFAULT_ORDERS = [
    1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0,
    48.0, 64.0, 96.0, 128.0, 256.0, 512.0,
]


def compute_gaussian_rdp(alpha: float, sigma: float) -> float:
    """Computes exact Rényi Differential Privacy (RDP) for the Gaussian mechanism at order alpha.

    RDP(alpha) = alpha / (2 * sigma^2)
    """
    if sigma <= 0.0:
        return float("inf")
    return alpha / (2.0 * (sigma ** 2))


def compute_subsampled_gaussian_rdp(alpha: float, sigma: float, q: float = 1.0) -> float:
    """Computes RDP for subsampled Gaussian mechanism.

    For full consortium participation (q = 1.0), this matches exact Gaussian RDP.
    For subsampling ratio q < 1.0 (client dropout / subsampling), privacy amplification by subsampling applies.
    """
    if sigma <= 0.0:
        return float("inf")
    if q >= 1.0:
        return compute_gaussian_rdp(alpha, sigma)
    if q <= 0.0:
        return 0.0

    # Subsampled RDP upper bound (Mironov et al., 2019 / Wang et al., 2019)
    base_rdp = compute_gaussian_rdp(alpha, sigma)
    exponent = base_rdp * (alpha - 1.0)

    if exponent > 70.0:
        # Asymptotic regime: log(1 + (q^2 / alpha) * (exp(x) - 1)) ≈ x + 2*log(q) - log(alpha)
        amplified_rdp = base_rdp + (2.0 * math.log(q) - math.log(alpha)) / (alpha - 1.0)
    else:
        amplified_rdp = math.log1p(q * q * (math.exp(exponent) - 1.0) / float(alpha)) / (alpha - 1.0)

    return max(0.0, min(base_rdp, amplified_rdp))


def rdp_to_dp(rdp_list: List[float], orders: List[float], delta: float) -> float:
    """Converts a list of cumulative RDP values across orders into standard (epsilon, delta)-DP.

    epsilon(delta) = min_{alpha > 1} ( RDP(alpha) + ln(1 / delta) / (alpha - 1) )
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must be in the open interval (0, 1)")

    best_eps = float("inf")
    log_inv_delta = math.log(1.0 / delta)

    for rdp, alpha in zip(rdp_list, orders):
        if math.isinf(rdp):
            continue
        eps = rdp + log_inv_delta / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps

    return best_eps


def compute_privacy_budget(
    noise_multiplier: float,
    rounds: int,
    delta: float = 1e-5,
    q: float = 1.0,
    orders: List[float] = None,
) -> Tuple[float, float]:
    """Computes cumulative (epsilon, delta) privacy spend over communication rounds.

    Args:
        noise_multiplier: Noise scale z (sigma = z * clip_norm / K)
        rounds: Number of federated communication rounds
        delta: Target delta (typically 1 / (N_patients) or 1e-5)
        q: Client sampling ratio (1.0 for cross-silo full participation)
        orders: List of Rényi orders to optimize over

    Returns:
        (epsilon, delta) privacy guarantee
    """
    if noise_multiplier <= 0.0:
        return float("inf"), delta
    if rounds <= 0:
        return 0.0, delta

    if orders is None:
        orders = DEFAULT_ORDERS

    # Under RDP, composition across independent rounds is strictly additive:
    # RDP_total(alpha) = sum_{t=1}^T RDP_t(alpha) = T * RDP_per_round(alpha)
    rdp_per_round = [compute_subsampled_gaussian_rdp(alpha, noise_multiplier, q) for alpha in orders]
    total_rdp = [r * rounds for r in rdp_per_round]

    eps = rdp_to_dp(total_rdp, orders, delta)
    return eps, delta


def compute_privacy_curve(
    noise_multiplier: float,
    max_rounds: int = 50,
    delta: float = 1e-5,
    q: float = 1.0,
) -> List[Dict[str, Any]]:
    """Generates analytical privacy spend across rounds for visualization and tracking."""
    curve = []
    for r in range(1, max_rounds + 1):
        eps, _ = compute_privacy_budget(noise_multiplier, rounds=r, delta=delta, q=q)
        curve.append({
            "round": r,
            "epsilon": round(eps, 4),
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "q": q,
        })
    return curve


def validate_clinical_privacy(
    epsilon: float,
    threshold: float = 10.0,
) -> Dict[str, Any]:
    """Evaluates whether the achieved epsilon satisfies academic differential privacy tiers.

    NOTE ON REGULATORY COMPLIANCE:
    The HIPAA Privacy Rule (45 CFR § 164.514) establishes standards for de-identification via
    either Safe Harbor (removal of 18 specific identifiers) or Expert Determination of very small
    re-identification risk. HIPAA does NOT specify any numerical differential privacy epsilon threshold.
    The epsilon tiers below represent academic benchmark thresholds commonly used in FL literature
    to contextualize privacy-utility trade-offs.

    Academic Tiers:
      - Strict (eps <= 2.0): High theoretical confidentiality, minimal reconstruction risk.
      - Moderate (2.0 < eps <= 8.0): Common benchmark in federated learning literature.
      - Relaxed (8.0 < eps <= 15.0): Modest differential privacy guarantees.
      - Non-compliant (eps > 15.0 or inf): Insufficient theoretical privacy guarantee.
    """
    if math.isinf(epsilon) or epsilon > 15.0:
        tier = "non-compliant"
        compliant = False
    elif epsilon <= 2.0:
        tier = "strict"
        compliant = True
    elif epsilon <= 8.0:
        tier = "moderate"
        compliant = True
    else:
        tier = "relaxed"
        compliant = epsilon <= threshold

    return {
        "epsilon": round(epsilon, 4) if not math.isinf(epsilon) else float("inf"),
        "threshold": threshold,
        "compliant": compliant,
        "tier": tier,
        "standard": "Academic Differential Privacy Benchmark Tiers (Note: HIPAA does not define numerical epsilon)",
    }
