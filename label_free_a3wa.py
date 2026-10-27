"""Label-free A3WA calculations for an isolated, not-yet-wired protocol.

Only caller-supplied risk signals and unlabeled development memberships are
accepted.  This module neither loads project data nor fits scoring labels.
Inputs and parameter provenance must still be audited by the caller.
"""

from math import isfinite

from calibration_utils import compute_a3wa_thresholds, discrete_a3wa_information_loss


def _finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _unit(value, name):
    value = _finite_number(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def validate_loss_parameters(parameters):
    required = {"lambda1", "lambda2", "mu1", "mu2"}
    if not isinstance(parameters, dict) or set(parameters) != required:
        raise ValueError("Exactly lambda1, lambda2, mu1 and mu2 are required")
    validated = {name: _finite_number(parameters[name], name) for name in required}
    if any(value <= 0 for value in validated.values()):
        raise ValueError("All A3WA loss coefficients must be positive")
    return validated


def equal_weight_membership(risks, *, valid_probe_count):
    """Return fuzzy membership, not a probability of a correct grade.

    Risk extraction is a separate concern; its provenance and evidence must be
    independently validated. Refuse incomplete scores instead of interpreting
    missing probes as perfect agreement.
    """
    if isinstance(valid_probe_count, bool) or not isinstance(valid_probe_count, int):
        raise ValueError("valid_probe_count must be an integer")
    if valid_probe_count < 2:
        raise ValueError("At least two valid grading probes are required")
    required = {"U_E", "U_S", "U_R"}
    if not isinstance(risks, dict) or set(risks) != required:
        raise ValueError("Exactly U_E, U_S and U_R are required")
    values = [_unit(risks[name], name) for name in ("U_E", "U_S", "U_R")]
    return 1.0 - sum(values) / 3.0


def route_membership(membership, *, m, loss_parameters):
    membership = _unit(membership, "membership")
    m = _finite_number(m, "m")
    if not 0.0 < m < 1.0:
        raise ValueError("m must be strictly between 0 and 1")
    params = validate_loss_parameters(loss_parameters)
    alpha, beta = compute_a3wa_thresholds(m=m, **params)
    if not 0.0 <= beta < m < alpha <= 1.0:
        raise ValueError("Invalid A3WA threshold ordering")
    route = "POS" if membership >= alpha else "NEG" if membership <= beta else "BND"
    return {"mu": membership, "m": m, "alpha": alpha, "beta": beta, "route": route}


def select_m_by_unlabeled_information_loss(memberships, *, candidates, loss_parameters):
    """Choose the smallest tied candidate on an explicitly supplied grid.

    Information-loss minimization is not an estimate of teacher-score error.
    The caller must freeze the grid and coefficients before examining labels.
    """
    values = [_unit(value, "membership") for value in memberships]
    if not values:
        raise ValueError("At least one unlabeled development membership is required")
    params = validate_loss_parameters(loss_parameters)
    grid = sorted({_finite_number(value, "m") for value in candidates})
    if not grid or any(not 0.0 < value < 1.0 for value in grid):
        raise ValueError("The m grid must contain values strictly between 0 and 1")
    results = []
    for candidate in grid:
        routed = route_membership(0.5, m=candidate, loss_parameters=params)
        loss = discrete_a3wa_information_loss(values, m=candidate, **params)
        results.append({"m": candidate, "alpha": routed["alpha"], "beta": routed["beta"], "information_loss": loss})
    best = min(results, key=lambda item: (item["information_loss"], item["m"]))
    return {"selected": dict(best), "candidates": results, "sample_count": len(values)}
