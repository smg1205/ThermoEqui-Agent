"""Optional PGSSI-to-parameter regression bridge.

PGSSI predicts temperature-dependent infinite-dilution activity coefficients
``log-gamma_inf = K1 + K2 / T``.  For engineering use inside ThermoEqui-Agent,
those predictions (or measured gamma-infinity data) can be regressed into
NRTL binary interaction parameters that the production NRTL backend consumes.

This module is an *optional auxiliary* of the PGSSI backend: PGSSI itself never
depends on it, and the NRTL backend never depends on PGSSI.  It only converts
gamma-infinity data into a reviewed ``ParameterSet``.

Scientific caveat: gamma-infinity is an infinite-dilution limit.  Parameters
regressed from it must be validated against finite-concentration VLE benchmark
data before they may be treated as production-ready.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from schemas.domain import FailureType, ParameterSet
from thermo_engine.errors import ThermoEquiError

NRTL_ALPHA_DEFAULT = 0.3
_PARAM_BOUNDS = (-8000.0, 8000.0)


def _nrtl_ln_gamma_infinity(b12: float, b21: float, temperature_K: float, alpha: float = NRTL_ALPHA_DEFAULT) -> float:
    """Return ln(gamma_inf) of component 1 at infinite dilution in component 2 (NRTL).

    Uses the same two-parameter NRTL form (a12 = a21 = 0, tau_ij = b_ij / T) as
    the PGSSI regression script, evaluated with CalebBell/thermo.
    """
    from thermo import NRTL

    tau = np.zeros((2, 2, 6))
    tau[0, 1, 0] = b12 / temperature_K
    tau[1, 0, 0] = b21 / temperature_K
    alp = np.zeros((2, 2, 2))
    alp[0, 1, 0] = alpha
    alp[1, 0, 0] = alpha
    xs = np.array([1e-6, 1.0 - 1e-6])
    gammas = NRTL(xs=xs, T=temperature_K, tau_coeffs=tau, alpha_coeffs=alp).gammas()
    return float(np.log(gammas[0]))


def regress_nrtl_from_gamma_infinity(
    component_order: list[str],
    temperatures_k: list[float],
    ln_gamma_inf: list[float],
    *,
    alpha: float = NRTL_ALPHA_DEFAULT,
    source_title: str | None = None,
    source_identifier: str | None = None,
    source_type: str = "estimated",
    quality_level: str = "pgssi-regressed-unreviewed",
    notes: str | None = None,
) -> ParameterSet:
    """Regress NRTL (b12, b21) from infinite-dilution activity coefficient data.

    ``ln_gamma_inf`` is the natural logarithm of gamma_infinity of component 0
    at infinite dilution in component 1 at each temperature.  At least two
    distinct temperatures are required.
    """
    if len(component_order) != 2:
        raise ThermoEquiError(
            FailureType.MISSING_DATA,
            "NRTL regression is binary.",
            "Provide exactly two components in component_order.",
        )
    if len(temperatures_k) < 2 or len(temperatures_k) != len(ln_gamma_inf):
        raise ThermoEquiError(
            FailureType.MISSING_DATA,
            "NRTL regression needs at least two gamma-infinity data points.",
            "Provide matching temperatures_k and ln_gamma_inf arrays with at least two entries.",
        )
    temperatures = np.asarray(temperatures_k, dtype=float)
    targets = np.asarray(ln_gamma_inf, dtype=float)
    if not np.all(np.isfinite(temperatures)) or not np.all(np.isfinite(targets)) or np.any(temperatures <= 0):
        raise ThermoEquiError(
            FailureType.MISSING_DATA,
            "NRTL regression inputs must be finite with positive temperatures.",
            "Review the gamma-infinity data before regression.",
        )

    def residuals(params: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        b12, b21 = params
        return np.array(
            [
                _nrtl_ln_gamma_infinity(b12, b21, float(T), alpha) - float(y)
                for T, y in zip(temperatures, targets, strict=True)
            ]
        )

    result = least_squares(
        residuals,
        x0=np.array([0.0, 0.0]),
        bounds=([_PARAM_BOUNDS[0], _PARAM_BOUNDS[0]], [_PARAM_BOUNDS[1], _PARAM_BOUNDS[1]]),
    )
    b12, b21 = (float(value) for value in result.x)
    if not np.all(np.isfinite([b12, b21])):
        raise ThermoEquiError(
            FailureType.NUMERICAL_NONCONVERGENCE,
            "NRTL gamma-infinity regression did not converge to finite parameters.",
            "Review the data range and starting point; do not use this result.",
        )

    temperature_low = float(np.min(temperatures))
    temperature_high = float(np.max(temperatures))
    return ParameterSet(
        model_name="NRTL",
        component_order=[component_order[0], component_order[1]],
        parameters={
            "tau12_a": 0.0,
            "tau12_b": b12,
            "tau21_a": 0.0,
            "tau21_b": b21,
            "alpha": alpha,
        },
        parameter_form="NRTL a+b/T binary",
        units={
            "tau12_a": "dimensionless",
            "tau12_b": "K",
            "tau21_a": "dimensionless",
            "tau21_b": "K",
            "alpha": "dimensionless",
        },
        temperature_range_K=(temperature_low, temperature_high),
        equilibrium_types=["VLE", "FLASH"],
        source_type=source_type,  # type: ignore[arg-type]
        source_title=source_title,
        source_identifier=source_identifier,
        quality_level=quality_level,
        notes=notes
        or "Regressed from infinite-dilution activity coefficient data; validate against "
        "finite-concentration VLE benchmarks before production use.",
    )
