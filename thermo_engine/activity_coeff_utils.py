"""Shared utilities for activity-coefficient-based VLE calculations (NRTL/UNIQUAC/Wilson)."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.optimize import brentq

from schemas.domain import (
    CalculationResult,
    EquilibriumPoint,
    FailureDetail,
    FailureType,
    PhaseResult,
    TaskManifest,
)
from thermo_engine.errors import ThermoEquiError

GammaFunction = Callable[
    [np.ndarray[Any, Any], float, list[str], dict[str, Any] | None],
    np.ndarray[Any, Any],
]

# Antoine constants: log10(P[mmHg]) = A - B/(C + T[C]), source: NIST
NIST_ANTOINE: dict[str, tuple[float, float, float, float, float]] = {
    "benzene": (6.90565, 1211.033, 220.790, 6.0, 106.0),
    "toluene": (6.95464, 1344.800, 219.480, 6.0, 111.0),
    "ethanol": (8.04494, 1554.300, 222.650, -5.0, 96.0),
    "water": (8.07131, 1730.630, 233.426, 1.0, 100.0),
    "methanol": (7.87863, 1473.110, 230.000, -14.0, 65.0),
    "acetone": (7.02447, 1161.000, 224.000, -13.0, 55.0),
    "hexane": (6.87601, 1171.170, 224.408, -10.0, 95.0),
    "heptane": (6.89341, 1264.370, 216.640, -7.0, 105.0),
    "octane": (6.90940, 1349.820, 209.385, -4.0, 115.0),
    "cyclohexane": (6.84130, 1201.530, 222.650, -10.0, 85.0),
}


def psat_Pa(component_name: str, T_K: float) -> float:
    """Compute vapor pressure (Pa) using Antoine equation."""
    key = component_name.lower().strip()
    if key not in NIST_ANTOINE:
        from thermo import Chemical

        chem = Chemical(key)
        return float(chem.VaporPressure(T_K) * 1e5)  # bar -> Pa
    A, B, C, _, _ = NIST_ANTOINE[key]
    T_C = T_K - 273.15
    return float((10.0 ** (A - B / (C + T_C))) * 133.322)


def component_names(request: TaskManifest) -> list[str]:
    """Extract component name strings from manifest."""
    return [c.name for c in request.components]


def require_pressure(request: TaskManifest) -> float:
    p = request.conditions.pressure_kPa
    if p is None:
        raise ThermoEquiError(FailureType.MISSING_DATA, "Pressure required.", "Provide pressure in kPa.")
    return p


def require_composition(request: TaskManifest, field: str, n: int) -> list[float]:
    comp = getattr(request.conditions, field, None)
    if comp is None:
        raise ThermoEquiError(FailureType.MISSING_DATA, f"{field} required.", f"Provide {field}.")
    if len(comp) != n:
        raise ThermoEquiError(FailureType.SEMANTIC_FAILURE, f"{field} length mismatch.", "Check composition length.")
    return [float(value) for value in comp]


def gamma_nrtl(
    xs: np.ndarray[Any, Any],
    T: float,
    names: list[str],
    params: dict[str, Any] | None,
) -> np.ndarray[Any, Any]:
    """Compute NRTL activity coefficients using thermo."""
    from thermo import NRTL

    if params is None:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS, "NRTL needs tau_coeffs.", "Provide binary interaction parameters."
        )
    tau = np.array(params.get("tau_coeffs", np.zeros((len(names), len(names), 6))))
    alpha = np.array(params.get("alpha_coeffs", np.zeros((len(names), len(names), 2))))
    if alpha.sum() == 0:
        N = len(names)
        alpha = np.zeros((N, N, 2))
        for i in range(N):
            for j in range(N):
                if i != j:
                    alpha[i, j, 0] = 0.3
    model = NRTL(xs=xs, T=T, tau_coeffs=tau, alpha_coeffs=alpha)
    return np.array(model.gammas())


def gamma_uniqac(
    xs: np.ndarray[Any, Any],
    T: float,
    names: list[str],
    params: dict[str, Any] | None,
) -> np.ndarray[Any, Any]:
    """Compute UNIQUAC activity coefficients using thermo."""
    from thermo import UNIQUAC

    if params is None:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS, "UNIQUAC needs tau_coeffs.", "Provide binary interaction parameters."
        )
    tau = np.array(params.get("tau_coeffs", np.zeros((len(names), len(names), 6))))
    rs = np.array(params.get("rs", [1.0] * len(names)))
    qs = np.array(params.get("qs", [1.0] * len(names)))
    model = UNIQUAC(xs=xs, rs=rs, qs=qs, T=T, tau_coeffs=tau)
    return np.array(model.gammas())


def gamma_wilson(
    xs: np.ndarray[Any, Any],
    T: float,
    names: list[str],
    params: dict[str, Any] | None,
) -> np.ndarray[Any, Any]:
    """Compute Wilson activity coefficients using thermo."""
    from thermo import Wilson

    if params is None:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS, "Wilson needs Lambda_coeffs.", "Provide binary interaction parameters."
        )
    lmbda = np.array(params.get("Lambda_coeffs", np.zeros((len(names), len(names), 6))))
    model = Wilson(xs=xs, T=T, lambda_coeffs=lmbda)
    return np.array(model.gammas())


def build_result(
    request: TaskManifest,
    model_name: str,
    backend_version: str,
    *,
    points: list[EquilibriumPoint] | None = None,
    phases: list[PhaseResult] | None = None,
    T: float | None = None,
    P: float | None = None,
    vf: float | None = None,
    phase_state: str = "unknown",
    residual: float = 0.0,
    iters: int = 0,
    warnings: list[str] | None = None,
    failure: FailureDetail | None = None,
    solver_name: str = "activity-coefficient VLE",
) -> CalculationResult:
    """Build a CalculationResult following the existing pattern."""
    return CalculationResult(
        run_id=str(uuid4()),
        task_id=request.task_id,
        calculation_type=request.calculation_type,
        input_snapshot=request.model_dump(mode="json"),
        model_name=model_name,
        parameter_set_id=None,
        points=points or [],
        phases=phases or [],
        temperature_K=T,
        pressure_kPa=P,
        vapor_fraction=vf,
        phase_state=phase_state,
        converged=failure is None,
        residual=float(residual),
        iterations=iters,
        warnings=list(dict.fromkeys(warnings or [])),
        backend_version=backend_version,
        solver_name=solver_name,
        failure=failure,
        created_at=datetime.now(UTC),
    )


def temperature_bounds(names: list[str], P_kPa: float) -> tuple[float, float]:
    """Estimate temperature bounds for bubble/dew iteration."""
    P_mmHg = P_kPa * 7.50062
    lows, highs = [], []
    for n in names:
        key = n.lower().strip()
        if key in NIST_ANTOINE:
            A, B, C, tmin, tmax = NIST_ANTOINE[key]
            try:
                Tb = B / (A - math.log10(P_mmHg)) - C + 273.15
                lows.append(Tb - 20)
                highs.append(Tb + 20)
            except (ValueError, OverflowError):
                pass
    if not lows:
        return 250.0, 500.0
    return min(lows), max(highs)


def vle_bubble(
    P_kPa: float,
    names: list[str],
    xs: list[float],
    gamma_fn: GammaFunction,
    params: dict[str, Any] | None,
    solver_tol: float = 1e-9,
) -> tuple[float, list[float], list[float], int]:
    """Bubble point: given P, x -> find T, y."""
    low, high = temperature_bounds(names, P_kPa)
    P_Pa = P_kPa * 1000.0

    def f(T: float) -> float:
        g = gamma_fn(np.array(xs), T, names, params)
        ys = [xs[i] * g[i] * psat_Pa(names[i], T) / P_Pa for i in range(len(names))]
        return float(sum(ys) - 1.0)

    try:
        T_sol, info = brentq(f, low, high, xtol=solver_tol, full_output=True, maxiter=200)
    except (ValueError, RuntimeError) as e:
        raise ThermoEquiError(
            FailureType.NUMERICAL_NONCONVERGENCE,
            "Bubble point did not converge.",
            "Check temperature bounds and parameters.",
        ) from e
    g = gamma_fn(np.array(xs), T_sol, names, params)
    ys = [xs[i] * g[i] * psat_Pa(names[i], T_sol) / P_Pa for i in range(len(names))]
    return T_sol, ys, g.tolist(), int(info.iterations)


def vle_dew(
    P_kPa: float,
    names: list[str],
    ys: list[float],
    gamma_fn: GammaFunction,
    params: dict[str, Any] | None,
    solver_tol: float = 1e-9,
) -> tuple[float, list[float], list[float], int]:
    """Dew point: given P, y -> find T, x."""
    low, high = temperature_bounds(names, P_kPa)
    P_Pa = P_kPa * 1000.0

    def f(T: float) -> float:
        xs = [ys[i] * P_Pa / psat_Pa(names[i], T) for i in range(len(names))]
        s = sum(xs)
        xs_n = [x / s for x in xs]
        g = gamma_fn(np.array(xs_n), T, names, params)
        xs2 = [ys[i] * P_Pa / (g[i] * psat_Pa(names[i], T)) for i in range(len(names))]
        return float(sum(xs2) - 1.0)

    try:
        T_sol, info = brentq(f, low, high, xtol=solver_tol, full_output=True, maxiter=200)
    except (ValueError, RuntimeError) as e:
        raise ThermoEquiError(
            FailureType.NUMERICAL_NONCONVERGENCE,
            "Dew point did not converge.",
            "Check temperature bounds and parameters.",
        ) from e
    initial_xs = [ys[i] * P_Pa / psat_Pa(names[i], T_sol) for i in range(len(names))]
    initial_sum = sum(initial_xs)
    xs_n = [x / initial_sum for x in initial_xs]
    g = gamma_fn(np.array(xs_n), T_sol, names, params)
    xs = [ys[i] * P_Pa / (g[i] * psat_Pa(names[i], T_sol)) for i in range(len(names))]
    s = sum(xs)
    xn = [x / s for x in xs]
    return T_sol, xn, g.tolist(), int(info.iterations)
