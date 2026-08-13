"""Base class for activity-coefficient VLE backends (NRTL/UNIQUAC/Wilson).

Binary parameters always come from reviewed ``ParameterSet`` records carried
on the task or attached from the production parameter repository.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import brentq

from schemas.domain import (
    CalculationResult,
    EquilibriumPoint,
    FailureType,
    PhaseResult,
    TaskManifest,
)
from thermo_engine.activity_coeff_utils import build_result, psat_Pa, temperature_bounds
from thermo_engine.errors import ThermoEquiError
from thermo_engine.parameters import matching_parameter_set, parameter_set_to_backend_params


class ActCoeffBackend:
    """Base class for activity-coefficient models. Subclass must implement _gamma() and set model_name, version."""

    model_name = ""
    version = ""

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        params = self._params(request)
        parameter_set = params.get("_parameter_set")
        assert parameter_set is not None
        return [
            {
                "component": " / ".join(parameter_set.component_order),
                "property": f"{self.model_name} interaction parameters",
                "parameter_form": parameter_set.parameter_form,
                "parameter_values": json.dumps(parameter_set.parameters),
                "parameter_set_id": parameter_set.parameter_set_id,
                "source_title": parameter_set.source_title or "user-supplied",
                "source_identifier": parameter_set.source_identifier or "user-supplied",
                "quality_level": parameter_set.quality_level,
                "temperature_range_K": str(parameter_set.temperature_range_K),
                "pressure_range_kPa": str(parameter_set.pressure_range_kPa),
            }
        ]

    def _gamma(self, xs: np.ndarray, T: float, names: list[str], params: dict | None) -> np.ndarray:
        raise NotImplementedError

    def _params(self, req: TaskManifest) -> dict | None:
        """Resolve the matching reviewed parameter set from the request manifest."""
        names = [c.name for c in req.components]
        parameter_set = matching_parameter_set(req.parameters, req.components, self.model_name)
        if parameter_set is not None:
            try:
                converted = parameter_set_to_backend_params(
                    parameter_set,
                    self.model_name,
                    parameter_set.component_order,
                )
            except ValueError as error:
                raise ThermoEquiError(
                    FailureType.MISSING_PARAMETERS,
                    f"{self.model_name} parameter set could not be converted: {error}",
                    "Provide a supported parameter_form with the required parameter names.",
                ) from error
            converted["_parameter_set"] = parameter_set
            return converted
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            f"{self.model_name} needs binary interaction parameters for {names}.",
            "Provide a reviewed or user-attested parameter set on the request.",
        )

    def bubble_point(self, req: TaskManifest) -> CalculationResult:
        names = [c.name for c in req.components]
        P = req.conditions.pressure_kPa
        xs = req.conditions.liquid_composition
        if P is None:
            raise ThermoEquiError(FailureType.MISSING_DATA, "Pressure required.", "")
        if xs is None:
            raise ThermoEquiError(FailureType.MISSING_DATA, "liquid_composition required.", "")
        lo, hi = temperature_bounds(names, P)
        PP = P * 1000.0
        params = self._params(req)

        def f(T):
            g = self._gamma(np.array(xs), T, names, params)
            return sum(xs[i] * g[i] * psat_Pa(names[i], T) / PP for i in range(len(names))) - 1.0

        try:
            T, info = brentq(f, lo, hi, xtol=1e-9, full_output=True, maxiter=200)
        except Exception as e:
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                f"Bubble failed: {e}",
                "",
            ) from None
        g = self._gamma(np.array(xs), float(T), names, params)
        ys = [xs[i] * g[i] * psat_Pa(names[i], float(T)) / PP for i in range(len(names))]
        point = EquilibriumPoint(
            temperature_K=float(T),
            pressure_kPa=P,
            liquid_composition=xs,
            vapor_composition=ys,
            equilibrium_residual=abs(sum(ys) - 1.0),
        )
        return build_result(
            req,
            self.model_name,
            self.version,
            T=float(T),
            P=P,
            residual=abs(sum(ys) - 1.0),
            iters=int(info.iterations),
            points=[point],
            phases=[
                PhaseResult(phase="liquid", fraction=1.0, composition=xs),
                PhaseResult(phase="vapor", fraction=0.0, composition=ys),
            ],
            phase_state="two_phase",
        )

    def dew_point(self, req: TaskManifest) -> CalculationResult:
        names = [c.name for c in req.components]
        P = req.conditions.pressure_kPa
        ys = req.conditions.vapor_composition
        if P is None:
            raise ThermoEquiError(FailureType.MISSING_DATA, "Pressure required.", "")
        if ys is None:
            raise ThermoEquiError(FailureType.MISSING_DATA, "vapor_composition required.", "")
        lo, hi = temperature_bounds(names, P)
        PP = P * 1000.0
        params = self._params(req)

        def f(T):
            xi = [ys[i] * PP / psat_Pa(names[i], T) for i in range(len(names))]
            s = sum(xi)
            xn = [x / s for x in xi] if s > 0 else xi
            for _ in range(20):
                g = self._gamma(np.array(xn), T, names, params)
                xs = [ys[i] * PP / (g[i] * psat_Pa(names[i], T)) for i in range(len(names))]
                xs_sum = sum(xs)
                xn_new = [x / xs_sum for x in xs] if xs_sum > 0 else xs
                if max(abs(a - b) for a, b in zip(xn_new, xn, strict=True)) < 1e-12:
                    xn = xn_new
                    break
                xn = xn_new
            g = self._gamma(np.array(xn), T, names, params)
            return sum(ys[i] * PP / (g[i] * psat_Pa(names[i], T)) for i in range(len(names))) - 1.0

        try:
            T, info = brentq(f, lo, hi, xtol=1e-9, full_output=True, maxiter=200)
        except Exception as e:
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                f"Dew failed: {e}",
                "",
            ) from None
        xs = [ys[i] * PP / psat_Pa(names[i], float(T)) for i in range(len(names))]
        s = sum(xs)
        xn = [x / s for x in xs]
        for _ in range(50):
            g = self._gamma(np.array(xn), float(T), names, params)
            xs = [ys[i] * PP / (g[i] * psat_Pa(names[i], float(T))) for i in range(len(names))]
            s = sum(xs)
            xn_new = [x / s for x in xs]
            if max(abs(a - b) for a, b in zip(xn_new, xn, strict=True)) < 1e-12:
                xn = xn_new
                break
            xn = xn_new
        g = self._gamma(np.array(xn), float(T), names, params)
        xs = [ys[i] * PP / (g[i] * psat_Pa(names[i], float(T))) for i in range(len(names))]
        s = sum(xs)
        point = EquilibriumPoint(
            temperature_K=float(T),
            pressure_kPa=P,
            liquid_composition=xn,
            vapor_composition=ys,
            equilibrium_residual=abs(s - 1.0),
        )
        return build_result(
            req,
            self.model_name,
            self.version,
            T=float(T),
            P=P,
            residual=abs(s - 1.0),
            iters=int(info.iterations),
            points=[point],
            phases=[
                PhaseResult(phase="liquid", fraction=1.0, composition=xn),
                PhaseResult(phase="vapor", fraction=0.0, composition=ys),
            ],
            phase_state="two_phase",
        )

    def isobaric_vle(self, req: TaskManifest) -> CalculationResult:
        """Isobaric T-x-y VLE curve: iterate x1, compute bubble point for each fraction.
        Binary mixtures only, matching the convention of Ideal/Raoult and PR backends.
        """
        if len(req.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The VLE curve endpoint supports binary mixtures only.",
                "Provide exactly two components or use a point/flash calculation.",
            )
        P = req.conditions.pressure_kPa
        if P is None:
            raise ThermoEquiError(FailureType.MISSING_DATA, "Pressure required.", "")
        all_points: list[EquilibriumPoint] = []
        iterations = 0
        warnings: list[str] = []
        # Slight offset from pure edges to avoid NaN in combinatorial terms
        _eps = 1e-12
        for fraction in np.linspace(_eps, 1.0 - _eps, req.points):
            sub_req = req.model_copy(
                update={
                    "conditions": req.conditions.model_copy(
                        update={
                            "liquid_composition": [float(fraction), float(1.0 - fraction)],
                        }
                    ),
                }
            )
            point_result = self.bubble_point(sub_req)
            all_points.extend(point_result.points)
            for w in point_result.warnings or []:
                if w not in warnings:
                    warnings.append(w)
            iterations += point_result.iterations
        residual = max(p.equilibrium_residual for p in all_points)
        return build_result(
            req,
            self.model_name,
            self.version,
            points=all_points,
            T=all_points[0].temperature_K if all_points else None,
            P=P,
            residual=residual,
            iters=iterations,
            warnings=warnings,
            phase_state="curve",
        )

    def isothermal_vle(self, req: TaskManifest) -> CalculationResult:
        """Isothermal P-x-y VLE curve: iterate x1, compute bubble pressure at each fraction.
        Binary mixtures only, matching the convention of Ideal/Raoult and PR backends.
        """
        if len(req.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The VLE curve endpoint supports binary mixtures only.",
                "Provide exactly two components or use a point/flash calculation.",
            )
        names = [c.name for c in req.components]
        T = req.conditions.temperature_K
        if T is None:
            raise ThermoEquiError(FailureType.MISSING_DATA, "Temperature required for isothermal VLE.", "")
        params = self._params(req)
        all_points: list[EquilibriumPoint] = []
        _eps = 1e-12
        for fraction in np.linspace(_eps, 1.0 - _eps, req.points):
            xs = np.array([fraction, 1.0 - fraction], dtype=float)
            g = self._gamma(xs, T, names, params)
            psat_vals = [psat_Pa(names[i], T) for i in range(2)]
            P_Pa = float(xs[0] * g[0] * psat_vals[0] + xs[1] * g[1] * psat_vals[1])
            y1 = float(xs[0] * g[0] * psat_vals[0] / P_Pa) if P_Pa > 1e-30 else 0.0
            y2 = float(xs[1] * g[1] * psat_vals[1] / P_Pa) if P_Pa > 1e-30 else 0.0
            s = y1 + y2
            if not (s > 1e-30 and all(np.isfinite([y1, y2, P_Pa]))):
                continue
            residual = abs(s - 1.0) if s > 1e-30 else 1.0
            yn = [y1 / s, y2 / s] if s > 1e-30 else [y1, y2]
            all_points.append(
                EquilibriumPoint(
                    temperature_K=T,
                    pressure_kPa=P_Pa / 1000.0,
                    liquid_composition=[float(xs[0]), float(xs[1])],
                    vapor_composition=yn,
                    equilibrium_residual=residual,
                )
            )
        if not all_points:
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                "All isothermal VLE points produced non-finite results.",
                "Check temperature and component parameters.",
            )
        residual = max(p.equilibrium_residual for p in all_points)
        return build_result(
            req,
            self.model_name,
            self.version,
            points=all_points,
            T=T,
            P=None,
            residual=residual,
            iters=0,
            phase_state="curve",
        )

    def tp_flash(self, req: TaskManifest) -> CalculationResult:
        """TP flash with composition-dependent K via successive substitution.
        K_i = gamma_i * P_sat_i / P depends on liquid composition,
        so an outer loop updates K after each Rachford-Rice solve.
        """
        T = req.conditions.temperature_K
        P = req.conditions.pressure_kPa
        zs = req.conditions.feed_composition
        if T is None:
            raise ThermoEquiError(FailureType.MISSING_DATA, "Temperature required for TP Flash.", "")
        if P is None:
            raise ThermoEquiError(FailureType.MISSING_DATA, "Pressure required for TP Flash.", "")
        if zs is None:
            raise ThermoEquiError(FailureType.MISSING_DATA, "feed_composition required for TP Flash.", "")
        names = [c.name for c in req.components]
        n = len(names)
        P_Pa = P * 1000.0
        params = self._params(req)
        K = np.array([psat_Pa(names[i], T) / P_Pa for i in range(n)])

        def _rr(beta):
            return float(sum(zs[i] * (K[i] - 1.0) / (1.0 + beta * (K[i] - 1.0)) for i in range(n)))

        f0 = _rr(0.0)
        f1 = _rr(1.0)

        if f0 <= 0.0:
            beta = 0.0
            xL = np.array(zs)
            xV = K * xL
            xV_sum = xV.sum()
            xV = xV / xV_sum if xV_sum > 0 else xV
            state = "liquid"
            iterations = 0
        elif f1 >= 0.0:
            beta = 1.0
            xV = np.array(zs)
            xL = xV / K
            xL_sum = xL.sum()
            xL = xL / xL_sum if xL_sum > 0 else xL
            state = "vapor"
            iterations = 0
        else:
            state = "two_phase"
            warnings: list[str] = []
            for iteration in range(100):
                try:
                    beta, info = brentq(
                        lambda b, current_K=K: float(
                            sum(zs[i] * (current_K[i] - 1.0) / (1.0 + b * (current_K[i] - 1.0)) for i in range(n))
                        ),
                        0.0,
                        1.0,
                        xtol=1e-9,
                        full_output=True,
                        maxiter=100,
                    )
                except (ValueError, RuntimeError) as e:
                    if iteration == 0:
                        raise ThermoEquiError(
                            FailureType.NUMERICAL_NONCONVERGENCE,
                            f"TP Flash Rachford-Rice failed: {e}",
                            "Check feed composition and conditions.",
                        ) from None
                    warnings.append(f"RR solve clamped at iteration {iteration}")
                    beta = 0.5

                xL_new = np.array([zs[i] / (1.0 + beta * (K[i] - 1.0)) for i in range(n)])
                xV_new = K * xL_new
                xL_new = np.clip(xL_new, 0.0, None)
                xV_new = np.clip(xV_new, 0.0, None)
                xL_sum = xL_new.sum()
                xV_sum = xV_new.sum()
                if xL_sum > 0:
                    xL_new /= xL_sum
                if xV_sum > 0:
                    xV_new /= xV_sum

                g = self._gamma(xL_new, T, names, params)
                K_new = np.array([g[i] * psat_Pa(names[i], T) / P_Pa for i in range(n)])
                K_change = float(np.max(np.abs(K_new - K)))
                K = 0.6 * K_new + 0.4 * K

                if K_change < 1e-10:
                    iterations = iteration + 1
                    break
            else:
                warnings.append("TP Flash successive substitution reached max iterations (100).")
                iterations = 100

            xL = xL_new
            xV = xV_new

        material_balance = np.array([(1.0 - beta) * xL[i] + beta * xV[i] for i in range(n)])
        material_error = float(np.max(np.abs(material_balance - np.array(zs))))
        rr_residual = abs(_rr(beta)) if state == "two_phase" else 0.0
        residual = max(material_error, rr_residual)

        phases = [
            PhaseResult(phase="liquid", fraction=1.0 - beta, composition=[float(x) for x in xL]),
            PhaseResult(phase="vapor", fraction=beta, composition=[float(x) for x in xV]),
        ]
        return build_result(
            req,
            self.model_name,
            self.version,
            T=T,
            P=P,
            vf=float(beta),
            phases=phases,
            residual=residual,
            iters=iterations,
            phase_state=state,
        )

    def phase_stability(self, req: TaskManifest) -> CalculationResult:
        result = self.tp_flash(req)
        message = (
            "Phase stability evaluation via activity-coefficient-based flash only; "
            "tangent-plane distance is not available for this backend."
        )
        return result.model_copy(
            update={
                "warnings": [message] + (result.warnings or []),
            }
        )

    def azeotrope(self, req: TaskManifest) -> CalculationResult:
        curve = self.isobaric_vle(req)
        candidates = [
            point
            for point in curve.points[1:-1]
            if max(
                abs(liquid - vapor)
                for liquid, vapor in zip(
                    point.liquid_composition,
                    point.vapor_composition,
                    strict=True,
                )
            )
            <= 1e-3
        ]
        warning = (
            "No azeotrope candidate met |x-y| <= 1e-3."
            if not candidates
            else "Candidate points require local refinement before engineering use."
        )
        return curve.model_copy(
            update={
                "points": candidates,
                "warnings": [*curve.warnings, warning],
            }
        )

    def lle(self, req: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            f"{self.model_name} cannot represent liquid-liquid equilibrium.",
            "Use a validated LLE backend when available.",
        )
