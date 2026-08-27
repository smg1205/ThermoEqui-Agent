"""Optional Clapeyron.jl adapter through the official ``pyclapeyron`` bridge."""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
from importlib.metadata import PackageNotFoundError, version
from math import isfinite, log
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from thermo import ChemicalConstantsPackage
from thermo.interaction_parameters import IPDB

from schemas.domain import CalculationResult, EquilibriumPoint, FailureType, PhaseResult, TaskManifest
from thermo_engine.errors import ThermoEquiError
from thermo_engine.thermo_backend import ThermoPengRobinsonBackend

GAS_CONSTANT = 8.31446261815324


class ClapeyronPengRobinsonBackend:
    """Clapeyron.jl Peng-Robinson VLE and Gibbs-minimization flash adapter."""

    dependency = "pyclapeyron==0.1.1"
    model_name = "Clapeyron/Peng-Robinson"
    parameter_table = "ChemSep PR"

    def __init__(self) -> None:
        self._clapeyron: Any | None = None
        self._parameter_set_id: str | None = None
        self._parameter_sources: list[dict[str, str]] = []

    @property
    def backend_version(self) -> str:
        try:
            return f"pyclapeyron/{version('pyclapeyron')} (Clapeyron.jl ^0.6.21)"
        except PackageNotFoundError:
            return "pyclapeyron/unavailable"

    def _load(self) -> Any:
        if self._clapeyron is not None:
            return self._clapeyron
        try:
            self._clapeyron = importlib.import_module("pyclapeyron")
        except (ImportError, OSError, RuntimeError) as exc:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The selected Clapeyron.jl backend is unavailable or its Julia runtime is not initialized.",
                "Install the optional phase-engines dependencies and allow pyclapeyron to initialize Julia once.",
                {"dependency": self.dependency, "cause": type(exc).__name__},
            ) from None
        return self._clapeyron

    @staticmethod
    def _require_pressure(request: TaskManifest) -> float:
        if request.conditions.pressure_kPa is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "Pressure is required for this calculation.",
                "Provide pressure in kPa.",
            )
        return request.conditions.pressure_kPa

    @staticmethod
    def _require_temperature(request: TaskManifest) -> float:
        if request.conditions.temperature_K is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "Temperature is required for this calculation.",
                "Provide temperature in K.",
            )
        return request.conditions.temperature_K

    @staticmethod
    def _require_composition(request: TaskManifest, field: str) -> list[float]:
        values = getattr(request.conditions, field)
        if values is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                f"{field} is required for this calculation.",
                f"Provide {len(request.components)} normalized mole fractions.",
            )
        if len(values) != len(request.components):
            raise ThermoEquiError(
                FailureType.SEMANTIC_FAILURE,
                f"{field} length does not match the component count.",
                "Provide one mole fraction per component in component order.",
            )
        return cast(list[float], values)

    @staticmethod
    def _require_binary(request: TaskManifest) -> None:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "Clapeyron curve and azeotrope endpoints currently support binary mixtures only.",
                "Provide two components or request a point/flash calculation.",
            )

    def _build_model(self, request: TaskManifest) -> Any:
        requested_identifiers = [component.cas_number or component.component_id for component in request.components]
        try:
            constants, _ = ChemicalConstantsPackage.from_IDs(requested_identifiers)
        except (ValueError, LookupError, TypeError):
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "The reviewed identity database could not resolve every Clapeyron component.",
                "Provide canonical names or CAS numbers.",
                {"components": requested_identifiers},
            ) from None
        inapplicable = ThermoPengRobinsonBackend.inapplicable_components(constants)
        if inapplicable:
            raise ThermoEquiError(
                FailureType.PARAMETER_OUT_OF_DOMAIN,
                "The Clapeyron Peng-Robinson adapter is limited to hydrocarbons and reviewed light gases.",
                "Select a reviewed association or activity-coefficient model.",
                {"inapplicable_components": inapplicable},
            )
        missing_pairs = [
            [constants.CASs[i], constants.CASs[j]]
            for i in range(len(constants.CASs))
            for j in range(i + 1, len(constants.CASs))
            if not IPDB.has_ip_specific(self.parameter_table, [constants.CASs[i], constants.CASs[j]], "kij")
        ]
        if missing_pairs:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "Clapeyron Peng-Robinson binary interaction parameters are missing.",
                "Import reviewed kij values or select another applicable model.",
                {"parameter_table": self.parameter_table, "missing_pairs": missing_pairs},
            )
        kijs = IPDB.get_ip_asymmetric_matrix(self.parameter_table, constants.CASs, "kij")
        identifiers = [str(name).casefold() for name in constants.names]
        parameter_stream = io.StringIO()
        writer = csv.writer(parameter_stream, lineterminator="\n")
        writer.writerow(["Clapeyron Database File", ""])
        writer.writerow(["ChemSep PR parameters [csvtype = unlike]"])
        writer.writerow(["species1", "species2", "k", "source"])
        for i in range(len(identifiers)):
            for j in range(i + 1, len(identifiers)):
                writer.writerow([identifiers[i], identifiers[j], kijs[i][j], self.parameter_table])
        clapeyron = self._load()
        try:
            model = clapeyron.PR(identifiers, userlocations=[parameter_stream.getvalue()])
            a_values = np.asarray(model.params.a.values, dtype=float).tolist()
            critical_temperatures = np.asarray(model.params.Tc.values, dtype=float).tolist()
            critical_pressures = np.asarray(model.params.Pc.values, dtype=float).tolist()
            molecular_weights = np.asarray(model.params.Mw.values, dtype=float).tolist()
            references = [str(item) for item in model.references]
        except (ValueError, RuntimeError, TypeError, AttributeError):
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "Clapeyron.jl could not resolve the requested components or PR parameters.",
                "Use names present in the Clapeyron database or provide a reviewed user parameter location.",
                {"components": requested_identifiers},
            ) from None
        snapshot = {
            "component_order": list(constants.CASs),
            "clapeyron_identifiers": identifiers,
            "a_matrix": a_values,
            "kij": kijs,
            "Tc_K": critical_temperatures,
            "Pc_Pa": critical_pressures,
            "Mw_g_mol": molecular_weights,
            "references": references,
            "mixing_rule": "vdW1fRule",
            "binary_parameter_source": self.parameter_table,
            "backend_version": self.backend_version,
        }
        digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self._parameter_set_id = f"clapeyron-pr:{digest}"
        self._parameter_sources = [
            {
                "component": " / ".join(identifiers),
                "property": "Peng-Robinson pure parameters and ChemSep kij matrix",
                "source_title": "Clapeyron.jl packaged pure data + ChemSep PR",
                "source_identifier": "https://github.com/ClapeyronThermo/Clapeyron.jl",
                "source_references": json.dumps(references),
                "parameter_set_id": self._parameter_set_id,
                "parameter_snapshot": json.dumps(snapshot),
            }
        ]
        return model

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        if not self._parameter_sources:
            self._build_model(request)
        return self._parameter_sources

    def _residual(
        self,
        model: Any,
        pressure_pa: float,
        temperature: float,
        liquid: NDArray[np.float64],
        vapor: NDArray[np.float64],
    ) -> float:
        clapeyron = self._load()
        try:
            phi_l = np.asarray(
                clapeyron.fugacity_coefficient(model, pressure_pa, temperature, liquid, phase="liquid"),
                dtype=float,
            )
            phi_v = np.asarray(
                clapeyron.fugacity_coefficient(model, pressure_pa, temperature, vapor, phase="vapor"),
                dtype=float,
            )
        except (ValueError, RuntimeError, TypeError):
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "Clapeyron.jl could not independently evaluate phase fugacity coefficients.",
                "Do not use the result; review model applicability and phase identification.",
            ) from None
        active = (liquid > 1e-14) & (vapor > 1e-14)
        residuals = [
            abs(log(float(x * phi_x) / float(y * phi_y)))
            for x, phi_x, y, phi_y in zip(liquid[active], phi_l[active], vapor[active], phi_v[active], strict=True)
        ]
        residual = max(residuals, default=0.0)
        if not isfinite(residual):
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "Clapeyron.jl returned non-finite fugacity residuals.",
                "Do not use the result; review conditions and model applicability.",
            )
        return residual

    def _point(
        self,
        model: Any,
        temperature: float,
        pressure_pa: float,
        liquid: NDArray[np.float64],
        vapor: NDArray[np.float64],
    ) -> EquilibriumPoint:
        return EquilibriumPoint(
            temperature_K=temperature,
            pressure_kPa=pressure_pa / 1000.0,
            liquid_composition=liquid.tolist(),
            vapor_composition=vapor.tolist(),
            equilibrium_residual=self._residual(model, pressure_pa, temperature, liquid, vapor),
        )

    def _boundary_result(self, request: TaskManifest, point: EquilibriumPoint) -> CalculationResult:
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            parameter_set_id=self._parameter_set_id,
            points=[point],
            temperature_K=point.temperature_K,
            pressure_kPa=point.pressure_kPa,
            phase_state="two_phase",
            converged=point.equilibrium_residual <= 2e-6,
            residual=point.equilibrium_residual,
            iterations=0,
            warnings=self._warnings(),
            backend_version=self.backend_version,
            solver_name="Clapeyron.jl PR / saturation solver",
        )

    @staticmethod
    def _warnings() -> list[str]:
        return [
            "Clapeyron PR uses packaged pure parameters and reviewed ChemSep PR kij values.",
            "ChemSep PR parameter applicability requires engineering review.",
            "pyclapeyron does not expose iteration counts for these solver calls.",
        ]

    @staticmethod
    def _nonconvergence(operation: str) -> ThermoEquiError:
        return ThermoEquiError(
            FailureType.NUMERICAL_NONCONVERGENCE,
            f"Clapeyron.jl did not converge for the requested {operation}.",
            "Review conditions, component identifiers, and Peng-Robinson applicability.",
        )

    def bubble_point(self, request: TaskManifest) -> CalculationResult:
        pressure_pa = self._require_pressure(request) * 1000.0
        liquid = np.asarray(self._require_composition(request, "liquid_composition"), dtype=float)
        model = self._build_model(request)
        clapeyron = self._load()
        try:
            temperature, _, _, vapor = clapeyron.bubble_temperature(model, pressure_pa, liquid)
        except (ValueError, RuntimeError, TypeError):
            raise self._nonconvergence("bubble point") from None
        vapor_values = np.asarray(vapor, dtype=float)
        if vapor_values.size != liquid.size and np.count_nonzero(liquid > 1e-14) == 1:
            vapor_values = liquid.copy()
        point = self._point(model, float(temperature), pressure_pa, liquid, vapor_values)
        return self._boundary_result(request, point)

    def dew_point(self, request: TaskManifest) -> CalculationResult:
        pressure_pa = self._require_pressure(request) * 1000.0
        vapor = np.asarray(self._require_composition(request, "vapor_composition"), dtype=float)
        model = self._build_model(request)
        clapeyron = self._load()
        try:
            temperature, _, _, liquid = clapeyron.dew_temperature(model, pressure_pa, vapor)
        except (ValueError, RuntimeError, TypeError):
            raise self._nonconvergence("dew point") from None
        liquid_values = np.asarray(liquid, dtype=float)
        if liquid_values.size != vapor.size and np.count_nonzero(vapor > 1e-14) == 1:
            liquid_values = vapor.copy()
        point = self._point(model, float(temperature), pressure_pa, liquid_values, vapor)
        return self._boundary_result(request, point)

    def _bubble_at_temperature(
        self,
        model: Any,
        temperature: float,
        liquid: NDArray[np.float64],
    ) -> EquilibriumPoint:
        clapeyron = self._load()
        try:
            pressure_pa, _, _, vapor = clapeyron.bubble_pressure(model, temperature, liquid)
        except (ValueError, RuntimeError, TypeError):
            raise self._nonconvergence("isothermal bubble point") from None
        vapor_values = np.asarray(vapor, dtype=float)
        if vapor_values.size != liquid.size and np.count_nonzero(liquid > 1e-14) == 1:
            vapor_values = liquid.copy()
        return self._point(model, temperature, float(pressure_pa), liquid, vapor_values)

    def isobaric_vle(self, request: TaskManifest) -> CalculationResult:
        self._require_binary(request)
        pressure = self._require_pressure(request)
        points: list[EquilibriumPoint] = []
        for fraction in np.linspace(0.0, 1.0, request.points):
            point_request = request.model_copy(
                update={
                    "calculation_type": "bubble_point",
                    "conditions": request.conditions.model_copy(
                        update={"liquid_composition": [float(fraction), float(1.0 - fraction)]}
                    ),
                }
            )
            points.extend(self.bubble_point(point_request).points)
        return self._curve_result(request, points, pressure_kpa=pressure)

    def isothermal_vle(self, request: TaskManifest) -> CalculationResult:
        self._require_binary(request)
        temperature = self._require_temperature(request)
        model = self._build_model(request)
        points = [
            self._bubble_at_temperature(model, temperature, np.asarray([fraction, 1.0 - fraction], dtype=float))
            for fraction in np.linspace(0.0, 1.0, request.points)
        ]
        return self._curve_result(request, points, temperature=temperature)

    def _curve_result(
        self,
        request: TaskManifest,
        points: list[EquilibriumPoint],
        *,
        temperature: float | None = None,
        pressure_kpa: float | None = None,
    ) -> CalculationResult:
        residual = max(point.equilibrium_residual for point in points)
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            parameter_set_id=self._parameter_set_id,
            points=points,
            temperature_K=temperature,
            pressure_kPa=pressure_kpa,
            phase_state="curve",
            converged=residual <= 2e-6,
            residual=residual,
            iterations=0,
            warnings=self._warnings(),
            backend_version=self.backend_version,
            solver_name="Clapeyron.jl PR / saturation solver",
        )

    def tp_flash(self, request: TaskManifest) -> CalculationResult:
        temperature = self._require_temperature(request)
        pressure_kpa = self._require_pressure(request)
        pressure_pa = pressure_kpa * 1000.0
        feed = np.asarray(self._require_composition(request, "feed_composition"), dtype=float)
        model = self._build_model(request)
        clapeyron = self._load()
        try:
            compositions_raw, phase_moles_raw, _ = clapeyron.tp_flash(model, pressure_pa, temperature, feed)
            compositions = np.asarray(compositions_raw, dtype=float)
            phase_moles = np.asarray(phase_moles_raw, dtype=float)
        except (ValueError, RuntimeError, TypeError):
            raise self._nonconvergence("TP flash") from None
        if compositions.ndim != 2 or phase_moles.shape != compositions.shape:
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "Clapeyron.jl returned an unexpected TP flash phase matrix.",
                "Do not use the result; check pyclapeyron compatibility.",
            )
        amounts = phase_moles.sum(axis=1)
        active = np.flatnonzero(amounts > 1e-12)
        if len(active) not in {1, 2}:
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "Clapeyron.jl returned an unsupported number of active phases.",
                "Do not use the result; this release supports VLE with one or two phases.",
                {"active_phase_count": int(len(active))},
            )
        total = float(amounts[active].sum())
        fractions = amounts[active] / total
        active_compositions = compositions[active]
        phases: list[PhaseResult] = []
        residual = 0.0
        vapor_fraction: float
        if len(active) == 1:
            composition = active_compositions[0]
            try:
                volume = float(clapeyron.volume(model, pressure_pa, temperature, composition, phase="stable"))
            except (ValueError, RuntimeError, TypeError):
                raise self._nonconvergence("single-phase volume classification") from None
            compressibility = pressure_pa * volume / (GAS_CONSTANT * temperature)
            phase_name = "liquid" if compressibility < 0.5 else "vapor"
            vapor_fraction = 1.0 if phase_name == "vapor" else 0.0
            phases.append(PhaseResult(phase=phase_name, fraction=1.0, composition=composition.tolist()))
            phase_state = phase_name
        else:
            try:
                volumes = [
                    float(clapeyron.volume(model, pressure_pa, temperature, composition, phase="stable"))
                    for composition in active_compositions
                ]
            except (ValueError, RuntimeError, TypeError):
                raise self._nonconvergence("phase-volume classification") from None
            liquid_index = int(np.argmin(volumes))
            vapor_index = int(np.argmax(volumes))
            liquid = active_compositions[liquid_index]
            vapor = active_compositions[vapor_index]
            liquid_fraction = float(fractions[liquid_index])
            vapor_fraction = float(fractions[vapor_index])
            phases = [
                PhaseResult(phase="liquid", fraction=liquid_fraction, composition=liquid.tolist()),
                PhaseResult(phase="vapor", fraction=vapor_fraction, composition=vapor.tolist()),
            ]
            residual = self._residual(model, pressure_pa, temperature, liquid, vapor)
            phase_state = "two_phase"
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            parameter_set_id=self._parameter_set_id,
            phases=phases,
            temperature_K=temperature,
            pressure_kPa=pressure_kpa,
            vapor_fraction=vapor_fraction,
            phase_state=phase_state,
            converged=residual <= 2e-6,
            residual=residual,
            iterations=0,
            warnings=[*self._warnings(), "Clapeyron tp_flash performs Gibbs-energy phase-split minimization."],
            backend_version=self.backend_version,
            solver_name="Clapeyron.jl tp_flash / PR",
        )

    def phase_stability(self, request: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "The Clapeyron adapter does not expose a standalone validated TPD result in this release.",
            "Use TP flash for phase split calculation and review its explicit stability warning.",
        )

    def azeotrope(self, request: TaskManifest) -> CalculationResult:
        curve = self.isobaric_vle(request)
        candidates = [
            point
            for point in curve.points[1:-1]
            if max(
                abs(liquid - vapor)
                for liquid, vapor in zip(point.liquid_composition, point.vapor_composition, strict=True)
            )
            <= 1e-3
        ]
        message = (
            "No internal azeotrope candidate met |x-y| <= 1e-3."
            if not candidates
            else "Candidate points require local refinement before engineering use."
        )
        return curve.model_copy(update={"points": candidates, "warnings": [*curve.warnings, message]})

    def lle(self, request: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "Clapeyron LLE is not enabled without an evidence-bearing activity-coefficient parameter set.",
            "Import reviewed NRTL or UNIQUAC parameters before enabling LLE.",
        )

    def infinite_dilution_activity(self, request: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "Clapeyron does not provide infinite-dilution activity coefficients through this adapter.",
            "Use the PGSSI backend for gamma-infinity predictions.",
        )
