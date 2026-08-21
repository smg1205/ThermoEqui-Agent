"""DWSIM Automation export for validated non-electrolyte equilibrium runs.

The module imports pythonnet only when an export is requested.  This keeps the
core calculation package runnable on hosts that do not have DWSIM installed.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from schemas.domain import FailureType, RunRecord
from thermo_engine.errors import ThermoEquiError


class DWSIMAutomation(Protocol):
    """Small adapter surface used from DWSIM's ``Automation3`` API."""

    def CreateFlowsheet(self) -> Any: ...


AutomationFactory = Callable[[], DWSIMAutomation]

_PROPERTY_PACKAGES = {
    "ideal/raoult": "Raoult's Law",
    "peng-robinson": "Peng-Robinson (PR)",
    "phasepy/peng-robinson": "Peng-Robinson (PR)",
    "clapeyron/peng-robinson": "Peng-Robinson (PR)",
    "nrtl": "NRTL",
    "wilson": "Wilson",
    "uniquac": "UNIQUAC",
}


def _runtime_error(message: str, *, details: dict[str, object] | None = None) -> ThermoEquiError:
    return ThermoEquiError(
        FailureType.MISSING_DATA,
        message,
        "Install DWSIM and pythonnet, then set DWSIM_HOME to DWSIM's installation directory.",
        details,
    )


def _automation_factory() -> tuple[AutomationFactory, Any]:
    """Load DWSIM assemblies only for an explicit export request."""

    dwsim_home = os.getenv("DWSIM_HOME")
    if not dwsim_home:
        raise _runtime_error("DWSIM_HOME is not configured.", details={"environment_variable": "DWSIM_HOME"})

    install_dir = Path(dwsim_home).expanduser().resolve()
    automation_dll = install_dir / "DWSIM.Automation.dll"
    if not automation_dll.is_file():
        raise _runtime_error(
            "DWSIM.Automation.dll was not found in DWSIM_HOME.",
            details={"dwsim_home": str(install_dir), "required_file": "DWSIM.Automation.dll"},
        )

    try:
        import clr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise _runtime_error("pythonnet is not installed.", details={"required_package": "pythonnet"}) from exc

    if str(install_dir) not in sys.path:
        sys.path.append(str(install_dir))
    try:
        clr.AddReference(str(automation_dll))
        from DWSIM.Automation import Automation3  # type: ignore[import-not-found]
        from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - requires a local DWSIM installation
        raise _runtime_error(
            "DWSIM assemblies could not be loaded.",
            details={"dwsim_home": str(install_dir), "exception": type(exc).__name__},
        ) from exc
    return Automation3, ObjectType


def _first_value(*values: float | None) -> float | None:
    return next((value for value in values if value is not None), None)


def _flowsheet_values(run: RunRecord) -> tuple[list[str], list[float], float, float, str]:
    snapshot = run.input_snapshot
    components = snapshot.get("components", [])
    names: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("The run snapshot does not contain valid DWSIM component names.")
        name = component.get("name")
        if not isinstance(name, str):
            raise ValueError("The run snapshot does not contain valid DWSIM component names.")
        names.append(name)
    if not names:
        raise ValueError("The run snapshot does not contain valid DWSIM component names.")

    conditions = snapshot.get("conditions", {})
    result = run.result
    points = result.get("points", [])
    first_point = points[0] if points and isinstance(points[0], dict) else {}
    composition = _first_composition(
        conditions.get("feed_composition"),
        conditions.get("liquid_composition"),
        conditions.get("vapor_composition"),
        first_point.get("liquid_composition"),
        first_point.get("vapor_composition"),
    )
    if composition is None or len(composition) != len(names):
        raise ValueError("The run does not contain a complete feed composition for DWSIM export.")

    temperature_K = _first_value(
        result.get("temperature_K"), conditions.get("temperature_K"), first_point.get("temperature_K")
    )
    pressure_kPa = _first_value(
        result.get("pressure_kPa"), conditions.get("pressure_kPa"), first_point.get("pressure_kPa")
    )
    if temperature_K is None or pressure_kPa is None:
        raise ValueError("The run does not contain both temperature and pressure for DWSIM export.")
    model_name = result.get("model_name")
    if not isinstance(model_name, str):
        raise ValueError("The run does not contain a thermodynamic model name.")
    property_package = _PROPERTY_PACKAGES.get(model_name.casefold())
    if property_package is None:
        raise ValueError(f"DWSIM export does not support model '{model_name}'.")
    return names, composition, temperature_K, pressure_kPa, property_package


def _first_composition(*values: object) -> list[float] | None:
    for value in values:
        if isinstance(value, list) and value and all(isinstance(item, float | int) for item in value):
            return [float(item) for item in value]
    return None


def _save_flowsheet(automation: DWSIMAutomation, flowsheet: Any, destination: Path) -> None:
    """Save through the public Automation API, supporting maintained DWSIM variants."""

    last_type_error: TypeError | None = None
    for owner, method_name, args in (
        (automation, "SaveFlowsheet2", (flowsheet, str(destination))),
        (automation, "SaveFlowsheet", (flowsheet, str(destination), True)),
        (automation, "SaveFlowsheet", (flowsheet, str(destination))),
        (flowsheet, "SaveToXML", (str(destination),)),
        (flowsheet, "SaveToFile", (str(destination),)),
    ):
        method = getattr(owner, method_name, None)
        if callable(method):
            try:
                method(*args)
            except TypeError as exc:
                last_type_error = exc
                continue
            else:
                return
    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError("The installed DWSIM Automation API has no supported flowsheet save method.")


def _add_property_package(flowsheet: Any, property_package: str) -> None:
    """Add a property package through the API supported by the installed DWSIM version."""

    create_and_add = getattr(flowsheet, "CreateAndAddPropertyPackage", None)
    if callable(create_and_add):
        create_and_add(property_package)
        return
    flowsheet.AddPropertyPackage(property_package)


def _simulation_object(automation_object: Any) -> Any:
    """Unwrap DWSIM 9's generic simulation-object interface when available."""

    get_as_object = getattr(automation_object, "GetAsObject", None)
    return get_as_object() if callable(get_as_object) else automation_object


def _composition_argument(composition: list[float]) -> Any:
    """Convert compositions to the .NET array required by DWSIM's API."""

    try:
        from System import Array, Double  # type: ignore[import-not-found]
    except ImportError:
        return composition
    return Array[Double](composition)


def export_dwsim_flowsheet(
    run: RunRecord,
    destination: Path,
    *,
    factory: AutomationFactory | None = None,
    object_type: Any | None = None,
) -> Path:
    """Create a DWSIM TP-flash flowsheet from an immutable validated run snapshot.

    The material feed is normalized to a total molar-flow basis of 1.0.
    Temperature is in K and the internal DWSIM pressure call receives Pa,
    converted from the API's kPa.
    """

    names, composition, temperature_K, pressure_kPa, property_package = _flowsheet_values(run)
    destination = destination.resolve()
    if destination.suffix.casefold() != ".dwxmz":
        raise ValueError("DWSIM exports must use the .dwxmz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if factory is None or object_type is None:
        factory, object_type = _automation_factory()
    try:
        automation = factory()
        flowsheet = automation.CreateFlowsheet()
        for name in names:
            flowsheet.AddCompound(name)
        _add_property_package(flowsheet, property_package)

        feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
        separator = flowsheet.AddObject(object_type.Vessel, 250, 0, "Equilibrium Flash")
        vapor = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Vapor Product")
        liquid = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Liquid Product")
        feed_stream = _simulation_object(feed)
        feed_stream.SetTemperature(temperature_K)
        feed_stream.SetPressure(pressure_kPa * 1000.0)
        feed_stream.SetMolarFlow(1.0)
        feed_stream.SetOverallComposition(_composition_argument(composition))
        flowsheet.ConnectObjects(feed.GraphicObject, separator.GraphicObject, 0, 0)
        flowsheet.ConnectObjects(separator.GraphicObject, vapor.GraphicObject, 0, 0)
        flowsheet.ConnectObjects(separator.GraphicObject, liquid.GraphicObject, 1, 0)
        _save_flowsheet(automation, flowsheet, destination)
    except ThermoEquiError:
        raise
    except Exception as exc:  # pragma: no cover - depends on installed DWSIM assemblies
        raise _runtime_error(
            "DWSIM could not create the phase-equilibrium flowsheet.",
            details={"exception": type(exc).__name__},
        ) from exc
    if not destination.is_file():
        raise _runtime_error("DWSIM did not create the requested flowsheet file.")
    return destination
