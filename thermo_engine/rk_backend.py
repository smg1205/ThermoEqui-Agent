"""CalebBell/thermo Redlich-Kwong pilot adapter."""

from __future__ import annotations

from importlib.metadata import version

from thermo import RKMIX
from thermo.eos import RK as PureRK

from thermo_engine.pilot_cubic_eos_backend import PilotKijCubicEosBackend


class ThermoRkBackend(PilotKijCubicEosBackend):
    """RK VLE/flash pilot executed by ``thermo`` with an explicit binary kij."""

    version = f"thermo/{version('thermo')}"
    model_name = "RK"
    solver_name = "thermo.FlashVL / RKMIX"
    eos_class = RKMIX
    pure_eos_class = PureRK
    parameter_form_aliases = frozenset({"rk", "rk kij", "rk binary", "rk binary kij"})
    parameter_property = "RK binary interaction parameter kij"
