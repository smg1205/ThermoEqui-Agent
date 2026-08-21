"""CalebBell/thermo Soave-Redlich-Kwong pilot adapter."""

from __future__ import annotations

from importlib.metadata import version

from thermo import SRKMIX
from thermo.eos import SRK as PureSRK

from thermo_engine.pilot_cubic_eos_backend import PilotKijCubicEosBackend


class ThermoSrkBackend(PilotKijCubicEosBackend):
    """SRK VLE/flash pilot executed by ``thermo`` with an explicit binary kij."""

    version = f"thermo/{version('thermo')}"
    model_name = "SRK"
    solver_name = "thermo.FlashVL / SRKMIX"
    eos_class = SRKMIX
    pure_eos_class = PureSRK
    parameter_form_aliases = frozenset({"srk", "srk kij", "srk binary", "srk binary kij"})
    parameter_property = "SRK binary interaction parameter kij"
