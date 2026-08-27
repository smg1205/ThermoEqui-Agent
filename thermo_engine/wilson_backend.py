"""Wilson production backend.
Uses thermo.Wilson for activity coefficients with reviewed request parameters.
Cannot handle LLE.
"""

from typing import Any

import numpy as np

from schemas.domain import CalculationResult, FailureType, TaskManifest
from thermo_engine._actcoeff_base import ActCoeffBackend
from thermo_engine.errors import ThermoEquiError


class WilsonBackend(ActCoeffBackend):
    model_name = "Wilson"
    version = "wilson/1.0.0"

    def _gamma(
        self,
        xs: np.ndarray[Any, Any],
        T: float,
        names: list[str],
        params: dict[str, Any] | None,
    ) -> np.ndarray[Any, Any]:
        from thermo import Wilson as M

        assert params is not None
        lmb = np.array(params.get("Lambda_coeffs", np.zeros((len(names), len(names), 6))))
        return np.array(M(xs=xs, T=T, lambda_coeffs=lmb).gammas())

    def lle(self, req: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(FailureType.UNSUPPORTED_MODEL, "Wilson cannot represent LLE.", "Use NRTL or UNIQUAC.")
