"""UNIQUAC production backend.
Uses thermo.UNIQUAC for activity coefficients with reviewed request parameters.
"""

from typing import Any

import numpy as np

from thermo_engine._actcoeff_base import ActCoeffBackend


class UniquacBackend(ActCoeffBackend):
    model_name = "UNIQUAC"
    version = "uniquac/1.0.0"

    def _gamma(
        self,
        xs: np.ndarray[Any, Any],
        T: float,
        names: list[str],
        params: dict[str, Any] | None,
    ) -> np.ndarray[Any, Any]:
        from thermo import UNIQUAC as M

        assert params is not None
        tau = np.array(params.get("tau_coeffs", np.zeros((len(names), len(names), 6))))
        rs = np.array(params.get("rs", [1.0] * len(names)))
        qs = np.array(params.get("qs", [1.0] * len(names)))
        return np.array(M(xs=xs, rs=rs, qs=qs, T=T, tau_coeffs=tau).gammas())
