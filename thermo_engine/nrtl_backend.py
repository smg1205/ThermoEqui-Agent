"""NRTL production backend.
Uses thermo.NRTL for activity coefficients with reviewed request parameters.
"""

from typing import Any

import numpy as np

from thermo_engine._actcoeff_base import ActCoeffBackend


class NrtlBackend(ActCoeffBackend):
    model_name = "NRTL"
    version = "nrtl/1.0.0"

    def _gamma(
        self,
        xs: np.ndarray[Any, Any],
        T: float,
        names: list[str],
        params: dict[str, Any] | None,
    ) -> np.ndarray[Any, Any]:
        from thermo import NRTL as M

        assert params is not None
        tau = np.array(params.get("tau_coeffs", np.zeros((len(names), len(names), 6))))
        alp = np.array(params.get("alpha_coeffs", np.zeros((len(names), len(names), 2))))
        if alp.sum() == 0:
            alp = np.zeros((len(names), len(names), 2))
            for i in range(len(names)):
                for j in range(len(names)):
                    if i != j:
                        alp[i, j, 0] = params.get("alpha", 0.3)
        return np.array(M(xs=xs, T=T, tau_coeffs=tau, alpha_coeffs=alp).gammas())
