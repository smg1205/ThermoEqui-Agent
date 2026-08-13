"""Wilson production backend.
Uses thermo.Wilson for activity coefficients with reviewed request parameters.
Cannot handle LLE.
"""

from thermo_engine._actcoeff_base import ActCoeffBackend


class WilsonBackend(ActCoeffBackend):
    model_name = "Wilson"
    version = "wilson/1.0.0"

    def _gamma(self, xs, T, names, params):
        import numpy as np
        from thermo import Wilson as M

        lmb = np.array(params.get("Lambda_coeffs", np.zeros((len(names), len(names), 6))))
        return np.array(M(xs=xs, T=T, lambda_coeffs=lmb).gammas())

    def lle(self, req):
        from schemas.domain import FailureType
        from thermo_engine.errors import ThermoEquiError

        raise ThermoEquiError(FailureType.UNSUPPORTED_MODEL, "Wilson cannot represent LLE.", "Use NRTL or UNIQUAC.")
