"""UNIQUAC production backend.
Uses thermo.UNIQUAC for activity coefficients + built-in parameter database.
"""
from thermo_engine._actcoeff_base import ActCoeffBackend
from thermo_engine.actcoeff_params import lookup_uniquac


class UniquacBackend(ActCoeffBackend):
    model_name = "UNIQUAC"
    version = "uniquac/1.0.0"
    _params_lookup = staticmethod(lookup_uniquac)

    def _gamma(self, xs, T, names, params):
        import numpy as np
        from thermo import UNIQUAC as M
        tau = np.array(params.get("tau_coeffs", np.zeros((len(names), len(names), 6))))
        rs = np.array(params.get("rs", [1.0] * len(names)))
        qs = np.array(params.get("qs", [1.0] * len(names)))
        return np.array(M(xs=xs, rs=rs, qs=qs, T=T, tau_coeffs=tau).gammas())
