"""NRTL production backend.
Uses thermo.NRTL for activity coefficients + built-in parameter database.
"""
from thermo_engine._actcoeff_base import ActCoeffBackend
from thermo_engine.actcoeff_params import lookup_nrtl


class NrtlBackend(ActCoeffBackend):
    model_name = "NRTL"
    version = "nrtl/1.0.0"
    _params_lookup = staticmethod(lookup_nrtl)

    def _gamma(self, xs, T, names, params):
        import numpy as np
        from thermo import NRTL as M
        tau = np.array(params.get("tau_coeffs", np.zeros((len(names), len(names), 6))))
        alp = np.array(params.get("alpha_coeffs", np.zeros((len(names), len(names), 2))))
        if alp.sum() == 0:
            alp = np.zeros((len(names), len(names), 2))
            for i in range(len(names)):
                for j in range(len(names)):
                    if i != j: alp[i, j, 0] = params.get("alpha", 0.3)
        return np.array(M(xs=xs, T=T, tau_coeffs=tau, alpha_coeffs=alp).gammas())
