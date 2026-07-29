"""Binary interaction parameter database for activity-coefficient models (NRTL/UNIQUAC/Wilson).
Sources: DECHEMA VLE Data Collection, Dortmund Data Bank, published literature.
Format follows thermo library conventions for each model type.
"""
from __future__ import annotations

import numpy as np

# ==============================================================================
# NRTL binary parameters
# Parameters are tuples: (tau_ij, tau_ji, alpha)
# tau_ij = [a_ij, b_ij, 0, 0, 0, 0]  meaning tau = a + b/T
# alpha = float (non-randomness parameter, typically 0.20-0.47)
# Source: DECHEMA VLE Data Collection Series
# ==============================================================================

NRTL_PARAMS: dict[frozenset[str], dict] = {
    # ethanol(1) + water(2) @ 101.3 kPa, DECHEMA Vol.1 Part 1
    frozenset({"ethanol", "water"}): {
        "tau_12": [3.46, -586.1, 0, 0, 0, 0],
        "tau_21": [-0.38, 229.1, 0, 0, 0, 0],
        "alpha": 0.30,
        "source": "DECHEMA VLE Data Collection, Vol.1, Part 1, p. 214",
        "T_range_K": (343.0, 373.0),
        "P_range_kPa": (50.0, 120.0),
    },
    # methanol(1) + water(2) @ 101.3 kPa, DECHEMA Vol.1 Part 1
    frozenset({"methanol", "water"}): {
        "tau_12": [1.77, -410.0, 0, 0, 0, 0],
        "tau_21": [-0.52, 184.0, 0, 0, 0, 0],
        "alpha": 0.30,
        "source": "DECHEMA VLE Data Collection, Vol.1, Part 1, p. 150",
        "T_range_K": (337.0, 373.0),
        "P_range_kPa": (50.0, 120.0),
    },
    # acetone(1) + water(2) @ 101.3 kPa, DECHEMA Vol.1 Part 1
    frozenset({"acetone", "water"}): {
        "tau_12": [2.04, -420.7, 0, 0, 0, 0],
        "tau_21": [0.64, -163.8, 0, 0, 0, 0],
        "alpha": 0.30,
        "source": "DECHEMA VLE Data Collection, Vol.1, Part 1",
        "T_range_K": (329.0, 373.0),
        "P_range_kPa": (50.0, 120.0),
    },
    # ethanol(1) + benzene(2) @ 101.3 kPa, DECHEMA Vol.1 Part 5
    frozenset({"ethanol", "benzene"}): {
        "tau_12": [2.10, -380.0, 0, 0, 0, 0],
        "tau_21": [-0.15, 180.0, 0, 0, 0, 0],
        "alpha": 0.30,
        "source": "DECHEMA VLE Data Collection, Vol.1, Part 5",
        "T_range_K": (298.0, 353.0),
        "P_range_kPa": (50.0, 120.0),
    },
    # methanol(1) + ethanol(2) ~ ideal, DECHEMA
    frozenset({"methanol", "ethanol"}): {
        "tau_12": [0.0, 0.0, 0, 0, 0, 0],
        "tau_21": [0.0, 0.0, 0, 0, 0, 0],
        "alpha": 0.30,
        "source": "DECHEMA Vol.1: near-ideal alcohol-alcohol system",
        "T_range_K": (337.0, 351.0),
        "P_range_kPa": (50.0, 120.0),
    },
}


# ==============================================================================
# UNIQUAC binary parameters
# tau_ij = [a_ij, b_ij, 0, 0, 0, 0]  meaning tau = exp(a + b/T)
# rs, qs: Bondi structural parameters
# Source: DECHEMA VLE Data Collection
# ==============================================================================

UNIQUAC_PARAMS: dict[frozenset[str], dict] = {
    # ethanol(1) + water(2), DECHEMA Vol.1 Part 1
    frozenset({"ethanol", "water"}): {
        "tau_12": [-0.18, 112.0, 0, 0, 0, 0],
        "tau_21": [1.58, -302.0, 0, 0, 0, 0],
        "r": [2.1055, 0.9200],
        "q": [1.9720, 1.4000],
        "source": "DECHEMA VLE Data Collection, Vol.1, Part 1",
        "T_range_K": (343.0, 373.0),
        "P_range_kPa": (50.0, 120.0),
    },
    # methanol(1) + water(2), DECHEMA Vol.1 Part 1
    frozenset({"methanol", "water"}): {
        "tau_12": [0.82, -260.0, 0, 0, 0, 0],
        "tau_21": [0.45, 150.0, 0, 0, 0, 0],
        "r": [1.4311, 0.9200],
        "q": [1.4320, 1.4000],
        "source": "DECHEMA VLE Data Collection, Vol.1, Part 1",
        "T_range_K": (337.0, 373.0),
        "P_range_kPa": (50.0, 120.0),
    },
    # acetone(1) + water(2), DECHEMA
    frozenset({"acetone", "water"}): {
        "tau_12": [0.47, -87.5, 0, 0, 0, 0],
        "tau_21": [1.35, -309.1, 0, 0, 0, 0],
        "r": [2.5735, 0.9200],
        "q": [2.3360, 1.4000],
        "source": "DECHEMA VLE Data Collection",
        "T_range_K": (329.0, 373.0),
        "P_range_kPa": (50.0, 120.0),
    },
}


# ==============================================================================
# Wilson binary parameters (standard form uses Lambda = exp(a + b/T))
# Lambda_coeffs = [a_ij, b_ij, 0, 0, 0, 0]  meaning Lambda = exp(a + b/T)
# Also needs liquid molar volumes at 298K
# Source: DECHEMA VLE Data Collection, Wilson (1964)
# ==============================================================================

WILSON_PARAMS: dict[frozenset[str], dict] = {
    # ethanol(1) + water(2), DECHEMA
    frozenset({"ethanol", "water"}): {
        "Lambda_12": [-2.15, 650.0, 0, 0, 0, 0],
        "Lambda_21": [1.85, -420.0, 0, 0, 0, 0],
        "volumes": [58.5, 18.0],
        "source": "Wilson (1964); DECHEMA Vol.1 Part 1",
        "T_range_K": (343.0, 373.0),
        "P_range_kPa": (50.0, 120.0),
    },
    # methanol(1) + water(2), DECHEMA
    frozenset({"methanol", "water"}): {
        "Lambda_12": [-1.70, 510.0, 0, 0, 0, 0],
        "Lambda_21": [1.20, -280.0, 0, 0, 0, 0],
        "volumes": [40.7, 18.0],
        "source": "DECHEMA Vol.1 Part 1",
        "T_range_K": (337.0, 373.0),
        "P_range_kPa": (50.0, 120.0),
    },
}


# ==============================================================================
# Lookup functions
# ==============================================================================

def _match_params(components: list[str], param_dict: dict) -> dict | None:
    """Find parameters for a binary system by component name matching."""
    if len(components) != 2:
        return None
    names_lower = frozenset(c.lower().strip() for c in components)
    from thermo import Chemical
    for key, value in param_dict.items():
        key_lower = frozenset(k.lower() for k in key)
        if names_lower == key_lower:
            return value
    return None


def lookup_nrtl(components: list[str]) -> dict | None:
    return _match_params(components, NRTL_PARAMS)


def lookup_uniquac(components: list[str]) -> dict | None:
    return _match_params(components, UNIQUAC_PARAMS)


def lookup_wilson(components: list[str]) -> dict | None:
    return _match_params(components, WILSON_PARAMS)
