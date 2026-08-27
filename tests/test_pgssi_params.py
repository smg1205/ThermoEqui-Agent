"""Behavioral tests for the PGSSI-to-NRTL parameter regression bridge."""

from __future__ import annotations

import numpy as np
import pytest

from schemas.domain import FailureType
from thermo_engine.errors import ThermoEquiError
from thermo_engine.pgssi_params import (
    _nrtl_ln_gamma_infinity,
    regress_nrtl_from_gamma_infinity,
)


def test_regression_recovers_parameters_from_synthetic_data() -> None:
    # Synthetic gamma-infinity generated from known NRTL parameters.
    b12, b21 = 300.0, -120.0
    temperatures = np.linspace(300.0, 360.0, 6)
    ln_gamma = np.array([_nrtl_ln_gamma_infinity(b12, b21, float(T)) for T in temperatures])

    parameter_set = regress_nrtl_from_gamma_infinity(
        ["ethanol", "water"],
        temperatures.tolist(),
        ln_gamma.tolist(),
        source_type="estimated",
    )

    assert parameter_set.model_name == "NRTL"
    assert parameter_set.component_order == ["ethanol", "water"]
    assert parameter_set.parameters["alpha"] == pytest.approx(0.3, abs=1e-6)
    assert parameter_set.parameters["tau12_a"] == pytest.approx(0.0, abs=1e-9)
    assert parameter_set.parameters["tau21_a"] == pytest.approx(0.0, abs=1e-9)
    assert parameter_set.parameters["tau12_b"] == pytest.approx(b12, abs=2.0)
    assert parameter_set.parameters["tau21_b"] == pytest.approx(b21, abs=2.0)
    assert parameter_set.temperature_range_K == pytest.approx((300.0, 360.0))


def test_regression_requires_two_distinct_temperatures() -> None:
    with pytest.raises(ThermoEquiError) as captured:
        regress_nrtl_from_gamma_infinity(
            ["ethanol", "water"],
            [300.0],
            [0.5],
        )
    assert captured.value.detail.failure_type == FailureType.MISSING_DATA


def test_regression_rejects_non_finite_inputs() -> None:
    with pytest.raises(ThermoEquiError) as captured:
        regress_nrtl_from_gamma_infinity(
            ["ethanol", "water"],
            [300.0, 310.0],
            [0.5, float("nan")],
        )
    assert captured.value.detail.failure_type == FailureType.MISSING_DATA


def test_regression_rejects_non_binary_component_order() -> None:
    with pytest.raises(ThermoEquiError) as captured:
        regress_nrtl_from_gamma_infinity(
            ["ethanol", "water", "benzene"],
            [300.0, 310.0],
            [0.5, 0.6],
        )
    assert captured.value.detail.failure_type == FailureType.MISSING_DATA
