"""Behavioral tests for the PGSSI first-class backend.

PGSSI requires a trained checkpoint, the PGSSI source tree, torch/PyG/RDKit, so
these tests exercise the backend contract with mocked prediction internals and
verify structured failures when runtime prerequisites are absent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from schemas.domain import (
    ComponentIdentity,
    FailureType,
    TaskManifest,
    ThermodynamicConditions,
)
from thermo_engine.errors import ThermoEquiError
from thermo_engine.pgssi_backend import (
    PgssiBackend,
    PgssiSettings,
    resolve_pgssi_settings,
)

ETHANOL = ComponentIdentity(
    component_id="ethanol",
    name="ethanol",
    cas_number="64-17-5",
    smiles="CCO",
    aliases=[],
)
WATER = ComponentIdentity(
    component_id="water",
    name="water",
    cas_number="7732-18-5",
    smiles="O",
    aliases=[],
)


def gamma_task() -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="infinite_dilution_activity",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(temperature_K=298.15),
        model_name="PGSSI",
    )


class _FakePredictor:
    def __init__(self, settings: PgssiSettings) -> None:
        self.settings = settings

    def predict(self, solute_smiles: str, solvent_smiles: str, temperatures_k: list[float]) -> list[float]:
        del solute_smiles, solvent_smiles
        return [-1.0 / float(temperature) for temperature in temperatures_k]


def test_missing_checkpoint_fails_structurally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGSSI_CHECKPOINT", raising=False)
    with pytest.raises(ThermoEquiError) as captured:
        resolve_pgssi_settings()
    assert captured.value.detail.failure_type == FailureType.MISSING_PARAMETERS
    assert "checkpoint" in captured.value.detail.message.casefold()


def test_missing_smiles_fails_structurally() -> None:
    task = gamma_task().model_copy(
        update={
            "components": [
                ETHANOL.model_copy(update={"smiles": None}),
                WATER,
            ]
        }
    )
    settings = PgssiSettings(
        checkpoint_path=Path("unused.pth"),
        hidden_dim=512,
        enable_cross_interaction=True,
    )
    backend = PgssiBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.infinite_dilution_activity(task)
    assert captured.value.detail.failure_type == FailureType.MISSING_PARAMETERS
    assert "SMILES" in captured.value.detail.message


def test_single_component_fails_with_missing_data() -> None:
    task = gamma_task().model_copy(update={"components": [ETHANOL]})
    settings = PgssiSettings(
        checkpoint_path=Path("unused.pth"),
        hidden_dim=512,
        enable_cross_interaction=True,
    )
    backend = PgssiBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.infinite_dilution_activity(task)
    assert captured.value.detail.failure_type == FailureType.MISSING_DATA


def test_missing_temperature_fails_with_missing_data() -> None:
    task = gamma_task().model_copy(update={"conditions": ThermodynamicConditions()})
    settings = PgssiSettings(
        checkpoint_path=Path("unused.pth"),
        hidden_dim=512,
        enable_cross_interaction=True,
    )
    backend = PgssiBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.infinite_dilution_activity(task)
    assert captured.value.detail.failure_type == FailureType.MISSING_DATA


def test_prediction_crosses_the_validation_gate_with_mocked_predictor() -> None:
    from thermo_engine.service import validate_equilibrium_result

    settings = PgssiSettings(
        checkpoint_path=Path("unused.pth"),
        hidden_dim=512,
        enable_cross_interaction=True,
    )
    backend = PgssiBackend(settings=settings)
    with patch("thermo_engine.pgssi_backend._PgssiPredictor", _FakePredictor):
        result = backend.infinite_dilution_activity(gamma_task())
    report = validate_equilibrium_result(result)

    assert result.model_name == "PGSSI"
    assert len(result.gamma_infinity) == 2  # ethanol in water, water in ethanol
    assert all(point.gamma_infinity > 0 for point in result.gamma_infinity)
    assert all(point.ln_gamma_infinity < 0 for point in result.gamma_infinity)
    assert report.overall_status in {"passed", "warning"}
    assert report.equilibrium_residual.passed
    assert report.convergence.passed


def test_unsupported_vle_operation_fails_structurally() -> None:
    settings = PgssiSettings(
        checkpoint_path=Path("unused.pth"),
        hidden_dim=512,
        enable_cross_interaction=True,
    )
    backend = PgssiBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.bubble_point(gamma_task())
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL


def test_parameter_sources_are_structured() -> None:
    settings = PgssiSettings(
        checkpoint_path=Path("model.pth"),
        hidden_dim=512,
        enable_cross_interaction=True,
    )
    backend = PgssiBackend(settings=settings)
    sources = backend.parameter_sources(gamma_task())
    assert sources
    assert "checkpoint" in sources[0]
    assert sources[0]["source_type"] == "model_prediction"
