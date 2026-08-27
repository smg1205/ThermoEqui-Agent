\"\"\"Behavioral tests for the ThermoFormer first-class backend.

ThermoFormer requires a trained checkpoint, the ThermoFormer source tree, torch/RDKit, so
these tests exercise the backend contract with mocked prediction internals and
verify structured failures when runtime prerequisites are absent.
\"\"\"

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
from thermo_engine.thermoformer_backend import (
    ThermoFormerBackend,
    ThermoFormerSettings,
    resolve_settings,
)

ETHANOL = ComponentIdentity(
    component_id=\"ethanol\",
    name=\"ethanol\",
    cas_number=\"64-17-5\",
    smiles=\"CCO\",
    aliases=[],
)
WATER = ComponentIdentity(
    component_id=\"water\",
    name=\"water\",
    cas_number=\"7732-18-5\",
    smiles=\"O\",
    aliases=[],
)
METHANOL = ComponentIdentity(
    component_id=\"methanol\",
    name=\"methanol\",
    cas_number=\"67-56-1\",
    smiles=\"CO\",
    aliases=[],
)


def _vle_task() -> TaskManifest:
    return TaskManifest(
        equilibrium_type=\"VLE\",
        calculation_type=\"bubble_point\",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(
            temperature_K=298.15,
            liquid_composition=[0.3, 0.7],
        ),
        model_name=\"ThermoFormer\",
        points=5,
    )


def _gamma_task() -> TaskManifest:
    return TaskManifest(
        equilibrium_type=\"VLE\",
        calculation_type=\"infinite_dilution_activity\",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(temperature_K=298.15),
        model_name=\"ThermoFormer\",
    )


# ── scope checks ──────────────────────────────────────────────────────────


def test_too_many_components_fails() -> None:
    \"\"\"ThermoFormer supports up to 3 components.\"\"\"
    components = [
        ComponentIdentity(component_id=f\"c{i}\", name=f\"c{i}\", smiles=f\"C{i}H\", aliases=[])
        for i in range(4)
    ]
    task = _vle_task().model_copy(update={\"components\": components})
    settings = ThermoFormerSettings(
        src_path=Path(\"unused\"),
        checkpoint_path=Path(\"unused.pt\"),
        feature_cache_path=Path(\".cache\"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.bubble_point(task)
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL
    assert \"3\" in captured.value.detail.message or \"components\" in captured.value.detail.message


def test_too_high_pressure_fails() -> None:
    \"\"\"ThermoFormer is validated up to 500 kPa.\"\"\"
    task = _vle_task().model_copy(
        update={\"conditions\": ThermodynamicConditions(pressure_kPa=600.0, liquid_composition=[0.3, 0.7])}
    )
    settings = ThermoFormerSettings(
        src_path=Path(\"unused\"),
        checkpoint_path=Path(\"unused.pt\"),
        feature_cache_path=Path(\".cache\"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.bubble_point(task)
    assert captured.value.detail.failure_type == FailureType.PARAMETER_OUT_OF_DOMAIN


def test_missing_smiles_fails() -> None:
    \"\"\"SMILES is required for every component.\"\"\"
    task = _vle_task().model_copy(
        update={
            \"components\": [
                ETHANOL.model_copy(update={\"smiles\": None}),
                WATER,
            ]
        }
    )
    settings = ThermoFormerSettings(
        src_path=Path(\"unused\"),
        checkpoint_path=Path(\"unused.pt\"),
        feature_cache_path=Path(\".cache\"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.bubble_point(task)
    assert captured.value.detail.failure_type == FailureType.MISSING_PARAMETERS
    assert \"SMILES\" in captured.value.detail.message


def test_missing_checkpoint_env_fails() -> None:
    \"\"\"resolve_settings() raises MISSING_PARAMETERS when THERMOFORMER_CHECKPOINT is unset.\"\"\"
    import os
    os.environ.pop(\"THERMOFORMER_CHECKPOINT\", None)
    os.environ.pop(\"THERMOFORMER_SRC\", None)
    with pytest.raises(ThermoEquiError) as captured:
        resolve_settings()
    assert captured.value.detail.failure_type == FailureType.MISSING_PARAMETERS


# ── unsupported operations ────────────────────────────────────────────────


def test_unsupported_dew_point_fails() -> None:
    settings = ThermoFormerSettings(
        src_path=Path(\"unused\"),
        checkpoint_path=Path(\"unused.pt\"),
        feature_cache_path=Path(\".cache\"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.dew_point(_vle_task())
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL


def test_unsupported_tp_flash_fails() -> None:
    settings = ThermoFormerSettings(
        src_path=Path(\"unused\"),
        checkpoint_path=Path(\"unused.pt\"),
        feature_cache_path=Path(\".cache\"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.tp_flash(_vle_task())
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL


def test_unsupported_azeotrope_fails() -> None:
    settings = ThermoFormerSettings(
        src_path=Path(\"unused\"),
        checkpoint_path=Path(\"unused.pt\"),
        feature_cache_path=Path(\".cache\"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.azeotrope(_vle_task())
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL


def test_unsupported_lle_fails() -> None:
    settings = ThermoFormerSettings(
        src_path=Path(\"unused\"),
        checkpoint_path=Path(\"unused.pt\"),
        feature_cache_path=Path(\".cache\"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.lle(_vle_task())
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL


# ── composition sweep helper ──────────────────────────────────────────────


def test_build_composition_sweep_binary() -> None:
    from thermo_engine.thermoformer_backend import _build_composition_sweep

    comps = _build_composition_sweep(2, 5)
    assert len(comps) == 3  # n_points=5 → 3 interior points
    for c in comps:
        assert len(c) == 2
        assert abs(sum(c) - 1.0) < 1e-10
        assert all(x > 0.0 for x in c)


def test_build_composition_sweep_ternary() -> None:
    from thermo_engine.thermoformer_backend import _build_composition_sweep

    comps = _build_composition_sweep(3, 10)
    assert len(comps) > 0
    for c in comps:
        assert len(c) == 3
        assert abs(sum(c) - 1.0) < 1e-10
        assert all(x > 0.0 for x in c)


def test_build_composition_sweep_unary_returns_empty() -> None:
    from thermo_engine.thermoformer_backend import _build_composition_sweep

    comps = _build_composition_sweep(1, 5)
    assert comps == []


# ── parameter sources ─────────────────────────────────────────────────────


def test_parameter_sources_are_structured() -> None:
    settings = ThermoFormerSettings(
        src_path=Path(\"model_src\"),
        checkpoint_path=Path(\"model.pt\"),
        feature_cache_path=Path(\".cache\"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    sources = backend.parameter_sources(_vle_task())
    assert sources
    assert \"checkpoint\" in sources[0]
    assert sources[0][\"source_type\"] == \"model_prediction\"
