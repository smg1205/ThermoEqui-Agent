"""PGSSI first-class backend: temperature-dependent infinite-dilution activity coefficients.

PGSSI (Physics-Guided 3D Solute-Solvent Interaction framework) predicts
``log-gamma_inf = K1 + K2 / T`` from solute/solvent SMILES and temperature.  This
module wraps a trained PGSSI checkpoint behind the standard
``ThermodynamicBackend`` protocol so that PGSSI sits next to NRTL/UNIQUAC/Wilson
as an independent, first-class model: it does not require any binary
``ParameterSet`` and no VLE backend depends on it.

Numerical policy (repository-wide):
- The LLM never calculates equilibrium values; all numbers come from the
  deterministic PGSSI adapter below and pass ``validate_equilibrium_result``.
- A missing checkpoint, SMILES, or group/geometry dependency is a structured
  ``missing_parameters`` failure, never a synthetic default.
- PGSSI predicts properties (gamma-infinity); it does not fabricate bubble/dew
  point numbers for the VLE backends.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from schemas.domain import (
    CalculationResult,
    FailureType,
    GammaInfinityPoint,
    TaskManifest,
)
from thermo_engine.backend import ThermodynamicBackend
from thermo_engine.errors import ThermoEquiError

logger = logging.getLogger("thermoequi.pgssi")

PGSSI_MODEL_NAME = "PGSSI"


def _checkpoint_from_env() -> Path | None:
    """Read PGSSI_CHECKPOINT at call time so .env loading order does not matter.

    ``apps.api.main`` calls ``load_dotenv()`` after importing the backend
    modules; reading the environment lazily here guarantees the .env values
    are honored regardless of import order.
    """
    value = os.getenv("PGSSI_CHECKPOINT", "").strip()
    if not value:
        return None
    return Path(value)


_REQUIRED_MODULES = ("torch", "torch_geometric", "rdkit")
_ARCHITECTURE_IMPORT_ERRORS: list[str] = []


def _ensure_torch_scatter_compat() -> None:
    """Provide a torch_scatter.scatter_add shim when the C++ extension is absent.

    The PGSSI architecture imports ``from torch_scatter import scatter_add``.
    torch_scatter is a compiled extension that often has no wheel for the
    installed torch version; PyG ships an equivalent ``torch_geometric.utils.
    scatter`` (``reduce="add"``) with a compatible signature, so we expose it
    under the torch_scatter name only when the real package is unavailable.
    """
    import sys

    if "torch_scatter" in sys.modules or importlib.util.find_spec("torch_scatter") is not None:
        return
    try:
        from torch_geometric.utils import scatter as _pyg_scatter

        def _scatter_add(src, index, dim: int = 0, dim_size: int | None = None, fill_value: float = 0.0):
            del fill_value
            return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="add")

        module = type(sys)("torch_scatter")
        module.scatter_add = _scatter_add  # type: ignore[attr-defined]
        sys.modules["torch_scatter"] = module
        logger.debug("Injected torch_scatter compatibility shim via torch_geometric scatter.")
    except ImportError:  # pragma: no cover - torch_geometric itself missing
        _ARCHITECTURE_IMPORT_ERRORS.append("torch_geometric is unavailable for the torch_scatter shim")


def _load_pgssi_architecture() -> Any:
    """Import the PGSSI model architecture from the group repository, if reachable.

    The group repository is not vendored into this package; the environment
    variable ``PGSSI_SRC`` points at its ``src/models/PGSSI`` directory.
    """
    source_dir = os.getenv("PGSSI_SRC", "")
    if not source_dir:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "PGSSI requires the PGSSI source directory.",
            "Set PGSSI_SRC to the PGSSI repository src/models/PGSSI directory.",
        )
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            f"PGSSI_SRC is not a directory: {source_path}",
            "Point PGSSI_SRC at the PGSSI repository src/models/PGSSI directory.",
        )
    import sys

    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    _ensure_torch_scatter_compat()
    try:
        from PGSSI_3D_architecture import PGSSIModel  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - environment dependent
        message = f"PGSSI architecture import failed: {error}"
        _ARCHITECTURE_IMPORT_ERRORS.append(message)
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "PGSSI model architecture is unavailable.",
            "Verify PGSSI_SRC points at the PGSSI src/models/PGSSI directory with its dependencies installed.",
            {"import_error": str(error)},
        ) from error
    return PGSSIModel


def _check_optional_dependencies() -> None:
    missing = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "PGSSI requires optional dependencies that are not installed.",
            "Install torch, torch_geometric, and rdkit in the active environment.",
            {"missing_modules": missing},
        )


@dataclass(frozen=True)
class PgssiSettings:
    """Resolved runtime settings for one PGSSI prediction request."""

    checkpoint_path: Path
    hidden_dim: int
    enable_cross_interaction: bool


def resolve_pgssi_settings() -> PgssiSettings:
    """Resolve PGSSI checkpoint and hyper-parameters from the environment.

    Environment values are read lazily so a later ``load_dotenv()`` in the
    application entrypoint is honored.  Raises a structured
    ``missing_parameters`` failure when the checkpoint is absent, mirroring the
    parameter-missing convention of the repository.
    """
    checkpoint = _checkpoint_from_env()
    if checkpoint is None or not checkpoint.is_file():
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "PGSSI requires a trained checkpoint but none is configured.",
            "Set PGSSI_CHECKPOINT to a trained PGSSI .pth file, or train one with the PGSSI repository.",
            {"model": PGSSI_MODEL_NAME},
        )
    hidden_dim = int(os.getenv("PGSSI_HIDDEN_DIM", "512"))
    enable_cross = os.getenv("PGSSI_ENABLE_CROSS_INTERACTION", "1") not in {"0", "false", "False"}
    return PgssiSettings(
        checkpoint_path=Path(checkpoint),
        hidden_dim=hidden_dim,
        enable_cross_interaction=enable_cross,
    )


class _PgssiPredictor:
    """Lazily loaded PGSSI model instance producing gamma-infinity predictions."""

    def __init__(self, settings: PgssiSettings) -> None:
        self._settings = settings
        self._model: Any = None
        self._device: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _check_optional_dependencies()
        import torch

        model_cls = _load_pgssi_architecture()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_cls(
            hidden_dim=self._settings.hidden_dim,
            enable_cross_interaction=self._settings.enable_cross_interaction,
        ).to(self._device)
        checkpoint = torch.load(self._settings.checkpoint_path, map_location=self._device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        # strict=True: the architecture must match the checkpoint exactly; any
        # mismatch is a hard failure instead of silently wrong predictions.
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._model = model

    def predict(self, solute_smiles: str, solvent_smiles: str, temperatures_k: list[float]) -> list[float]:
        """Return log-gamma_infinity predictions at the requested temperatures."""
        self._ensure_loaded()
        import sys

        source_dir = os.getenv("PGSSI_SRC", "").strip()
        if not source_dir:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "PGSSI requires the PGSSI source directory for data building.",
                "Set PGSSI_SRC to the PGSSI repository src/models/PGSSI directory.",
            )
        source_path = Path(source_dir)
        # PGSSI_data / PGSSI_train import ``src.models.PGSSI.PGSSI_3D_architecture``,
        # so the repository root (parent of src/) must also be importable.
        for entry in (source_path, source_path.parents[2]):
            if str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        _ensure_torch_scatter_compat()

        import pandas as pd
        import torch
        from PGSSI_data import build_pair_dataset  # type: ignore[import-not-found]
        from torch_geometric.loader import DataLoader

        df = pd.DataFrame(
            {
                "Solute_SMILES": [solute_smiles] * len(temperatures_k),
                "Solvent_SMILES": [solvent_smiles] * len(temperatures_k),
                "T": [float(value) - 273.15 for value in temperatures_k],
            }
        )
        cache_dir = Path(os.getenv("PGSSI_CACHE_DIR", str(Path.cwd() / ".pgssi-cache")))
        cache_dir.mkdir(parents=True, exist_ok=True)
        import hashlib

        cache_key = hashlib.sha256(
            (
                f"{solute_smiles}|{solvent_smiles}|{len(temperatures_k)}|"
                f"{min(temperatures_k):.2f}|{max(temperatures_k):.2f}"
            ).encode()
        ).hexdigest()[:16]
        cache_path = cache_dir / f"pred_{cache_key}.pt"
        dataset = build_pair_dataset(
            df,
            str(cache_path),
            include_k_targets=False,
            cache_desc="build PGSSI 3D dataset",
        )
        loader = DataLoader(dataset, batch_size=len(temperatures_k), shuffle=False, num_workers=0)
        predictions: list[float] = []
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                batch = batch.to(self._device)
                out = self._model(batch, return_dict=True)
                predictions.extend(float(value) for value in out["log_gamma"].view(-1).cpu().numpy())
        if len(predictions) != len(temperatures_k):
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "PGSSI returned a mismatched prediction count.",
                "Review the checkpoint and input table; do not use this result.",
            )
        return predictions


class PgssiBackend(ThermodynamicBackend):
    """First-class PGSSI backend exposing gamma-infinity as a calculation type.

    PGSSI predicts infinite-dilution activity coefficients from molecular
    structure and temperature.  It does not implement VLE/flash operations;
    those fail with an explicit structured ``unsupported_model`` error so that
    PGSSI never silently pretends to be a phase-equilibrium solver.
    """

    model_name = PGSSI_MODEL_NAME
    version = "pgssi/1.0.0"
    solver_name = "PGSSI (physics-guided 3D solute-solvent)"

    def __init__(self, settings: PgssiSettings | None = None) -> None:
        self._settings = settings
        self._sources: list[dict[str, str]] = []

    # -- capability ---------------------------------------------------------

    def _resolve_settings(self) -> PgssiSettings:
        if self._settings is None:
            self._settings = resolve_pgssi_settings()
        return self._settings

    def _unsupported(self, operation: str) -> ThermoEquiError:
        return ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            f"PGSSI predicts infinite-dilution activity coefficients and does not implement {operation}.",
            "Use an activity-coefficient or EOS backend for phase-equilibrium calculations.",
            {"model": PGSSI_MODEL_NAME, "operation": operation},
        )

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        del request
        if self._sources:
            return self._sources
        settings = self._resolve_settings()
        self._sources = [
            {
                "model": PGSSI_MODEL_NAME,
                "property": "infinite-dilution activity coefficients (PGSSI prediction)",
                "checkpoint": str(settings.checkpoint_path),
                "source_type": "model_prediction",
                "source_title": "PGSSI trained checkpoint",
                "source_identifier": str(settings.checkpoint_path),
            }
        ]
        return self._sources

    def infinite_dilution_activity(self, request: TaskManifest) -> CalculationResult:
        """Predict gamma-infinity(T) for every solute/solvent pair in the task.

        Components are ordered solute-first: component[0] is the solute and
        component[1] is the solvent for each pair.

        Two modes:
        - Single point: ``conditions.temperature_K`` given (and no span).
        - Curve: ``conditions.temperature_span_K = [low, high]`` given; the
          backend sweeps ``request.points`` temperatures across the span.
        """
        settings = self._resolve_settings()
        components = request.components
        if len(components) < 2:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "PGSSI needs at least a solute and a solvent component.",
                "Provide two components; the first is the solute and the second the solvent.",
            )
        span = request.conditions.temperature_span_K
        single_temperature = request.conditions.temperature_K
        if span is not None:
            low, high = span
            temperatures = np.linspace(low, high, request.points).tolist()
        elif single_temperature is not None:
            temperatures = [single_temperature]
        else:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "PGSSI gamma-infinity prediction requires a temperature.",
                "Provide temperature_K (single point) or temperature_span_K (curve) in the task conditions.",
            )
        predictor = _PgssiPredictor(settings)
        points: list[GammaInfinityPoint] = []
        warnings: list[str] = []
        for solute_index in range(len(components)):
            for solvent_index in range(len(components)):
                if solute_index == solvent_index:
                    continue
                solute = components[solute_index]
                solvent = components[solvent_index]
                if solute.smiles is None or solvent.smiles is None:
                    raise ThermoEquiError(
                        FailureType.MISSING_PARAMETERS,
                        "PGSSI requires SMILES for every component.",
                        "Provide the smiles field on each component identity.",
                        {
                            "model": PGSSI_MODEL_NAME,
                            "missing": [
                                name
                                for name, component in (("solute", solute), ("solvent", solvent))
                                if component.smiles is None
                            ],
                        },
                    )
                try:
                    ln_gamma_values = predictor.predict(solute.smiles, solvent.smiles, temperatures)
                except (KeyError, TypeError, ValueError) as error:
                    raise ThermoEquiError(
                        FailureType.MISSING_PARAMETERS,
                        f"PGSSI could not predict for solute {solute.name!r} in solvent {solvent.name!r}: {error}",
                        "Verify the component identities and PGSSI dependencies.",
                        {"solute": solute.name, "solvent": solvent.name},
                    ) from error
                for temperature, ln_value in zip(temperatures, ln_gamma_values, strict=True):
                    if not np.isfinite(ln_value):
                        raise ThermoEquiError(
                            FailureType.PHYSICAL_VALIDATION_FAILURE,
                            f"PGSSI returned a non-finite gamma-infinity for {solute.name} in {solvent.name}.",
                            "Do not use this result; review the checkpoint and component identities.",
                        )
                    points.append(
                        GammaInfinityPoint(
                            temperature_K=float(temperature),
                            solute_index=solute_index,
                            solvent_index=solvent_index,
                            gamma_infinity=float(np.exp(ln_value)),
                            ln_gamma_infinity=ln_value,
                        )
                    )
        warnings.append("PGSSI is a predictive pilot; benchmark closure and applicability review are pending.")
        return CalculationResult(
            task_id=request.task_id,
            calculation_type="infinite_dilution_activity",
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            gamma_infinity=points,
            temperature_K=float(temperatures[0]) if len(temperatures) == 1 else None,
            converged=True,
            residual=0.0,
            iterations=0,
            warnings=warnings,
            backend_version=self.version,
            solver_name=self.solver_name,
            phase_state="unknown",
        )

    # -- protocol stubs that must fail structurally -------------------------

    def bubble_point(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("bubble_point")

    def dew_point(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("dew_point")

    def isobaric_vle(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("isobaric_vle")

    def isothermal_vle(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("isothermal_vle")

    def tp_flash(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("tp_flash")

    def phase_stability(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("phase_stability")

    def azeotrope(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("azeotrope")

    def lle(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("lle")
