"""GHGEAT first-class backend: temperature-dependent infinite-dilution activity coefficients.

GHGEAT (Graph Neural Network with Gibbs-Helmholtz and External Attention)
predicts ``log-gamma_inf = A + B/T`` from solute/solvent SMILES and temperature.
This module wraps a trained GHGEAT checkpoint behind the standard
``ThermodynamicBackend`` protocol so that GHGEAT sits next to NRTL/UNIQUAC/Wilson
as an independent, first-class model: it does not require any binary
``ParameterSet`` and no VLE backend depends on it.

Numerical policy (repository-wide):
- The LLM never calculates equilibrium values; all numbers come from the
  deterministic GHGEAT adapter below and pass ``validate_equilibrium_result``.
- A missing checkpoint, SMILES, or dependency is a structured
  ``missing_parameters`` failure, never a synthetic default.
- GHGEAT predicts properties (gamma-infinity); it does not fabricate bubble/dew
  point numbers for the VLE backends.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
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

logger = logging.getLogger("thermoequi.ghgeat")

GHGEAT_MODEL_NAME = "GHGEAT"


def _checkpoint_from_env() -> Path | None:
    """Read GHGEAT_CHECKPOINT at call time so .env loading order does not matter."""
    value = os.getenv("GHGEAT_CHECKPOINT", "").strip()
    if not value:
        return None
    return Path(value)


_REQUIRED_MODULES = ("torch", "torch_geometric", "rdkit")
_ARCHITECTURE_IMPORT_ERRORS: list[str] = []


def _ensure_torch_scatter_compat() -> None:
    """Provide a torch_scatter.scatter_add / scatter_mean shim when the C++ extension is absent.

    GHGEAT architecture imports ``from torch_scatter import scatter_add, scatter_mean``.
    torch_scatter is a compiled extension that often has no wheel for the
    installed torch version; PyG ships an equivalent ``torch_geometric.utils.
    scatter`` (``reduce="add"`` / ``reduce="mean"``) with a compatible signature,
    so we expose them under the torch_scatter name only when the real package is
    unavailable.
    """
    if "torch_scatter" in sys.modules or importlib.util.find_spec("torch_scatter") is not None:
        return
    try:
        from torch_geometric.utils import scatter as _pyg_scatter

        def _scatter_add(src, index, dim: int = 0, dim_size: int | None = None, fill_value: float = 0.0):
            del fill_value
            return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="add")

        def _scatter_mean(src, index, dim: int = 0, dim_size: int | None = None, out=None):
            del out
            return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")

        module = type(sys)("torch_scatter")
        module.scatter_add = _scatter_add
        module.scatter_mean = _scatter_mean
        sys.modules["torch_scatter"] = module
        logger.debug("Injected torch_scatter compatibility shim via torch_geometric scatter.")
    except ImportError:
        _ARCHITECTURE_IMPORT_ERRORS.append("torch_geometric is unavailable for the torch_scatter shim")


def _load_ghgeat_architecture() -> Any:
    """Import the GHGEAT model architecture class from the group repository, if reachable.

    The group repository is not vendored into this package; the environment
    variable ``GHGEAT_SRC`` points at the GHGEAT-master repository root.
    """
    source_dir = os.getenv("GHGEAT_SRC", "").strip()
    if not source_dir:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "GHGEAT requires the GHGEAT source directory.",
            "Set GHGEAT_SRC to the GHGEAT-master repository root directory.",
        )
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            f"GHGEAT_SRC is not a directory: {source_path}",
            "Point GHGEAT_SRC at the GHGEAT-master repository root directory.",
        )
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    _ensure_torch_scatter_compat()
    try:
        from src.models.GHGEAT import GHGEAT_architecture as _ghgeat_arch
        import torch
        
        # Monkey-patch cat_dummy_graph to handle newer PyTorch Geometric where
        # all attributes live in _store, not __dict__ (the original iterates __dict__)
        _orig_cat_dummy = _ghgeat_arch.cat_dummy_graph
        def _patched_cat_dummy(graph):
            """Replacement for cat_dummy_graph that works with newer PyTorch Geometric."""
            from torch_geometric.data import Data
            num_nodes = graph.x.shape[0]
            x = torch.cat([graph.x, torch.zeros(1, graph.x.shape[1], dtype=graph.x.dtype)], dim=0)
            self_loop = torch.tensor([[num_nodes], [num_nodes]], dtype=torch.long)
            edge_index = torch.cat([graph.edge_index, self_loop], dim=1)
            edge_attr = torch.cat(
                [graph.edge_attr,
                 torch.zeros(1, graph.edge_attr.shape[1], dtype=graph.edge_attr.dtype)],
                dim=0
            )
            new_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            for key in graph.keys():
                if key in ('x', 'edge_index', 'edge_attr'):
                    continue
                value = graph[key]
                if key == 'batch':
                    new_graph.batch = torch.cat(
                        [value, torch.tensor([int(value.max()) + 1], dtype=value.dtype)]
                    )
                elif key == 'inter_hb':
                    new_graph.inter_hb = torch.cat(
                        [value, torch.tensor([0.0], dtype=value.dtype)]
                    )
                elif key == 'y':
                    new_graph.y = torch.cat(
                        [value, torch.zeros(1, *value.shape[1:], dtype=value.dtype)], dim=0
                    )
                elif key == 'temp_features':
                    new_graph.temp_features = torch.cat(
                        [value, torch.zeros(1, value.shape[1], dtype=value.dtype)], dim=0
                    )
                else:
                    setattr(new_graph, key, value)
            return new_graph
        _ghgeat_arch.cat_dummy_graph = _patched_cat_dummy
        GHGEAT = _ghgeat_arch.GHGEAT
    except ImportError as error:
        message = f"GHGEAT architecture import failed: {error}"
        _ARCHITECTURE_IMPORT_ERRORS.append(message)
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "GHGEAT model architecture is unavailable.",
            "Verify GHGEAT_SRC points at the GHGEAT-master repo root with its dependencies installed.",
            {"import_error": str(error)},
        ) from error
    return GHGEAT


def _check_optional_dependencies() -> None:
    missing = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "GHGEAT requires optional dependencies that are not installed.",
            "Install torch, torch_geometric, and rdkit in the active environment.",
            {"missing_modules": missing},
        )


@dataclass(frozen=True)
class GhgeatSettings:
    """Resolved runtime settings for one GHGEAT prediction request."""

    checkpoint_path: Path
    hidden_dim: int
    attention_weight: float


def resolve_ghgeat_settings() -> GhgeatSettings:
    """Resolve GHGEAT checkpoint and hyper-parameters from the environment.

    Environment values are read lazily so a later ``load_dotenv()`` in the
    application entrypoint is honored.  Raises a structured
    ``missing_parameters`` failure when the checkpoint is absent, mirroring the
    parameter-missing convention of the repository.
    """
    checkpoint = _checkpoint_from_env()
    if checkpoint is None or not checkpoint.is_file():
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "GHGEAT requires a trained checkpoint but none is configured.",
            "Set GHGEAT_CHECKPOINT to a trained GHGEAT .pth file, or train one with the GHGEAT repository.",
            {"model": GHGEAT_MODEL_NAME},
        )
    hidden_dim = int(os.getenv("GHGEAT_HIDDEN_DIM", "38"))
    attention_weight = float(os.getenv("GHGEAT_ATTENTION_WEIGHT", "0.8"))
    return GhgeatSettings(
        checkpoint_path=Path(checkpoint),
        hidden_dim=hidden_dim,
        attention_weight=attention_weight,
    )


class _GhgeatPredictor:
    """Lazily loaded GHGEAT model instance producing gamma-infinity predictions."""

    def __init__(self, settings: GhgeatSettings) -> None:
        self._settings = settings
        self._model: Any = None
        self._device: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _check_optional_dependencies()
        import torch

        model_cls = _load_ghgeat_architecture()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model_cls(
            v_in=37,
            e_in=9,
            u_in=3,
            hidden_dim=self._settings.hidden_dim,
            attention_weight=self._settings.attention_weight,
        ).to(self._device)

        checkpoint = torch.load(
            self._settings.checkpoint_path,
            map_location=self._device,
            weights_only=False,
        )

        # Handle attention_weight < 1.0: need a dummy forward pass to initialize
        # lazy input_projection layers before loading the checkpoint.
        if self._settings.attention_weight < 1.0:
            logger.debug("GHGEAT attention_weight < 1.0: running dummy forward to init input_projection layers.")
            model = self._init_input_projection(model)

        load_result = model.load_state_dict(checkpoint, strict=False)
        if load_result.missing_keys:
            logger.debug("GHGEAT checkpoint load missing keys: %d", len(load_result.missing_keys))
        if load_result.unexpected_keys:
            logger.debug("GHGEAT checkpoint load unexpected keys: %d", len(load_result.unexpected_keys))

        model.eval()
        self._model = model

    def _init_input_projection(self, model: Any) -> Any:
        """Run a dummy forward pass to trigger lazy initialization of input_projection layers.

        The GHGEAT NodeModel._build_layers creates input_projection lazily on first
        forward. When attention_weight < 1.0, the checkpoint contains these weights
        but they can't be loaded until the layer exists. Run one sample through the
        model to trigger creation.
        """
        import torch
        from torch_geometric.data import Batch, Data

        dummy = Data(
            x=torch.zeros(1, 37, dtype=torch.float32),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_attr=torch.zeros(1, 9, dtype=torch.float32),
        )
        dummy.batch = torch.tensor([0], dtype=torch.long)
        dummy.ap = torch.tensor([0.0], dtype=torch.float32)
        dummy.bp = torch.tensor([0.0], dtype=torch.float32)
        dummy.topopsa = torch.tensor([0.0], dtype=torch.float32)
        dummy.inter_hb = torch.tensor([0.0], dtype=torch.float32)
        dummy.y = torch.tensor([0.0], dtype=torch.float32)

        batch_solv = Batch.from_data_list([dummy])
        batch_solu = Batch.from_data_list([dummy])
        batch_T = Batch.from_data_list([Data(x=torch.tensor([[25.0]], dtype=torch.float32))])

        model.eval()
        with torch.no_grad():
            _ = model(batch_solv.to(self._device), batch_solu.to(self._device), batch_T.to(self._device))
        return model

    def predict(self, solute_smiles: str, solvent_smiles: str, temperatures_k: list[float]) -> list[float]:
        """Return log-gamma_infinity predictions at the requested temperatures."""
        self._ensure_loaded()
        import torch

        source_dir = os.getenv("GHGEAT_SRC", "").strip()
        if not source_dir:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "GHGEAT requires the GHGEAT source directory for data building.",
                "Set GHGEAT_SRC to the GHGEAT-master repository root directory.",
            )
        source_path = Path(source_dir)
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        if str(source_path / "src") not in sys.path:
            sys.path.insert(0, str(source_path / "src"))
        _ensure_torch_scatter_compat()

        import pandas as pd
        from rdkit import Chem
        from src.models.utilities_v2.mol2graph import (
            get_dataloader_pairs_T,
            sys2graph,
        )

        # Convert Kelvin to Celsius for the GHGEAT model
        temperatures_c = [float(t) - 273.15 for t in temperatures_k]

        df = pd.DataFrame(
            {
                "Solvent_SMILES": [solvent_smiles] * len(temperatures_k),
                "Solute_SMILES": [solute_smiles] * len(temperatures_k),
                "T": temperatures_c,
                "log-gamma": [0.0] * len(temperatures_k),
            }
        )

        df["Molecule_Solvent"] = df["Solvent_SMILES"].apply(Chem.MolFromSmiles)
        df["Molecule_Solute"] = df["Solute_SMILES"].apply(Chem.MolFromSmiles)

        if df["Molecule_Solvent"].iloc[0] is None:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                f"GHGEAT could not parse solvent SMILES: {solvent_smiles!r}",
                "Verify the SMILES string is valid.",
            )
        if df["Molecule_Solute"].iloc[0] is None:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                f"GHGEAT could not parse solute SMILES: {solute_smiles!r}",
                "Verify the SMILES string is valid.",
            )

        graphs_solv, graphs_solu = sys2graph(df, "Molecule_Solvent", "Molecule_Solute", "log-gamma")
        df["g_solv"], df["g_solu"] = graphs_solv, graphs_solu

        indices = df.index.tolist()
        predict_loader = get_dataloader_pairs_T(
            df,
            indices,
            "g_solv",
            "g_solu",
            batch_size=len(temperatures_k),
            shuffle=False,
            drop_last=False,
        )

        predictions: list[float] = []
        with torch.no_grad():
            for batch_solvent, batch_solute, batch_T in predict_loader:
                batch_solvent = batch_solvent.to(self._device)
                batch_solute = batch_solute.to(self._device)
                batch_T = batch_T.to(self._device)
                out = self._model(batch_solvent, batch_solute, batch_T)
                predictions.extend(float(v) for v in out.view(-1).cpu().numpy())

        if len(predictions) != len(temperatures_k):
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "GHGEAT returned a mismatched prediction count.",
                "Review the checkpoint and input table; do not use this result.",
            )
        return predictions


class GhgeatBackend(ThermodynamicBackend):
    """First-class GHGEAT backend exposing gamma-infinity as a calculation type.

    GHGEAT predicts infinite-dilution activity coefficients from molecular
    structure and temperature.  It does not implement VLE/flash operations;
    those fail with an explicit structured ``unsupported_model`` error so that
    GHGEAT never silently pretends to be a phase-equilibrium solver.
    """

    model_name = GHGEAT_MODEL_NAME
    version = "ghgeat/1.0.0"
    solver_name = "GHGEAT (GNN with Gibbs-Helmholtz and External Attention)"

    def __init__(self, settings: GhgeatSettings | None = None) -> None:
        self._settings = settings
        self._sources: list[dict[str, str]] = []

    def _resolve_settings(self) -> GhgeatSettings:
        if self._settings is None:
            self._settings = resolve_ghgeat_settings()
        return self._settings

    def _unsupported(self, operation: str) -> ThermoEquiError:
        return ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            f"GHGEAT predicts infinite-dilution activity coefficients and does not implement {operation}.",
            "Use an activity-coefficient or EOS backend for phase-equilibrium calculations.",
            {"model": GHGEAT_MODEL_NAME, "operation": operation},
        )

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        del request
        if self._sources:
            return self._sources
        settings = self._resolve_settings()
        self._sources = [
            {
                "model": GHGEAT_MODEL_NAME,
                "property": "infinite-dilution activity coefficients (GHGEAT prediction)",
                "checkpoint": str(settings.checkpoint_path),
                "source_type": "model_prediction",
                "source_title": "GHGEAT trained checkpoint",
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
                "GHGEAT needs at least a solute and a solvent component.",
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
                "GHGEAT gamma-infinity prediction requires a temperature.",
                "Provide temperature_K (single point) or temperature_span_K (curve) in the task conditions.",
            )
        predictor = _GhgeatPredictor(settings)
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
                        "GHGEAT requires SMILES for every component.",
                        "Provide the smiles field on each component identity.",
                        {
                            "model": GHGEAT_MODEL_NAME,
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
                        f"GHGEAT could not predict for solute {solute.name!r} in solvent {solvent.name!r}: {error}",
                        "Verify the component identities and GHGEAT dependencies.",
                        {"solute": solute.name, "solvent": solvent.name},
                    ) from error
                for temperature, ln_value in zip(temperatures, ln_gamma_values, strict=True):
                    if not np.isfinite(ln_value):
                        raise ThermoEquiError(
                            FailureType.PHYSICAL_VALIDATION_FAILURE,
                            f"GHGEAT returned a non-finite gamma-infinity for {solute.name} in {solvent.name}.",
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
        warnings.append("GHGEAT is a predictive pilot; benchmark closure and applicability review are pending.")
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
