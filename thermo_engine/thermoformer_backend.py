"""ThermoFormer predictive backend: neural VLE from molecular structure.

ThermoFormer uses a thermodynamically structured Transformer to predict
activity coefficients and vapor-liquid equilibrium from molecular features
(RDKit descriptors + Uni-Mol v2 embeddings + SMARTS functional groups)
without requiring binary interaction parameters.

Scope: binary and ternary mixtures, pressure ≤ 500 kPa.

Numerical policy (repository-wide):
- The LLM never calculates equilibrium values; all numbers come from the
  deterministic ThermoFormer adapter below and pass ``validate_equilibrium_result``.
- A missing checkpoint, SMILES, or feature dependency is a structured
  ``missing_parameters`` failure, never a synthetic default.
- ThermoFormer predicts VLE and activity coefficients; it does not fabricate
  parameters for the classical backends.
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
    EquilibriumPoint,
    FailureType,
    GammaInfinityPoint,
    TaskManifest,
)
from thermo_engine.activity_coeff_utils import build_result
from thermo_engine.backend import ThermodynamicBackend
from thermo_engine.errors import ThermoEquiError

logger = logging.getLogger("thermoequi.thermoformer")


# ── Windows DLL workaround for rdkit ────────────────────────────────────
# On Windows, pip/conda rdkit may fail to load rdMolDraw2D.pyd because the
# conda environment's Library/bin directory is not in the DLL search path.
# We patch sys.modules with a mock before the import chain (unimol_tools →
# PandasTools → Draw → rdMolDraw2D) can trigger the DLL load failure.


def _patch_rdkit_dll_on_windows() -> None:
    """Pre-empt rdMolDraw2D DLL load failure on Windows by providing a mock.

    Must be called before any import of ``rdkit.Chem.Draw`` or any package
    that transitively imports it (e.g. ``unimol_tools``, ``PandasTools``).
    Safe to call multiple times; the mock is installed only once.
    """
    if sys.platform != "win32":
        return
    if "rdkit.Chem.Draw.rdMolDraw2D" in sys.modules:
        return  # already patched

    # Add the conda Library/bin directory to the DLL search path.
    _lib_bin = os.path.join(
        os.path.dirname(sys.executable), "Library", "bin"
    )
    _lib_bin = os.path.normpath(_lib_bin)
    if os.path.isdir(_lib_bin):
        os.environ.setdefault("PATH", "")
        os.environ["PATH"] = _lib_bin + os.pathsep + os.environ["PATH"]

    # Build a mock module that rdkit.Chem.Draw.__init__ can import without
    # triggering the actual DLL load.  The real rdMolDraw2D is never needed
    # at runtime by the ThermoFormer pipeline (Draw is only used by
    # PandasTools for rendering, which we never invoke).
    import types as _types

    _mock = _types.ModuleType("rdkit.Chem.Draw.rdMolDraw2D")
    _mock.__file__ = "<mock-rdMolDraw2D>"
    _mock.__package__ = "rdkit.Chem.Draw"
    _mock.__path__ = []
    _mock.__all__ = []

    # Stub the names that are imported at module scope in
    # rdkit/Chem/Draw/__init__.py
    _mock.MolDraw2D = object
    _mock.MolDraw2DSVG = object
    _mock.MolDraw2DCairo = object
    _mock.MolDraw2DUtils = object
    _mock.PrepareMolForDrawing = staticmethod(lambda x: x)
    _mock.SetComicMode = staticmethod(lambda x: None)

    sys.modules["rdkit.Chem.Draw.rdMolDraw2D"] = _mock
    logger.debug("Installed mock rdMolDraw2D (Windows DLL workaround)")


# Apply the patch immediately so that any subsequent import of
# unimol_tools / rdkit.Chem.PandasTools does not crash.
_patch_rdkit_dll_on_windows()

THERMOFORMER_MODEL_NAME = "ThermoFormer"
_REQUIRED_MODULES = ("torch", "rdkit")
_MAX_PRESSURE_KPA = 500.0
_MAX_COMPONENTS = 3


# ── environment resolution ────────────────────────────────────────────────


def _src_from_env() -> Path | None:
    value = os.getenv("THERMOFORMER_SRC", "").strip()
    if not value:
        return None
    return Path(value)


def _checkpoint_from_env() -> Path | None:
    value = os.getenv("THERMOFORMER_CHECKPOINT", "").strip()
    if not value:
        return None
    return Path(value)


def _feature_cache_from_env() -> Path | None:
    value = os.getenv("THERMOFORMER_FEATURE_CACHE", "").strip()
    if not value:
        return None
    return Path(value)


def _use_cuda() -> bool:
    value = os.getenv("THERMOFORMER_USE_CUDA", "1").strip()
    return value not in {"0", "false", "False"}


@dataclass(frozen=True)
class ThermoFormerSettings:
    """Resolved runtime settings for one ThermoFormer prediction request."""

    src_path: Path
    checkpoint_path: Path
    feature_cache_path: Path
    use_cuda: bool


def resolve_settings() -> ThermoFormerSettings:
    """Resolve ThermoFormer paths from the environment.

    Raises a structured ``missing_parameters`` failure when checkpoint or
    source directory is absent, mirroring the PGSSI convention.
    """
    src = _src_from_env()
    if src is None or not src.is_dir():
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "ThermoFormer requires the source directory but THERMOFORMER_SRC is not set.",
            "Set THERMOFORMER_SRC to the ThermoFormer repository src directory.",
            {"model": THERMOFORMER_MODEL_NAME},
        )
    checkpoint = _checkpoint_from_env()
    if checkpoint is None or not checkpoint.is_file():
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "ThermoFormer requires a trained checkpoint but THERMOFORMER_CHECKPOINT is not set.",
            "Set THERMOFORMER_CHECKPOINT to a trained ThermoFormer .pt file.",
            {"model": THERMOFORMER_MODEL_NAME},
        )
    cache = _feature_cache_from_env()
    if cache is None:
        cache = Path.cwd() / ".thermoformer-cache"
    cache.mkdir(parents=True, exist_ok=True)
    return ThermoFormerSettings(
        src_path=src,
        checkpoint_path=checkpoint,
        feature_cache_path=cache,
        use_cuda=_use_cuda(),
    )


# ── optional dependency check ─────────────────────────────────────────────


def _check_optional_dependencies() -> None:
    missing = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "ThermoFormer requires optional dependencies that are not installed.",
            "Install torch and rdkit in the active environment via the thermoformer optional group.",
            {"missing_modules": missing},
        )


_ARCHITECTURE_IMPORT_ERRORS: list[str] = []


def _import_thermoformer_architecture(src_path: Path) -> tuple[Any, Any, Any, Any]:
    """Import the ThermoFormer model classes from the source tree.

    Returns (ThermoFormerConfig, ThermoFormer, solve_isothermal, solve_isobaric).
    """
    src_str = str(src_path)
    if src_str not in sys.path:
        import thermo  # noqa: F811 ensure thermo package is cached before adding src path
        sys.path.insert(0, src_str)

    try:
        from thermoformer.models.thermoformer import (  # type: ignore[import-not-found]
            ThermoFormer,
            ThermoFormerConfig,
        )
        from thermoformer.thermodynamics.vle_solver import (  # type: ignore[import-not-found]
            solve_isothermal,
            solve_isobaric,
        )
    except ImportError as error:
        message = f"ThermoFormer architecture import failed: {error}"
        _ARCHITECTURE_IMPORT_ERRORS.append(message)
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "ThermoFormer model architecture is unavailable.",
            "Verify THERMOFORMER_SRC points at the ThermoFormer src directory with its dependencies installed.",
            {"import_error": str(error)},
        ) from error

    return ThermoFormerConfig, ThermoFormer, solve_isothermal, solve_isobaric


# ── molecular feature encoder ─────────────────────────────────────────────


class _ThermoFormerFeatureEncoder:
    """Lazy-loaded molecular feature encoder wrapping ThermoFormer's pipeline."""

    def __init__(self, settings: ThermoFormerSettings) -> None:
        self._settings = settings
        self._encoder: Any = None
        self._feature_dimensions: dict[str, int] = {}

    def _ensure_loaded(self) -> None:
        if self._encoder is not None:
            return
        _check_optional_dependencies()
        import torch

        src_path = self._settings.src_path
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

        try:
            from thermoformer.features.fusion import (  # type: ignore[import-not-found]
                HybridMolecularEncoder,
            )
        except ImportError as error:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                f"ThermoFormer feature encoder import failed: {error}",
                "Verify that THERMOFORMER_SRC contains the full ThermoFormer source tree.",
            ) from error

        cache_path = self._settings.feature_cache_path / "hybrid_molecular_features.npz"
        use_cuda = self._settings.use_cuda and torch.cuda.is_available()

        self._encoder = HybridMolecularEncoder(
            cache_path=cache_path,
            batch_size=16,
            model_size="84m",
            use_cuda=use_cuda,
            use_rdkit_descriptors=False,
            use_unimol=True,
            use_functional_groups=False,
        )

    def encode(self, smiles_list: list[str]) -> np.ndarray:
        """Return a [components, feature_dim] feature array for the given SMILES."""
        self._ensure_loaded()
        result = self._encoder.encode(smiles_list)
        stacked = np.stack([result[smiles] for smiles in smiles_list])
        self._feature_dimensions = self._encoder.feature_block_sizes
        return stacked


# ── model predictor ────────────────────────────────────────────────────────


class _ThermoFormerPredictor:
    """Lazily loaded ThermoFormer model instance producing VLE predictions."""

    def __init__(self, settings: ThermoFormerSettings) -> None:
        self._settings = settings
        self._model: Any = None
        self._device: Any = None
        self._solver_isothermal: Any = None
        self._solver_isobaric: Any = None
        self._feature_encoder: _ThermoFormerFeatureEncoder | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _check_optional_dependencies()
        import torch

        self._device = torch.device(
            "cuda" if self._settings.use_cuda and torch.cuda.is_available() else "cpu"
        )
        logger.debug("ThermoFormer device: %s", self._device)

        ThermoFormerConfig, ThermoFormer, solve_iso, solve_ibar = (
            _import_thermoformer_architecture(self._settings.src_path)
        )
        self._solver_isothermal = solve_iso
        self._solver_isobaric = solve_ibar

        checkpoint = torch.load(
            self._settings.checkpoint_path, map_location=self._device, weights_only=False
        )

        # Extract config from checkpoint
        config_data = checkpoint.get("config", checkpoint.get("model_config", {}))
        if isinstance(config_data, dict):
            config = ThermoFormerConfig(**config_data)
        elif isinstance(config_data, ThermoFormerConfig):
            config = config_data
        else:
            config = ThermoFormerConfig()
            logger.warning("No config found in checkpoint; using defaults.")

        model = ThermoFormer(config).to(self._device)
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._model = model
        self._feature_encoder = _ThermoFormerFeatureEncoder(self._settings)

    def encode_molecules(self, smiles_list: list[str]) -> np.ndarray:
        """Encode SMILES into feature vectors using the ThermoFormer pipeline."""
        self._ensure_loaded()
        assert self._feature_encoder is not None
        return self._feature_encoder.encode(smiles_list)

    def predict_bubble_isothermal(
        self,
        molecules: np.ndarray,
        temperature_k: float,
        x: list[float],
        mask: list[bool],
    ) -> dict[str, Any]:
        """Predict bubble pressure and vapor composition at fixed T."""
        self._ensure_loaded()
        import torch

        mol_tensor = torch.from_numpy(molecules).float().unsqueeze(0).to(self._device)
        temp_tensor = torch.full((1, 1), temperature_k, device=self._device)
        x_tensor = torch.tensor([x], device=self._device)
        mask_tensor = torch.tensor([mask], device=self._device)

        state = self._solver_isothermal(
            self._model, mol_tensor, temp_tensor, x_tensor, mask_tensor,
            iterations=24, strict=False,
        )
        return {
            "pressure_kpa": float(state.calculated_pressure_kpa.squeeze().detach().cpu().numpy()),
            "y": state.y.squeeze().detach().cpu().numpy().tolist(),
            "gamma": state.gamma.squeeze().detach().cpu().numpy().tolist(),
            "psat_kpa": state.psat_kpa.squeeze().detach().cpu().numpy().tolist(),
            "residual": float(state.pressure_residual_kpa.squeeze().detach().cpu().numpy()),
            "converged": (
                bool(state.converged.squeeze().detach().cpu().numpy())
                if state.converged is not None else True
            ),
            "iterations": state.iterations,
        }

    def predict_bubble_isobaric(
        self,
        molecules: np.ndarray,
        pressure_kpa: float,
        x: list[float],
        mask: list[bool],
    ) -> dict[str, Any]:
        """Predict bubble temperature and vapor composition at fixed P."""
        self._ensure_loaded()
        import torch

        mol_tensor = torch.from_numpy(molecules).float().unsqueeze(0).to(self._device)
        pres_tensor = torch.full((1, 1), pressure_kpa, device=self._device)
        x_tensor = torch.tensor([x], device=self._device)
        mask_tensor = torch.tensor([mask], device=self._device)

        state = self._solver_isobaric(
            self._model, mol_tensor, pres_tensor, x_tensor, mask_tensor,
            iterations=16, strict=False,
        )
        return {
            "temperature_k": float(state.temperature_k.squeeze().detach().cpu().numpy()),
            "y": state.y.squeeze().detach().cpu().numpy().tolist(),
            "gamma": state.gamma.squeeze().detach().cpu().numpy().tolist(),
            "psat_kpa": state.psat_kpa.squeeze().detach().cpu().numpy().tolist(),
            "residual": float(state.pressure_residual_kpa.squeeze().detach().cpu().numpy()),
            "converged": (
                bool(state.converged.squeeze().detach().cpu().numpy())
                if state.converged is not None else True
            ),
            "iterations": state.iterations,
        }


# ── scope helpers ─────────────────────────────────────────────────────────


def _check_system_scope(components: list[Any], pressure_kpa: float | None) -> None:
    """Raise if the system is outside ThermoFormer's validated scope."""
    if len(components) > _MAX_COMPONENTS:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            f"ThermoFormer supports up to {_MAX_COMPONENTS} components, "
            f"but {len(components)} were provided.",
            "Use a classical activity-coefficient or EOS backend for this system.",
            {"model": THERMOFORMER_MODEL_NAME, "component_count": len(components)},
        )
    if pressure_kpa is not None and pressure_kpa > _MAX_PRESSURE_KPA:
        raise ThermoEquiError(
            FailureType.PARAMETER_OUT_OF_DOMAIN,
            f"ThermoFormer is validated up to {_MAX_PRESSURE_KPA} kPa, "
            f"but {pressure_kpa:.1f} kPa was requested.",
            "Use an equation-of-state backend for this pressure regime.",
            {"model": THERMOFORMER_MODEL_NAME, "pressure_kPa": pressure_kpa},
        )


def _smiles_from_components(components: list[Any]) -> list[str]:
    """Extract SMILES from component identities, raising if any are missing."""
    smiles_list: list[str] = []
    for component in components:
        if component.smiles is None:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                f"ThermoFormer requires SMILES for component {component.name!r}.",
                "Provide the smiles field on each component identity.",
                {"model": THERMOFORMER_MODEL_NAME, "component": component.name},
            )
        smiles_list.append(component.smiles)
    return smiles_list


def _build_composition_sweep(
    n_components: int, n_points: int
) -> list[list[float]]:
    """Build a composition sweep for binary or ternary systems.

    For binary: x1 sweeps composition space, x2 = 1 - x1.
    For ternary: triangular grid, x1 + x2 + x3 = 1.
    Pure-component endpoints are excluded (gamma is singular at x=0).
    """
    if n_components == 2:
        compositions: list[list[float]] = []
        for i in range(1, n_points - 1):
            x1 = i / (n_points - 1)
            compositions.append([x1, 1.0 - x1])
        return compositions
    if n_components == 3:
        grid_size = max(4, int(np.sqrt(n_points * 2)))
        compositions = []
        for i in range(1, grid_size):
            for j in range(1, grid_size - i):
                x1 = i / (grid_size - 1)
                x2 = j / (grid_size - 1)
                x3 = 1.0 - x1 - x2
                if x3 > 0.0:
                    compositions.append([x1, x2, x3])
        return compositions
    return []


# ── backend ────────────────────────────────────────────────────────────────


class ThermoFormerBackend(ThermodynamicBackend):
    """First-class ThermoFormer backend for neural VLE prediction.

    ThermoFormer predicts bubble-point VLE (P-x-y, T-x-y) and activity
    coefficients from molecular structure alone. It does not require binary
    interaction parameters.

    Supported calculations: bubble_point, isothermal_vle, isobaric_vle,
    infinite_dilution_activity.
    """

    model_name = THERMOFORMER_MODEL_NAME
    version = "thermoformer/0.1.0"
    solver_name = "ThermoFormer (thermodynamically structured neural model)"

    def __init__(self, settings: ThermoFormerSettings | None = None) -> None:
        self._settings = settings
        self._sources: list[dict[str, str]] = []
        self._predictor: _ThermoFormerPredictor | None = None

    # ── internal helpers ──────────────────────────────────────────────────

    def _resolve_settings(self) -> ThermoFormerSettings:
        if self._settings is None:
            self._settings = resolve_settings()
        return self._settings

    def _get_predictor(self) -> _ThermoFormerPredictor:
        if self._predictor is None:
            self._predictor = _ThermoFormerPredictor(self._resolve_settings())
        return self._predictor

    def _unsupported(self, operation: str) -> ThermoEquiError:
        return ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            f"ThermoFormer predicts VLE and activity coefficients "
            f"and does not implement {operation}.",
            "Use an activity-coefficient or EOS backend for this operation.",
            {"model": THERMOFORMER_MODEL_NAME, "operation": operation},
        )

    def _build_equilibrium_points(
        self,
        request: TaskManifest,
        compositions: list[list[float]],
        temperature_k: float | None,
        pressure_kpa: float | None,
    ) -> list[EquilibriumPoint]:
        """Build equilibrium points for a composition sweep."""
        smiles_list = _smiles_from_components(request.components)
        n_components = len(smiles_list)
        mask = [True] * n_components
        predictor = self._get_predictor()
        molecules = predictor.encode_molecules(smiles_list)

        points: list[EquilibriumPoint] = []
        is_isothermal = temperature_k is not None

        for x in compositions:
            if len(x) != n_components:
                continue
            if is_isothermal:
                result = predictor.predict_bubble_isothermal(
                    molecules, temperature_k, x, mask
                )
                points.append(
                    EquilibriumPoint(
                        temperature_K=temperature_k,
                        pressure_kPa=result["pressure_kpa"],
                        liquid_composition=x,
                        vapor_composition=result["y"],
                        equilibrium_residual=abs(result["residual"]),
                    )
                )
            else:
                result = predictor.predict_bubble_isobaric(
                    molecules, pressure_kpa, x, mask
                )
                points.append(
                    EquilibriumPoint(
                        temperature_K=result["temperature_k"],
                        pressure_kPa=pressure_kpa,
                        liquid_composition=x,
                        vapor_composition=result["y"],
                        equilibrium_residual=abs(result["residual"]),
                    )
                )

        return points

    # ── parameter sources ─────────────────────────────────────────────────

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        del request
        if self._sources:
            return self._sources
        settings = self._resolve_settings()
        self._sources = [
            {
                "model": THERMOFORMER_MODEL_NAME,
                "property": "VLE and activity coefficients "
                           "(ThermoFormer prediction)",
                "checkpoint": str(settings.checkpoint_path),
                "source_type": "model_prediction",
                "source_title": "ThermoFormer trained checkpoint",
                "source_identifier": str(settings.checkpoint_path),
            }
        ]
        return self._sources

    # ── calculation methods ───────────────────────────────────────────────

    def bubble_point(self, request: TaskManifest) -> CalculationResult:
        """Single-point bubble calculation.

        Isothermal: provide conditions.temperature_K → predicts P + y.
        Isobaric: provide conditions.pressure_kPa → predicts T + y.
        """
        _check_system_scope(request.components, request.conditions.pressure_kPa)
        smiles_list = _smiles_from_components(request.components)
        n_components = len(smiles_list)
        mask = [True] * n_components
        predictor = self._get_predictor()
        molecules = predictor.encode_molecules(smiles_list)

        temperature_k = request.conditions.temperature_K
        pressure_kpa = request.conditions.pressure_kPa
        x = request.conditions.liquid_composition

        if x is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "ThermoFormer bubble_point requires liquid_composition.",
                "Provide liquid_composition in the task conditions.",
            )
        if len(x) != n_components:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                f"ThermoFormer expects {n_components} composition values, "
                f"got {len(x)}.",
                "Provide a composition matching the component count.",
            )

        if temperature_k is not None:
            result = predictor.predict_bubble_isothermal(
                molecules, temperature_k, x, mask
            )
            point = EquilibriumPoint(
                temperature_K=temperature_k,
                pressure_kPa=result["pressure_kpa"],
                liquid_composition=x,
                vapor_composition=result["y"],
                equilibrium_residual=abs(result["residual"]),
            )
            return build_result(
                request, self.model_name, self.version,
                T=temperature_k, P=result["pressure_kpa"],
                residual=abs(result["residual"]),
                iters=result["iterations"], points=[point],
                warnings=self._warnings(),
                solver_name=self.solver_name,
                phase_state="two_phase",
            )
        elif pressure_kpa is not None:
            result = predictor.predict_bubble_isobaric(
                molecules, pressure_kpa, x, mask
            )
            point = EquilibriumPoint(
                temperature_K=result["temperature_k"],
                pressure_kPa=pressure_kpa,
                liquid_composition=x,
                vapor_composition=result["y"],
                equilibrium_residual=abs(result["residual"]),
            )
            return build_result(
                request, self.model_name, self.version,
                T=result["temperature_k"], P=pressure_kpa,
                residual=abs(result["residual"]),
                iters=result["iterations"], points=[point],
                warnings=self._warnings(),
                solver_name=self.solver_name,
                phase_state="two_phase",
            )
        else:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "ThermoFormer bubble_point requires either temperature_K "
                "(isothermal) or pressure_kPa (isobaric).",
                "Provide at least one of temperature_K or pressure_kPa "
                "in the task conditions.",
            )

    def isothermal_vle(self, request: TaskManifest) -> CalculationResult:
        """Full P-x-y curve at fixed temperature."""
        _check_system_scope(request.components, request.conditions.pressure_kPa)
        temperature_k = request.conditions.temperature_K
        if temperature_k is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "ThermoFormer isothermal_vle requires temperature_K.",
                "Provide temperature_K in the task conditions.",
            )

        n_components = len(request.components)
        compositions = _build_composition_sweep(n_components, request.points)
        if not compositions:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "No valid compositions could be generated for the sweep.",
                "Check component count and point count.",
            )

        points = self._build_equilibrium_points(
            request, compositions,
            temperature_k=temperature_k, pressure_kpa=None,
        )

        if not points:
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                "ThermoFormer isothermal_vle produced no valid points.",
                "Review the temperature and component identities.",
            )

        pressures = [p.pressure_kPa for p in points]
        return build_result(
            request, self.model_name, self.version,
            T=temperature_k, P=float(np.mean(pressures)),
            residual=max(p.equilibrium_residual for p in points),
            iters=0, points=points,
            warnings=self._warnings(),
            solver_name=self.solver_name,
            phase_state="curve",
        )

    def isobaric_vle(self, request: TaskManifest) -> CalculationResult:
        """Full T-x-y curve at fixed pressure."""
        _check_system_scope(request.components, request.conditions.pressure_kPa)
        pressure_kpa = request.conditions.pressure_kPa
        if pressure_kpa is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "ThermoFormer isobaric_vle requires pressure_kPa.",
                "Provide pressure_kPa in the task conditions.",
            )

        n_components = len(request.components)
        compositions = _build_composition_sweep(n_components, request.points)
        if not compositions:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "No valid compositions could be generated for the sweep.",
                "Check component count and point count.",
            )

        points = self._build_equilibrium_points(
            request, compositions,
            temperature_k=None, pressure_kpa=pressure_kpa,
        )

        if not points:
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                "ThermoFormer isobaric_vle produced no valid points.",
                "Review the pressure and component identities.",
            )

        temperatures = [p.temperature_K for p in points]
        return build_result(
            request, self.model_name, self.version,
            T=float(np.mean(temperatures)), P=pressure_kpa,
            residual=max(p.equilibrium_residual for p in points),
            iters=0, points=points,
            warnings=self._warnings(),
            solver_name=self.solver_name,
            phase_state="curve",
        )

    def infinite_dilution_activity(
        self, request: TaskManifest
    ) -> CalculationResult:
        """Predict gamma∞ at infinite dilution from molecular structure.

        For each solute/solvent pair, evaluates gamma at x_solute → 0.
        """
        _check_system_scope(request.components, request.conditions.pressure_kPa)
        smiles_list = _smiles_from_components(request.components)
        n_components = len(smiles_list)
        predictor = self._get_predictor()
        molecules = predictor.encode_molecules(smiles_list)

        span = request.conditions.temperature_span_K
        single_temperature = request.conditions.temperature_K
        if span is not None:
            temperatures = np.linspace(
                span[0], span[1], request.points
            ).tolist()
        elif single_temperature is not None:
            temperatures = [single_temperature]
        else:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "ThermoFormer gamma-infinity prediction requires "
                "a temperature.",
                "Provide temperature_K or temperature_span_K "
                "in the task conditions.",
            )

        gamma_infinity_points: list[GammaInfinityPoint] = []
        for solute_index in range(n_components):
            for solvent_index in range(n_components):
                if solute_index == solvent_index:
                    continue
                for temp in temperatures:
                    # x ≈ [0, 1] for binary with solute at trace
                    x = [
                        1e-6 if i == solute_index
                        else 1.0 - 1e-6 if i == solvent_index
                        else 0.0
                        for i in range(n_components)
                    ]
                    mask = [True] * n_components
                    result = predictor.predict_bubble_isothermal(
                        molecules, temp, x, mask
                    )
                    gamma_inf = result["gamma"][solute_index]
                    if not np.isfinite(gamma_inf) or gamma_inf <= 0:
                        continue
                    gamma_infinity_points.append(
                        GammaInfinityPoint(
                            temperature_K=float(temp),
                            solute_index=solute_index,
                            solvent_index=solvent_index,
                            gamma_infinity=float(gamma_inf),
                            ln_gamma_infinity=float(np.log(gamma_inf)),
                        )
                    )

        warnings = self._warnings()
        warnings.append(
            "ThermoFormer gamma-infinity is derived from x→0 extrapolation "
            "of the full-concentration gamma prediction; "
            "review numerical stability."
        )

        return CalculationResult(
            task_id=request.task_id,
            calculation_type="infinite_dilution_activity",
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            gamma_infinity=gamma_infinity_points,
            temperature_K=(
                float(temperatures[0]) if len(temperatures) == 1 else None
            ),
            converged=True,
            residual=0.0,
            iterations=0,
            warnings=warnings,
            backend_version=self.version,
            solver_name=self.solver_name,
            phase_state="unknown",
        )

    # ── protocol stubs that must fail structurally ────────────────────────

    def dew_point(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("dew_point")

    def tp_flash(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("tp_flash")

    def phase_stability(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("phase_stability")

    def azeotrope(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("azeotrope")

    def lle(self, request: TaskManifest) -> CalculationResult:
        raise self._unsupported("lle")

    def _warnings(self) -> list[str]:
        return [
            "ThermoFormer is a predictive ML backend; results are approximate "
            "and have not been validated against experimental data for this "
            "system. Benchmark closure and applicability review are pending.",
        ]
