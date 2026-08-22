"""Canonical domain schemas for calculations, validation, routing, and chat."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CalculationType: TypeAlias = Literal[
    "bubble_point",
    "dew_point",
    "isobaric_vle",
    "isothermal_vle",
    "tp_flash",
    "phase_stability",
    "azeotrope",
    "lle",
    "infinite_dilution_activity",
]
RunStatus: TypeAlias = Literal["passed", "warning", "failed"]

_CALCULATION_TYPE_ALIASES: dict[str, CalculationType] = {
    "bubble": "bubble_point",
    "bubblepoint": "bubble_point",
    "bubble_point": "bubble_point",
    "dew": "dew_point",
    "dewpoint": "dew_point",
    "dew_point": "dew_point",
    "isobaric_vle": "isobaric_vle",
    "isobaricvle": "isobaric_vle",
    "t_x_y": "isobaric_vle",
    "txy": "isobaric_vle",
    "isothermal_vle": "isothermal_vle",
    "isothermalvle": "isothermal_vle",
    "p_x_y": "isothermal_vle",
    "pxy": "isothermal_vle",
    "flash": "tp_flash",
    "tp_flash": "tp_flash",
    "tpflash": "tp_flash",
    "phase_stability": "phase_stability",
    "phasestability": "phase_stability",
    "stability": "phase_stability",
    "azeotrope": "azeotrope",
    "azeotrope_search": "azeotrope",
    "liquid_liquid_equilibrium": "lle",
    "lle": "lle",
    "infinite_dilution_activity": "infinite_dilution_activity",
    "infinite_dilution": "infinite_dilution_activity",
    "gamma_infinity": "infinite_dilution_activity",
    "gamma_inf": "infinite_dilution_activity",
    "activity_coefficient_at_infinite_dilution": "infinite_dilution_activity",
}


class Intent(StrEnum):
    CONCEPT_QA = "CONCEPT_QA"
    MODEL_SELECTION_QA = "MODEL_SELECTION_QA"
    PARAMETER_QUERY = "PARAMETER_QUERY"
    DATA_QUERY = "DATA_QUERY"
    EQUILIBRIUM_CALCULATION = "EQUILIBRIUM_CALCULATION"
    RESULT_INTERPRETATION = "RESULT_INTERPRETATION"
    SENSITIVITY_ANALYSIS = "SENSITIVITY_ANALYSIS"
    PROCESS_RECOMMENDATION = "PROCESS_RECOMMENDATION"
    FLOW_DESIGN_QA = "FLOW_DESIGN_QA"
    TASK_CORRECTION = "TASK_CORRECTION"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"


class FailureType(StrEnum):
    SEMANTIC_FAILURE = "semantic_failure"
    MISSING_DATA = "missing_data"
    MISSING_PARAMETERS = "missing_parameters"
    UNSUPPORTED_MODEL = "unsupported_model"
    NUMERICAL_NONCONVERGENCE = "numerical_nonconvergence"
    PHYSICAL_VALIDATION_FAILURE = "physical_validation_failure"
    PHASE_INSTABILITY = "phase_instability"
    PARAMETER_OUT_OF_DOMAIN = "parameter_out_of_domain"
    MODEL_CONFLICT = "model_conflict"


class ComponentIdentity(BaseModel):
    component_id: str
    name: str
    cas_number: str | None = None
    smiles: str | None = None
    aliases: list[str] = Field(default_factory=list)


class ThermodynamicConditions(BaseModel):
    temperature_K: float | None = Field(default=None, gt=0)
    pressure_kPa: float | None = Field(default=None, gt=0)
    liquid_composition: list[float] | None = None
    vapor_composition: list[float] | None = None
    feed_composition: list[float] | None = None
    #: Optional low/high temperature bounds for curve-style calculations such as
    #: PGSSI gamma-infinity(T).  When set together with ``points`` the backend
    #: sweeps a curve; when absent, ``temperature_K`` is a single point.
    temperature_span_K: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_compositions(self) -> ThermodynamicConditions:
        for label in ("liquid_composition", "vapor_composition", "feed_composition"):
            values = getattr(self, label)
            if values is None:
                continue
            if not values or any(value < 0 or value > 1 for value in values):
                raise ValueError(f"{label} must contain mole fractions in [0, 1]")
            if abs(sum(values) - 1.0) > 1e-8:
                raise ValueError(f"{label} must sum to one within 1e-8")
        if self.temperature_span_K is not None:
            lower, upper = self.temperature_span_K
            if lower <= 0 or upper <= lower:
                raise ValueError("temperature_span_K must contain positive ascending bounds")
        return self


class TaskManifest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    equilibrium_type: Literal["VLE", "LLE", "FLASH"]
    calculation_type: CalculationType
    components: list[ComponentIdentity] = Field(min_length=1)
    conditions: ThermodynamicConditions
    composition_basis: Literal["mole_fraction", "mass_fraction"] = "mole_fraction"
    requested_outputs: list[str] = Field(default_factory=lambda: ["table", "validation"])
    validation_requirements: list[str] = Field(
        default_factory=lambda: ["composition_balance", "equilibrium_residual", "convergence"]
    )
    assumptions: list[str] = Field(default_factory=list)
    model_name: str | None = None
    points: int = Field(default=21, ge=2, le=501)
    original_question: str | None = None
    parameters: list[ParameterSet] = Field(default_factory=list)

    @field_validator("calculation_type", mode="before")
    @classmethod
    def normalize_calculation_type(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
        return _CALCULATION_TYPE_ALIASES.get(normalized, normalized)


class SystemProfile(BaseModel):
    component_count: int
    is_electrolyte: bool
    all_hydrocarbons: bool
    is_polar: bool
    association_risk: bool
    pressure_regime: Literal["low", "moderate", "high", "unknown"]
    phase_split_risk: Literal["low", "medium", "high", "unknown"]
    supported: bool
    evidence: list[str] = Field(default_factory=list)


class ModelCard(BaseModel):
    model_name: str
    family: str
    supported_tasks: list[str]
    excluded_systems: list[str]
    requires_binary_parameters: bool
    pressure_regime: list[str]
    validation_requirements: list[str]
    implementation_status: Literal["available", "contract_only", "planned"]
    production_ready: bool


class ParameterSet(BaseModel):
    parameter_set_id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str = Field(min_length=1, max_length=128)
    component_order: list[str] = Field(min_length=2)
    parameters: dict[str, float] = Field(min_length=1)
    parameter_form: str = Field(min_length=1, max_length=256)
    units: dict[str, str] = Field(min_length=1)
    temperature_range_K: tuple[float, float] | None = None
    pressure_range_kPa: tuple[float, float] | None = None
    equilibrium_types: list[Literal["VLE", "LLE", "FLASH"]] = Field(min_length=1)
    source_title: str | None = None
    source_identifier: str | None = None
    source_type: Literal["literature", "database", "user_supplied", "test_fixture", "estimated", "unknown"]
    quality_level: str = Field(min_length=1, max_length=64)
    notes: str | None = None

    @field_validator("model_name", "parameter_form", "quality_level")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("component_order")
    @classmethod
    def validate_component_order(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("component_order entries must not be blank")
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("component_order entries must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_parameter_evidence(self) -> ParameterSet:
        if any(not name.strip() for name in self.parameters):
            raise ValueError("parameter names must not be blank")
        if any(not isfinite(value) for value in self.parameters.values()):
            raise ValueError("parameter values must be finite")
        if set(self.parameters) != set(self.units):
            raise ValueError("units must contain exactly one entry for every parameter")
        if any(not unit.strip() for unit in self.units.values()):
            raise ValueError("parameter units must not be blank")

        for label, bounds in (
            ("temperature_range_K", self.temperature_range_K),
            ("pressure_range_kPa", self.pressure_range_kPa),
        ):
            if bounds is None:
                continue
            lower, upper = bounds
            if not isfinite(lower) or not isfinite(upper) or lower <= 0 or upper <= lower:
                raise ValueError(f"{label} must contain finite positive bounds in ascending order")

        if self.source_type in {"literature", "database"}:
            if self.source_title is None or not self.source_title.strip():
                raise ValueError(f"{self.source_type} parameter sets require source_title")
            if self.source_identifier is None or not self.source_identifier.strip():
                raise ValueError(f"{self.source_type} parameter sets require source_identifier")
        return self


class FailureDetail(BaseModel):
    failure_type: FailureType
    message: str
    recovery_action: str
    details: dict[str, Any] = Field(default_factory=dict)


class EquilibriumPoint(BaseModel):
    temperature_K: float
    pressure_kPa: float
    liquid_composition: list[float]
    vapor_composition: list[float]
    equilibrium_residual: float


class GammaInfinityPoint(BaseModel):
    """One infinite-dilution activity coefficient datum at a temperature.

    ``solute_index``/``solvent_index`` refer to the task component order; the
    coefficient is for the solute at infinite dilution in the solvent.
    """

    temperature_K: float
    solute_index: int = Field(ge=0)
    solvent_index: int = Field(ge=0)
    gamma_infinity: float = Field(gt=0)
    ln_gamma_infinity: float


class PhaseResult(BaseModel):
    phase: Literal["liquid", "vapor"]
    fraction: float = Field(ge=0, le=1)
    composition: list[float]


class CalculationResult(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    calculation_type: CalculationType
    input_snapshot: dict[str, Any]
    model_name: str
    parameter_set_id: str | None = None
    points: list[EquilibriumPoint] = Field(default_factory=list)
    gamma_infinity: list[GammaInfinityPoint] = Field(default_factory=list)
    phases: list[PhaseResult] = Field(default_factory=list)
    temperature_K: float | None = None
    pressure_kPa: float | None = None
    vapor_fraction: float | None = None
    phase_state: Literal["liquid", "vapor", "two_phase", "curve", "unknown"] = "unknown"
    converged: bool
    residual: float
    iterations: int
    warnings: list[str] = Field(default_factory=list)
    backend_version: str
    solver_name: str = "unspecified"
    failure: FailureDetail | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CheckResult(BaseModel):
    passed: bool
    metric: float | None = None
    tolerance: float | None = None
    message: str


class ValidationReport(BaseModel):
    overall_status: RunStatus
    composition_balance: CheckResult
    material_balance: CheckResult
    equilibrium_residual: CheckResult
    convergence: CheckResult
    parameter_applicability: CheckResult
    phase_stability: CheckResult | None = None
    warnings: list[str] = Field(default_factory=list)
    recommended_action: str | None = None
    maximum_equilibrium_residual: float
    mean_equilibrium_residual: float
    solver_converged: bool


class CalculationEnvelope(BaseModel):
    result: CalculationResult
    validation: ValidationReport
    parameter_sources: list[dict[str, str]] = Field(default_factory=list)
    model_recommendations: list[ModelRecommendation] = Field(default_factory=list)


class ModelComparisonEntry(BaseModel):
    model_name: str
    score: float
    executable: bool
    result: CalculationResult | None = None
    validation: ValidationReport | None = None
    failure: FailureDetail | None = None
    parameter_sources: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ModelComparisonResponse(BaseModel):
    task: TaskManifest
    entries: list[ModelComparisonEntry]
    executed_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    summary: str


class ScoreBreakdown(BaseModel):
    phase_support_score: float
    system_match_score: float
    condition_match_score: float
    parameter_availability_score: float
    evidence_quality_score: float
    extrapolation_penalty: float
    numerical_risk_penalty: float

    @property
    def total(self) -> float:
        return (
            self.phase_support_score
            + self.system_match_score
            + self.condition_match_score
            + self.parameter_availability_score
            + self.evidence_quality_score
            - self.extrapolation_penalty
            - self.numerical_risk_penalty
        )


class ModelRecommendation(BaseModel):
    model_name: str
    score: float
    executable: bool
    reasons: list[str]
    exclusions: list[str] = Field(default_factory=list)
    breakdown: ScoreBreakdown


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class EvidenceStatement(BaseModel):
    category: Literal["Knowledge", "Database", "Calculation", "Inference", "Estimate", "Warning"]
    text: str


class AgentStep(BaseModel):
    phase: Literal["plan", "execute", "validate", "respond"]
    status: Literal["completed", "failed", "blocked"]
    summary: str
    tool_name: str | None = None


FlowParameterSource: TypeAlias = Literal["LLM_SUGGESTED", "RULE_SUGGESTED", "VALIDATED"]

_ALLOWED_UNIT_OPERATION_TYPES = frozenset(
    {
        "preheater",
        "distillation_column",
        "flash_drum",
        "condenser",
        "reboiler",
        "heat_exchanger",
        "mixer",
        "splitter",
    }
)
UnitOperationType: TypeAlias = Literal[
    "preheater",
    "distillation_column",
    "flash_drum",
    "condenser",
    "reboiler",
    "heat_exchanger",
    "mixer",
    "splitter",
]


class FlowParameter(BaseModel):
    """A single parameter value attached to a feed or unit-operation slot.

    Every value is explicitly labeled so consumers can distinguish validated
    thermodynamic numbers from LLM/rule-based placeholders.
    """

    value: float | int | str
    source: FlowParameterSource
    needs_validation: bool = True
    note: str | None = None


class FlowFeed(BaseModel):
    """Description of a single process feed stream."""

    components: list[str] = Field(min_length=1)
    composition_mole: list[float] = Field(default_factory=list)
    temperature_K: float | None = Field(default=None, gt=0)
    pressure_kPa: float | None = Field(default=None, gt=0)
    flow_rate_mol_s: float | None = Field(default=None, gt=0)
    assumption: str | None = None

    @model_validator(mode="after")
    def validate_composition(self) -> FlowFeed:
        if not self.composition_mole:
            return self
        if len(self.composition_mole) != len(self.components):
            raise ValueError("composition_mole must match components length")
        if any(value < 0 or value > 1 for value in self.composition_mole):
            raise ValueError("composition_mole mole fractions must be in [0, 1]")
        if abs(sum(self.composition_mole) - 1.0) > 1e-6:
            raise ValueError("composition_mole must sum to 1.0 (within 1e-6)")
        return self


class FlowUnitOperation(BaseModel):
    """One unit operation in the designed flowsheet sequence.

    The type vocabulary is intentionally narrow (white-listed) so downstream
    exporters (DWSIM etc.) can map each node deterministically.
    """

    id: str = Field(min_length=1, max_length=64)
    type: UnitOperationType
    name: str = Field(min_length=1, max_length=256)
    input_stream: str | None = None
    output_streams: dict[str, str] = Field(default_factory=dict)
    conditions: dict[str, FlowParameter] = Field(default_factory=dict)

    @field_validator("id", "name", "input_stream")
    @classmethod
    def strip_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class FlowProductSpec(BaseModel):
    """Named product purity or recovery requirement on a stream."""

    stream: str = Field(min_length=1)
    spec: str = Field(min_length=1)


class FlowDesignDraft(BaseModel):
    """Structured process-flow design draft produced by a skill.

    The draft is always a non-final recommendation: every parameter that can
    be verified against a deterministic thermodynamics engine carries an
    explicit ``needs_validation`` label. Backends (DWSIM export, column
    design) consume this schema instead of ad-hoc dictionaries.
    """

    flow_name: str = Field(min_length=1, max_length=512)
    flow_type: str = Field(default="custom", max_length=128)
    feed: FlowFeed
    unit_operations: list[FlowUnitOperation] = Field(min_length=1)
    streams_connectivity_note: str | None = None
    thermodynamic_model: str | None = Field(default=None, max_length=128)
    model_recommendation_note: str | None = None
    product_specs: list[FlowProductSpec] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(use_attribute_docstrings=False)

    @field_validator("unit_operations")
    @classmethod
    def validate_unique_ids(cls, units: list[FlowUnitOperation]) -> list[FlowUnitOperation]:
        seen_ids: set[str] = set()
        for unit in units:
            if unit.type not in _ALLOWED_UNIT_OPERATION_TYPES:
                raise ValueError(
                    f"unit_operations type {unit.type!r} is not allowed; "
                    f"use one of {sorted(_ALLOWED_UNIT_OPERATION_TYPES)}"
                )
            if unit.id in seen_ids:
                raise ValueError(f"unit_operations contains duplicate id {unit.id!r}")
            seen_ids.add(unit.id)
        return units


class ChatResponse(BaseModel):
    conversation_id: str
    intent: Intent
    answer: str
    statements: list[EvidenceStatement]
    execution_steps: list[AgentStep] = Field(default_factory=list)
    task: TaskManifest | None = None
    calculation: CalculationEnvelope | None = None
    #: Structured process-flow design payload when ``intent`` is
    #: :py:attr:`Intent.FLOW_DESIGN_QA`. Other modules (DWSIM export,
    #: downstream flowsheet tooling) should read this field instead of
    #: re-parsing the natural-language ``answer``.
    flow_design: FlowDesignDraft | None = None
    request_id: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class RunRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    request_id: str
    task_id: str
    status: RunStatus
    input_snapshot: dict[str, Any]
    result: dict[str, Any]
    validation: dict[str, Any]
    created_at: datetime


class RunSummary(BaseModel):
    run_id: str
    request_id: str
    task_id: str
    status: RunStatus
    calculation_type: CalculationType
    model_name: str
    backend_version: str
    created_at: datetime


class RunListResponse(BaseModel):
    items: list[RunSummary]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


TaskManifest.model_rebuild()
CalculationEnvelope.model_rebuild()
