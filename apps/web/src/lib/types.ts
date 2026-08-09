export type Intent =
  | "CONCEPT_QA"
  | "MODEL_SELECTION_QA"
  | "PARAMETER_QUERY"
  | "DATA_QUERY"
  | "EQUILIBRIUM_CALCULATION"
  | "RESULT_INTERPRETATION"
  | "SENSITIVITY_ANALYSIS"
  | "PROCESS_RECOMMENDATION"
  | "TASK_CORRECTION"
  | "UNSUPPORTED_TASK";

export type CalculationType =
  | "bubble_point"
  | "dew_point"
  | "isobaric_vle"
  | "isothermal_vle"
  | "tp_flash"
  | "phase_stability"
  | "azeotrope"
  | "lle";

export interface ComponentIdentity {
  component_id: string;
  name: string;
  cas_number?: string | null;
  aliases: string[];
}

export interface Conditions {
  temperature_K?: number | null;
  pressure_kPa?: number | null;
  liquid_composition?: number[] | null;
  vapor_composition?: number[] | null;
  feed_composition?: number[] | null;
}

export interface ParameterSet {
  parameter_set_id: string;
  model_name: string;
  component_order: string[];
  parameters: Record<string, number>;
  parameter_form: string;
  units: Record<string, string>;
  temperature_range_K?: [number, number] | null;
  pressure_range_kPa?: [number, number] | null;
  equilibrium_types: Array<"VLE" | "LLE" | "FLASH">;
  source_title?: string | null;
  source_identifier?: string | null;
  source_type: "literature" | "database" | "user_supplied" | "test_fixture" | "estimated" | "unknown";
  quality_level: string;
  notes?: string | null;
}

export interface TaskManifest {
  task_id: string;
  equilibrium_type: "VLE" | "LLE" | "FLASH";
  calculation_type: CalculationType;
  components: ComponentIdentity[];
  conditions: Conditions;
  composition_basis: "mole_fraction" | "mass_fraction";
  requested_outputs: string[];
  validation_requirements: string[];
  assumptions: string[];
  model_name?: string | null;
  points: number;
  original_question?: string | null;
  parameters: ParameterSet[];
}

export interface EquilibriumPoint {
  temperature_K: number;
  pressure_kPa: number;
  liquid_composition: number[];
  vapor_composition: number[];
  equilibrium_residual: number;
}

export interface CheckResult {
  passed: boolean;
  metric?: number | null;
  tolerance?: number | null;
  message: string;
}

export interface ValidationReport {
  overall_status: "passed" | "warning" | "failed";
  composition_balance: CheckResult;
  material_balance: CheckResult;
  equilibrium_residual: CheckResult;
  convergence: CheckResult;
  parameter_applicability: CheckResult;
  phase_stability?: CheckResult | null;
  warnings: string[];
  recommended_action?: string | null;
  maximum_equilibrium_residual: number;
  mean_equilibrium_residual: number;
  solver_converged: boolean;
}

export interface ModelRecommendation {
  model_name: string;
  score: number;
  executable: boolean;
  reasons: string[];
  exclusions: string[];
  breakdown: ScoreBreakdown;
}

export interface ScoreBreakdown {
  phase_support_score: number;
  system_match_score: number;
  condition_match_score: number;
  parameter_availability_score: number;
  evidence_quality_score: number;
  extrapolation_penalty: number;
  numerical_risk_penalty: number;
}

export interface ModelCard {
  model_name: string;
  family: string;
  supported_tasks: string[];
  excluded_systems: string[];
  requires_binary_parameters: boolean;
  pressure_regime: string[];
  validation_requirements: string[];
  implementation_status: "available" | "contract_only" | "planned";
  production_ready: boolean;
}

export interface PhaseResult {
  phase: "liquid" | "vapor";
  fraction: number;
  composition: number[];
}

export interface CalculationEnvelope {
  result: {
    run_id: string;
    task_id: string;
    calculation_type: CalculationType;
    input_snapshot: Record<string, unknown>;
    model_name: string;
    parameter_set_id?: string | null;
    points: EquilibriumPoint[];
    phases: PhaseResult[];
    temperature_K?: number | null;
    pressure_kPa?: number | null;
    vapor_fraction?: number | null;
    phase_state: string;
    converged: boolean;
    residual: number;
    iterations: number;
    warnings: string[];
    backend_version: string;
    solver_name: string;
    failure?: Record<string, unknown> | null;
    created_at: string;
  };
  validation: ValidationReport;
  parameter_sources: Array<Record<string, string>>;
  model_recommendations: ModelRecommendation[];
}

export interface EvidenceStatement {
  category: "Knowledge" | "Database" | "Calculation" | "Inference" | "Estimate" | "Warning";
  text: string;
}

export interface AgentStep {
  phase: "plan" | "execute" | "validate" | "respond";
  status: "completed" | "failed" | "blocked";
  summary: string;
  tool_name?: string | null;
}

export interface ChatResponse {
  conversation_id: string;
  intent: Intent;
  answer: string;
  statements: EvidenceStatement[];
  execution_steps: AgentStep[];
  task?: TaskManifest | null;
  calculation?: CalculationEnvelope | null;
  request_id?: string | null;
}

export type RunStatus = "passed" | "warning" | "failed";

export interface RunSummary {
  run_id: string;
  request_id: string;
  task_id: string;
  status: RunStatus;
  calculation_type: CalculationType;
  model_name: string;
  backend_version: string;
  created_at: string;
}

export interface RunListResponse {
  items: RunSummary[];
  total: number;
  limit: number;
  offset: number;
}
