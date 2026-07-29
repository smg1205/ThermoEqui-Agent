"""Guard the hand-maintained TypeScript contract against backend field drift."""

from pathlib import Path

from schemas.domain import (
    CalculationResult,
    ModelRecommendation,
    RunListResponse,
    RunSummary,
    TaskManifest,
    ValidationReport,
)


def test_frontend_contract_declares_all_backend_fields() -> None:
    source = (Path(__file__).parents[1] / "apps" / "web" / "src" / "lib" / "types.ts").read_text(encoding="utf-8")
    for model in (TaskManifest, CalculationResult, ValidationReport, ModelRecommendation, RunSummary, RunListResponse):
        for field_name in model.model_fields:
            assert field_name in source, f"Frontend contract is missing {model.__name__}.{field_name}"
