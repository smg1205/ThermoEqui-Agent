"""Persistence tests for immutable runs and fixture exclusion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agent.executor import execute_task
from database.models import CalculationRunRow, ConversationRow, MessageRow, TaskRow
from database.session import ParameterSetConflictError, Repository, create_database_engine, initialize_database
from schemas.domain import (
    CalculationEnvelope,
    ComponentIdentity,
    ParameterSet,
    RunStatus,
    TaskManifest,
    ThermodynamicConditions,
)


def run_envelope(status: RunStatus) -> CalculationEnvelope:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isobaric_vle",
        components=[
            ComponentIdentity(component_id="benzene", name="Benzene"),
            ComponentIdentity(component_id="toluene", name="Toluene"),
        ],
        conditions=ThermodynamicConditions(pressure_kPa=101.325),
        model_name="Ideal/Raoult",
        points=3,
    )
    envelope = execute_task(task)
    return envelope.model_copy(
        update={"validation": envelope.validation.model_copy(update={"overall_status": status})},
    )


def test_test_fixture_parameter_cannot_enter_production_repository() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)
    fixture = ParameterSet.model_validate_json(
        (Path(__file__).parent / "fixtures" / "synthetic_nrtl.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="cannot enter"):
        repository.add_parameter_set(fixture)


def test_default_database_is_created_under_local_app_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    engine = create_database_engine()

    assert Path(engine.url.database).resolve() == (tmp_path / "ThermoEqui-Agent" / "thermoequi.db").resolve()
    assert (tmp_path / "ThermoEqui-Agent").is_dir()


def test_duplicate_parameter_set_identifier_is_rejected_without_overwrite() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)
    parameter_set = ParameterSet.model_validate_json(
        (Path(__file__).parent / "fixtures" / "user_supplied_parameter.json").read_text(encoding="utf-8")
    )

    repository.add_parameter_set(parameter_set)
    with pytest.raises(ParameterSetConflictError, match="already exists"):
        repository.add_parameter_set(parameter_set)

    stored = repository.search_parameter_sets(parameter_set.model_name, parameter_set.component_order)
    assert stored == [parameter_set]


def test_run_history_is_paginated_sorted_and_filterable() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)
    statuses: list[RunStatus] = ["passed", "warning", "failed"]
    run_ids: list[str] = []
    base_time = datetime(2026, 7, 24, tzinfo=UTC)

    for index, status in enumerate(statuses):
        envelope = run_envelope(status)
        repository.save_run(envelope, f"request-{index}")
        run_ids.append(envelope.result.run_id)
        with Session(engine) as session:
            session.execute(
                update(CalculationRunRow)
                .where(CalculationRunRow.id == envelope.result.run_id)
                .values(created_at=base_time + timedelta(minutes=index))
            )
            session.commit()

    first_page, total = repository.list_runs(limit=2)
    second_page, second_total = repository.list_runs(limit=2, offset=2)
    failed_runs, failed_total = repository.list_runs(status="failed")

    assert total == second_total == 3
    assert [item.run_id for item in first_page] == list(reversed(run_ids))[0:2]
    assert [item.run_id for item in second_page] == [run_ids[0]]
    assert failed_total == 1
    assert [item.run_id for item in failed_runs] == [run_ids[2]]
    assert first_page[0].model_name == "Ideal/Raoult"
    assert first_page[0].calculation_type == "isobaric_vle"


def test_save_run_persists_the_task_manifest() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)
    envelope = run_envelope("passed")

    repository.save_run(envelope, "direct-calculation-request")

    with Session(engine) as session:
        task_row = session.get(TaskRow, envelope.result.task_id)
        assert task_row is not None
        assert task_row.conversation_id is None
        assert task_row.manifest == envelope.result.input_snapshot


def test_chat_and_run_persistence_rolls_back_as_one_transaction() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)
    envelope = run_envelope("passed")
    repository.save_run(envelope, "existing-run-request")
    payload: dict[str, object] = {
        "answer": "contract test response",
        "task": envelope.result.input_snapshot,
    }

    with pytest.raises(IntegrityError):
        repository.save_chat_and_run(
            "atomic-conversation",
            "contract test question",
            payload,
            envelope,
            "duplicate-run-request",
        )

    with Session(engine) as session:
        message_count = session.scalar(
            select(func.count()).select_from(MessageRow).where(MessageRow.conversation_id == "atomic-conversation")
        )
        task_row = session.get(TaskRow, envelope.result.task_id)
        assert session.get(ConversationRow, "atomic-conversation") is None
        assert message_count == 0
        assert task_row is not None
        assert task_row.conversation_id is None
