"""Database setup and repositories."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import (
    Base,
    CalculationPointRow,
    CalculationRunRow,
    ConversationRow,
    EvidenceRecordRow,
    ExportRecordRow,
    MessageRow,
    ParameterSetRow,
    TaskRow,
    ValidationReportRow,
)
from schemas.domain import CalculationEnvelope, ParameterSet, RunRecord, RunStatus, RunSummary


def _default_database_url() -> str:
    local_app_data = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    database_path = local_app_data / "ThermoEqui-Agent" / "thermoequi.db"
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        database_path = Path(gettempdir()) / "ThermoEqui-Agent" / "thermoequi.db"
        database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path.as_posix()}"


def create_database_engine(url: str | None = None) -> Engine:
    database_url: str = url if url is not None else (os.environ.get("DATABASE_URL") or _default_database_url())
    arguments = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=arguments)


load_dotenv()
engine = create_database_engine()


class ParameterSetConflictError(ValueError):
    """Raised when an immutable parameter-set identifier already exists."""


def initialize_database(target: Engine = engine) -> None:
    Base.metadata.create_all(target)


def session_scope(target: Engine = engine) -> Iterator[Session]:
    with Session(target) as session:
        yield session


class Repository:
    def __init__(self, target: Engine = engine) -> None:
        self.engine = target

    @staticmethod
    def _upsert_task(
        session: Session,
        *,
        task_id: str,
        manifest: dict[str, object],
        conversation_id: str | None = None,
    ) -> None:
        task_row = session.get(TaskRow, task_id)
        if task_row is None:
            session.add(
                TaskRow(
                    id=task_id,
                    conversation_id=conversation_id,
                    manifest=manifest,
                )
            )
            return
        task_row.manifest = manifest
        if conversation_id is not None:
            task_row.conversation_id = conversation_id

    def _add_chat_rows(
        self,
        session: Session,
        conversation_id: str,
        user_message: str,
        response_payload: dict[str, object],
    ) -> None:
        conversation = session.get(ConversationRow, conversation_id)
        if conversation is None:
            session.add(ConversationRow(id=conversation_id))
        session.add(
            MessageRow(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role="user",
                content=user_message,
            )
        )
        session.add(
            MessageRow(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content=str(response_payload.get("answer", "")),
                structured=response_payload,
            )
        )
        task = response_payload.get("task")
        if isinstance(task, dict):
            manifest = {str(key): value for key, value in task.items()}
            task_id = manifest.get("task_id")
            if isinstance(task_id, str):
                self._upsert_task(
                    session,
                    task_id=task_id,
                    manifest=manifest,
                    conversation_id=conversation_id,
                )

    def _add_run_rows(self, session: Session, envelope: CalculationEnvelope, request_id: str) -> None:
        result = envelope.result
        validation = envelope.validation
        self._upsert_task(
            session,
            task_id=result.task_id,
            manifest=dict(result.input_snapshot),
        )
        session.add(
            CalculationRunRow(
                id=result.run_id,
                request_id=request_id,
                task_id=result.task_id,
                status=validation.overall_status,
                input_snapshot=result.input_snapshot,
                result_snapshot=result.model_dump(mode="json"),
                validation_snapshot=validation.model_dump(mode="json"),
                software_version=result.backend_version,
            )
        )
        for ordinal, point in enumerate(result.points):
            session.add(
                CalculationPointRow(
                    run_id=result.run_id,
                    ordinal=ordinal,
                    temperature_K=point.temperature_K,
                    pressure_kPa=point.pressure_kPa,
                    liquid_composition=point.liquid_composition,
                    vapor_composition=point.vapor_composition,
                )
            )
        session.add(
            ValidationReportRow(
                run_id=result.run_id,
                overall_status=validation.overall_status,
                report=validation.model_dump(mode="json"),
            )
        )
        for source in envelope.parameter_sources:
            session.add(
                EvidenceRecordRow(
                    run_id=result.run_id,
                    category="Database",
                    source_identifier=source.get("source_identifier"),
                    payload=source,
                )
            )
        for recommendation in envelope.model_recommendations:
            session.add(
                EvidenceRecordRow(
                    run_id=result.run_id,
                    category="Inference",
                    source_identifier=None,
                    payload=recommendation.model_dump(mode="json"),
                )
            )

    def save_chat(self, conversation_id: str, user_message: str, response_payload: dict[str, object]) -> None:
        with Session(self.engine) as session:
            self._add_chat_rows(session, conversation_id, user_message, response_payload)
            session.commit()

    def save_run(self, envelope: CalculationEnvelope, request_id: str) -> None:
        with Session(self.engine) as session:
            self._add_run_rows(session, envelope, request_id)
            session.commit()

    def save_chat_and_run(
        self,
        conversation_id: str,
        user_message: str,
        response_payload: dict[str, object],
        envelope: CalculationEnvelope | None,
        request_id: str,
    ) -> None:
        with Session(self.engine) as session:
            self._add_chat_rows(session, conversation_id, user_message, response_payload)
            if envelope is not None:
                self._add_run_rows(session, envelope, request_id)
            session.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        with Session(self.engine) as session:
            row = session.get(CalculationRunRow, run_id)
            if row is None:
                return None
            return RunRecord(
                run_id=row.id,
                request_id=row.request_id,
                task_id=row.task_id,
                status=row.status,
                input_snapshot=row.input_snapshot,
                result=row.result_snapshot,
                validation=row.validation_snapshot,
                created_at=row.created_at,
            )

    def list_runs(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: RunStatus | None = None,
    ) -> tuple[list[RunSummary], int]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if status is not None and status not in {"passed", "warning", "failed"}:
            raise ValueError("status must be passed, warning, or failed")

        with Session(self.engine) as session:
            query = select(CalculationRunRow)
            count_query = select(func.count(CalculationRunRow.id))
            if status is not None:
                query = query.where(CalculationRunRow.status == status)
                count_query = count_query.where(CalculationRunRow.status == status)
            total = session.scalar(count_query) or 0
            rows = session.scalars(
                query.order_by(CalculationRunRow.created_at.desc(), CalculationRunRow.id.desc())
                .offset(offset)
                .limit(limit)
            ).all()

        summaries = [
            RunSummary.model_validate(
                {
                    "run_id": row.id,
                    "request_id": row.request_id,
                    "task_id": row.task_id,
                    "status": row.status,
                    "calculation_type": row.result_snapshot.get("calculation_type"),
                    "model_name": row.result_snapshot.get("model_name"),
                    "backend_version": row.software_version,
                    "created_at": row.created_at,
                }
            )
            for row in rows
        ]
        return summaries, total

    def add_parameter_set(self, parameter_set: ParameterSet) -> None:
        if parameter_set.source_type == "test_fixture":
            raise ValueError("test_fixture parameter sets cannot enter the production database")
        row = ParameterSetRow(
            id=parameter_set.parameter_set_id,
            model_name=parameter_set.model_name,
            component_key="|".join(parameter_set.component_order),
            payload=parameter_set.model_dump(mode="json"),
            source_type=parameter_set.source_type,
        )
        with Session(self.engine) as session:
            try:
                session.add(row)
                session.commit()
            except IntegrityError:
                session.rollback()
                raise ParameterSetConflictError(
                    f"Parameter set {parameter_set.parameter_set_id} already exists"
                ) from None

    def upsert_parameter_set(self, parameter_set: ParameterSet) -> str:
        """Insert or refresh one production parameter set; returns added/updated/unchanged."""
        if parameter_set.source_type == "test_fixture":
            raise ValueError("test_fixture parameter sets cannot enter the production database")
        payload = parameter_set.model_dump(mode="json")
        with Session(self.engine) as session:
            row = session.get(ParameterSetRow, parameter_set.parameter_set_id)
            if row is None:
                session.add(
                    ParameterSetRow(
                        id=parameter_set.parameter_set_id,
                        model_name=parameter_set.model_name,
                        component_key="|".join(parameter_set.component_order),
                        payload=payload,
                        source_type=parameter_set.source_type,
                    )
                )
                session.commit()
                return "added"
            if row.payload == payload:
                return "unchanged"
            row.model_name = parameter_set.model_name
            row.component_key = "|".join(parameter_set.component_order)
            row.payload = payload
            row.source_type = parameter_set.source_type
            session.commit()
            return "updated"

    def delete_duplicate_parameter_sets(self, production_sets: list[ParameterSet]) -> int:
        """Remove non-production rows that duplicate a production model/component pair."""
        removed = 0
        with Session(self.engine) as session:
            for parameter_set in production_sets:
                component_key = "|".join(parameter_set.component_order)
                rows = session.scalars(
                    select(ParameterSetRow).where(
                        ParameterSetRow.model_name == parameter_set.model_name,
                        ParameterSetRow.component_key == component_key,
                        ParameterSetRow.id != parameter_set.parameter_set_id,
                    )
                ).all()
                for row in rows:
                    session.delete(row)
                    removed += 1
            session.commit()
        return removed

    def search_parameter_sets(self, model_name: str | None, components: list[str]) -> list[ParameterSet]:
        with Session(self.engine) as session:
            query = select(ParameterSetRow)
            if model_name:
                query = query.where(ParameterSetRow.model_name == model_name)
            rows = session.scalars(query).all()
        requested = set(components)
        results: list[ParameterSet] = []
        for row in rows:
            parameter_set = ParameterSet.model_validate(row.payload)
            if not requested or set(parameter_set.component_order) == requested:
                results.append(parameter_set)
        return results

    def record_export(self, run_id: str, format_name: str) -> None:
        with Session(self.engine) as session:
            session.add(ExportRecordRow(run_id=run_id, format=format_name))
            session.commit()
