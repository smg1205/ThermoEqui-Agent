"""Behavioral tests for the DWSIM phase-equilibrium flowsheet export."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pytest

from schemas.domain import RunRecord
from thermo_engine.dwsim_export import export_dwsim_flowsheet


class FakeObjectType(Enum):
    MaterialStream = "material-stream"
    Vessel = "vessel"


class FakeStream:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.GraphicObject = tag
        self.temperature: float | None = None
        self.pressure: float | None = None
        self.flow: float | None = None
        self.composition: list[float] | None = None

    def SetTemperature(self, value: float) -> None:
        self.temperature = value

    def SetPressure(self, value: float) -> None:
        self.pressure = value

    def SetMolarFlow(self, value: float) -> None:
        self.flow = value

    def SetOverallComposition(self, value: list[float]) -> None:
        self.composition = value


class FakeFlowsheet:
    def __init__(self) -> None:
        self.compounds: list[str] = []
        self.property_package: str | None = None
        self.objects: dict[str, FakeStream] = {}
        self.connections: list[tuple[str, str, int, int]] = []

    def AddCompound(self, name: str) -> None:
        self.compounds.append(name)

    def AddPropertyPackage(self, name: str) -> None:
        self.property_package = name

    def AddObject(self, _: FakeObjectType, _x: int, _y: int, tag: str) -> FakeStream:
        stream = FakeStream(tag)
        self.objects[tag] = stream
        return stream

    def ConnectObjects(self, source: str, target: str, source_port: int, target_port: int) -> None:
        self.connections.append((source, target, source_port, target_port))


class FakeAutomation:
    def __init__(self) -> None:
        self.flowsheet = FakeFlowsheet()

    def CreateFlowsheet(self) -> FakeFlowsheet:
        return self.flowsheet

    def SaveFlowsheet(self, _: FakeFlowsheet, destination: str) -> None:
        Path(destination).write_bytes(b"fake-dwsim-file")


def run_record() -> RunRecord:
    return RunRecord.model_validate(
        {
            "run_id": "dwsim-test-run",
            "request_id": "request-id",
            "task_id": "task-id",
            "status": "passed",
            "input_snapshot": {
                "components": [{"name": "Benzene"}, {"name": "Toluene"}],
                "conditions": {"pressure_kPa": 101.325, "liquid_composition": [0.4, 0.6]},
            },
            "result": {
                "model_name": "Ideal/Raoult",
                "points": [{"temperature_K": 365.0, "pressure_kPa": 101.325, "liquid_composition": [0.4, 0.6]}],
            },
            "validation": {},
            "created_at": "2026-08-10T00:00:00Z",
        }
    )


def test_export_creates_a_dwsim_phase_equilibrium_flowsheet(tmp_path: Path) -> None:
    automation = FakeAutomation()
    destination = export_dwsim_flowsheet(
        run_record(),
        tmp_path / "equilibrium.dwxmz",
        factory=lambda: automation,
        object_type=FakeObjectType,
    )

    assert destination.read_bytes() == b"fake-dwsim-file"
    assert automation.flowsheet.compounds == ["Benzene", "Toluene"]
    assert automation.flowsheet.property_package == "Raoult's Law"
    feed = automation.flowsheet.objects["Feed"]
    assert feed.temperature == 365.0
    assert feed.pressure == 101325.0
    assert feed.flow == 1.0
    assert feed.composition == [0.4, 0.6]
    assert automation.flowsheet.connections == [
        ("Feed", "Equilibrium Flash", 0, 0),
        ("Equilibrium Flash", "Vapor Product", 0, 0),
        ("Equilibrium Flash", "Liquid Product", 1, 0),
    ]


def test_export_rejects_a_non_dwsim_file_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".dwxmz"):
        export_dwsim_flowsheet(
            run_record(),
            tmp_path / "equilibrium.json",
            factory=FakeAutomation,
            object_type=FakeObjectType,
        )
