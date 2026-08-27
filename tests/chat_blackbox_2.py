"""Black-box routing checks for the /api/chat endpoint."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000/api/chat"

TEST_CASES = [
    {
        "name": "binary_isobaric_txy",
        "message": "计算苯-甲苯在101.325 kPa下的T-x-y相图，使用Ideal/Raoult模型",
        "calculation_type": "isobaric_vle",
    },
    {
        "name": "binary_isothermal_pxy",
        "message": "计算苯-甲苯体系在350 K下的P-x-y相图，使用Ideal/Raoult模型",
        "calculation_type": "isothermal_vle",
    },
    {
        "name": "bubble_point",
        "message": "计算苯-甲苯混合物泡点，压力101.325 kPa，液相组成苯0.5、甲苯0.5",
        "calculation_type": "bubble_point",
    },
    {
        "name": "dew_point",
        "message": "计算苯-甲苯混合物露点，压力101.325 kPa，气相组成苯0.5、甲苯0.5",
        "calculation_type": "dew_point",
    },
    {
        "name": "tp_flash",
        "message": (
            "使用Peng-Robinson模型计算甲烷、乙烷、氮气混合物TP Flash，"
            "温度110 K，压力100 kPa，摩尔组成为0.965、0.018、0.017"
        ),
        "calculation_type": "tp_flash",
    },
    {
        "name": "azeotrope_candidate_search",
        "message": "搜索乙醇-水体系在101.325 kPa下的共沸候选点",
        "calculation_type": "azeotrope",
    },
    {
        "name": "phase_classification",
        "message": "判断甲烷-乙烷混合物在300 K、5000 kPa条件下的相态",
        "calculation_type": "phase_stability",
    },
]

OUTPUT_FILE = Path(__file__).parent / "thermo_core_chat_test_results.json"


def send_request(message: str) -> dict[str, object]:
    try:
        response = requests.post(API_URL, json={"message": message}, timeout=120)
        try:
            body: object = response.json()
        except ValueError:
            body = response.text
        return {"status_code": response.status_code, "body": body}
    except requests.RequestException as error:
        return {"error": str(error)}


def route_matches(case: dict[str, str], response: dict[str, object]) -> bool:
    if response.get("status_code") != 200:
        return False
    body = response.get("body")
    if not isinstance(body, dict) or body.get("intent") != "EQUILIBRIUM_CALCULATION":
        return False
    task = body.get("task")
    return isinstance(task, dict) and task.get("calculation_type") == case["calculation_type"]


def main() -> int:
    results: list[dict[str, object]] = []
    print("=" * 70)
    print("Start ThermoEqui-Agent routing tests")
    print("=" * 70)

    for case in TEST_CASES:
        response = send_request(case["message"])
        passed = route_matches(case, response)
        results.append(
            {
                "case": case["name"],
                "time": datetime.now().isoformat(),
                "request": {"message": case["message"]},
                "expected_calculation_type": case["calculation_type"],
                "passed": passed,
                "response": response,
            }
        )
        print(f"{case['name']}: {'PASS' if passed else 'FAIL'}")

    OUTPUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = sum(not bool(result["passed"]) for result in results)
    print("=" * 70)
    print(f"Finished: {len(results) - failed} passed, {failed} failed")
    print(f"Result saved: {OUTPUT_FILE}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
