"""
Black-box test runner for /api/chat endpoint.

Test coverage:
- Binary isobaric T-x-y
- Binary isothermal P-x-y
- Bubble point
- Dew point
- TP Flash
- Azeotrope candidate search
- Phase classification
- Ideal/Raoult model
- Peng-Robinson model
- Model knowledge QA
- Unsupported model contract
- Missing parameter failure
- Unit normalization
- Out-of-scope rejection
- Export request
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import requests

# =========================
# API地址
# =========================

API_URL = "http://127.0.0.1:8000/api/chat"


# =========================
# 测试请求
# =========================

TEST_CASES = [
    # =====================================================
    # 二元等压 T-x-y
    # =====================================================
    {
        "name": "binary_isobaric_txy_benzene_toluene",
        "request": {"message": "计算苯-甲苯在101.325 kPa下的T-x-y相图，使用Ideal/Raoult模型"},
    },
    # =====================================================
    # 二元等温 P-x-y
    # =====================================================
    {
        "name": "binary_isothermal_pxy_benzene_toluene",
        "request": {"message": "计算苯-甲苯在350 K下的P-x-y相图"},
    },
    # =====================================================
    # 泡点
    # =====================================================
    {
        "name": "bubble_point_calculation",
        "request": {"message": "计算苯-甲苯混合物泡点，压力101.325 kPa，液相组成苯0.5、甲苯0.5"},
    },
    # =====================================================
    # 露点
    # =====================================================
    {
        "name": "dew_point_calculation",
        "request": {"message": "计算苯-甲苯混合物露点，压力101.325 kPa，气相组成苯0.5、甲苯0.5"},
    },
    # =====================================================
    # TP Flash
    # =====================================================
    {
        "name": "tp_flash_peng_robinson",
        "request": {
            "message": "使用Peng-Robinson计算甲烷、乙烷和氮气的TP Flash，"
            "温度110 K，压力100 kPa，"
            "摩尔组成为0.965、0.018、0.017"
        },
    },
    # =====================================================
    # 共沸候选搜索
    # =====================================================
    {
        "name": "azeotrope_candidate_search",
        "request": {"message": "搜索乙醇-水体系在101.325 kPa下的共沸候选点"},
    },
    # =====================================================
    # 基础相态分类
    # =====================================================
    {
        "name": "phase_classification",
        "request": {"message": "判断甲烷乙烷混合物在300 K、5000 kPa条件下的相态"},
    },
    # =====================================================
    # PR模型计算
    # =====================================================
    {
        "name": "peng_robinson_vle",
        "request": {"message": "使用Peng-Robinson模型计算甲烷-乙烷体系VLE"},
    },
    # =====================================================
    # 模型知识问答
    # =====================================================
    {
        "name": "thermo_model_question",
        "request": {"message": "Wilson、NRTL、UNIQUAC和Peng-Robinson模型有什么区别？"},
    },
    # =====================================================
    # 未实现模型契约
    # =====================================================
    {
        "name": "unsupported_activity_model",
        "request": {"message": "使用NRTL模型计算苯-甲苯T-x-y"},
    },
    # =====================================================
    # 参数缺失
    # =====================================================
    {
        "name": "missing_pressure_parameter",
        "request": {"message": "计算苯-甲苯泡点"},
    },
    {
        "name": "missing_composition_parameter",
        "request": {"message": "计算苯-甲苯在101.325 kPa下的泡点"},
    },
    # =====================================================
    # 单位规范化
    # =====================================================
    {
        "name": "pressure_unit_normalization",
        "request": {"message": "计算苯-甲苯在1 atm下的T-x-y曲线"},
    },
    {
        "name": "temperature_unit_normalization",
        "request": {"message": "计算甲烷乙烷混合物在-163摄氏度、100 kPa下TP Flash"},
    },
    # =====================================================
    # 超范围拒绝
    # =====================================================
    {
        "name": "out_of_scope_task",
        "request": {"message": "帮我设计一个核电站控制系统"},
    },
    {
        "name": "non_thermo_question",
        "request": {"message": "今天天气怎么样"},
    },
    # =====================================================
    # 导出请求
    # =====================================================
    {
        "name": "export_result_request",
        "request": {"message": "计算苯-甲苯T-x-y并导出CSV结果"},
    },
]


# =========================
# 输出文件
# =========================

OUTPUT_FILE = Path(__file__).parent / "chat_request_response.json"


# =========================
# 请求函数
# =========================


def send_request(message: dict):
    try:
        response = requests.post(
            API_URL,
            json=message,
            timeout=120,
        )

        try:
            body = response.json()

        except Exception:
            body = response.text

        return {
            "status_code": response.status_code,
            "response_headers": dict(response.headers),
            "body": body,
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# 主测试
# =========================


def main():
    results = []

    print("=" * 70)
    print("Start /api/chat black-box testing")
    print("=" * 70)

    for case in TEST_CASES:
        print(f"\nRunning: {case['name']}")

        request_data = case["request"]

        response_data = send_request(request_data)

        result = {
            "case": case["name"],
            "time": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data,
        }

        results.append(result)

        print("status:", response_data.get("status_code", "ERROR"))

    # 保存结果

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 70)

    print("Finished")

    print("Result saved:", OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()
