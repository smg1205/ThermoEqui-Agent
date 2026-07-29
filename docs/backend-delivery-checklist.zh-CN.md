# 后端第一阶段交付清单

## 1. 环境

推荐 Python 3.12。当前本机环境位于 `E:\anaconda3\envs\thermoequi`。

```powershell
conda activate E:\anaconda3\envs\thermoequi
Copy-Item .env.example .env
python -m uvicorn apps.api.main:app --reload --port 8000
```

默认 `LLM_PROVIDER=deterministic`，后端闭环不依赖外部 API Key。

## 2. 演示顺序

1. 打开 `http://localhost:8000/health`，确认服务与 Provider 状态。
2. 打开 `http://localhost:8000/docs`，确认 OpenAPI 路由。
3. 调用 `/api/chat` 提交“计算苯-甲苯在 101.325 kPa 下的 T-x-y 曲线”。
4. 从响应取得 `run_id`，调用 `/api/runs` 和 `/api/runs/{run_id}`。
5. 调用 `/api/runs/{run_id}/export?format=json` 与 `format=csv`。
6. 展示验证报告、参数来源、后端版本和 request ID。
7. 提交缺参、非法范围和超出当前科学边界的请求，展示结构化失败。

## 3. 质量门

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy agent apps database schemas thermo_engine
```

涉及公共 Schema 时还要运行前端测试、Lint 和构建。

## 4. 后端验收点

- 所有公共输入输出经过 Pydantic。
- 聊天消息、任务、运行、验证和证据原子写入。
- 直接计算也保存对应 TaskRow。
- 运行历史支持倒序分页和状态筛选。
- 参数值有限、单位完整、适用范围合法、来源可追溯。
- 重复参数集返回 `409 duplicate_parameter_set`。
- 未预期异常返回脱敏 `500 internal_server_error`。
- 每个正常和错误响应均带 `X-Request-ID`。
- 所有数值来自 `thermo_engine` 并经过独立验证。
- 测试夹具不会进入生产参数库。

## 5. 当前非交付范围

- 电解质、反应平衡、SLE、聚合物、水合物、石油假组分、VLLE 和流程设计。
- 生产 NRTL/UNIQUAC LLE、参数回归、多模型敏感性分析。
- 用户认证授权、正式 Alembic 迁移和 PostgreSQL 驱动部署。
