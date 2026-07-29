# API

FastAPI publishes OpenAPI at `/docs`. Every response carries `X-Request-ID`; errors use
`{"error":{"code","message","details","request_id"}}`. Calculation routes create immutable run
snapshots. `GET /api/runs?limit=20&offset=0&status=passed|warning|failed` returns lightweight run
summaries in reverse chronological order. `GET /api/runs/{run_id}/export?format=json|csv` returns a
downloadable artifact.

Unexpected exceptions return a sanitized `500 internal_server_error` with the request ID in both
the JSON body and response header. Exception messages and database/provider details are not
returned to the client.

The conversation orchestrator supports `deterministic`, `deepseek`, and `openai` providers through
`LLM_PROVIDER`. DeepSeek uses its OpenAI-compatible `/chat/completions` endpoint; external providers
may structure and explain tasks but never emit thermodynamic calculation values.

Key routes: `/api/chat`, `/api/tasks/parse`, `/api/models/recommend`, `/api/models`,
`/api/parameters`, `/api/parameters/search`, all calculation endpoints (including the typed LLE
contract at `/api/calculations/lle`), `/api/validation`, run query and export, and `/health`.

`POST /api/parameters` validates component order, finite values, complete units, applicability
ranges, and source evidence before persistence. Parameter-set IDs are immutable; posting an
existing ID returns `409 duplicate_parameter_set`. Test fixtures are rejected by the production
repository.

Both chat-triggered and structured `/api/calculations/*` runs persist the normalized task manifest
before the immutable run snapshot is written. Chat tasks retain their conversation association;
direct calculation tasks use a null conversation ID.

For `/api/chat`, the conversation, user/assistant messages, task manifest, run, points, validation,
and evidence records commit in one database transaction. A persistence failure rolls back the
entire response snapshot instead of leaving a conversation without its calculation.
