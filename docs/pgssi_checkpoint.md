# PGSSI private weights configuration

PGSSI is registered as a first-class backend, but its predictions require a
**trained model checkpoint** (`.pth`). The checkpoint is a private artifact of
the research group and is **never committed to this repository**. This document
explains how members and deployments provide their own weights so the feature
works without exposing the weights to anyone who should not see them.

## Design: weights stay private, code stays public

```
Public repository (git-tracked)          Private weights (git-ignored)
----------------------------------       ----------------------------------
thermo_engine/pgssi_backend.py            weights/PGSSI_best.pth   (or anywhere
  reads PGSSI_CHECKPOINT / PGSSI_SRC      on your machine)
.env.example (explains the variables)     .env                     (real values)
docs/pgssi_checkpoint.md                  docker volume mount      (production)
```

- The backend **only reads environment variables**; it never contains a path
  or the weight bytes.
- `.env` and `weights/` are in `.gitignore`; `*.pth`/`*.pt`/`*.ckpt` are also
  ignored globally as a safety net.
- Members who do not have weights simply leave the variables blank: PGSSI stays
  registered and returns a structured `missing_parameters` failure instead of
  fabricating numbers.

## Required variables

| Variable | Meaning |
|---|---|
| `PGSSI_CHECKPOINT` | Absolute path to the trained `*_best.pth` (or `*_resume.pth`) |
| `PGSSI_SRC` | Absolute path to the PGSSI repository `src/models/PGSSI` directory |
| `PGSSI_HIDDEN_DIM` | Hidden dimension used at training time (default `512`) |
| `PGSSI_ENABLE_CROSS_INTERACTION` | `1` if cross-interaction was enabled (default `1`) |

## Local development

1. Copy `.env.example` to `.env` (already git-ignored).
2. Set the four variables above to **your** private paths.
3. Restart the backend (`python -m uvicorn apps.api.main:app --port 8000`).

The values are read lazily at request time, so the order in which `.env` is
loaded does not matter.

## Docker deployment

`docker-compose.yml` passes the `PGSSI_*` variables through and mounts
`./weights` (host) read-only into `/weights` inside the container:

```bash
# host .env
PGSSI_CHECKPOINT=/weights/PGSSI_best.pth
PGSSI_SRC=/opt/PGSSI/src/models/PGSSI
PGSSI_WEIGHTS_DIR=/absolute/host/path/to/weights
```

The weights are mounted from the host and are **never baked into the image**,
so the image itself is safe to share.

## Verification

With a checkpoint configured, request gamma-infinity:

```
POST /api/calculations/infinite-dilution-activity
{ "components": [{"component_id":"ethanol","name":"ethanol","smiles":"CCO","cas_number":"64-17-5","aliases":[]},
                 {"component_id":"water","name":"water","smiles":"O","cas_number":"7732-18-5","aliases":[]}],
  "conditions": {"temperature_K": 298.15} }
```

Expected: a `CalculationEnvelope` with `model_name = "PGSSI"` and
`gamma_infinity` points. Without a checkpoint the same request returns a
structured `missing_parameters` failure mentioning the checkpoint.
