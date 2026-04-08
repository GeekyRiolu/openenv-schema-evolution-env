---
title: Schema Evolution Env
emoji: "🗄️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# Schema Evolution Env

Schema Evolution Env is an OpenEnv-compatible FastAPI environment for training and evaluating agents that perform real SQLite schema migrations. Each episode starts from a seeded in-memory database and asks the agent to inspect the current schema, run migrations, validate integrity, recover with rollback when needed, and finally submit a completed migration for deterministic grading.

## Why This Environment Matters

Database schema evolution is a real operational problem: teams need to change schemas without losing data, violating constraints, or breaking downstream consumers. This environment captures that workflow with three tasks that scale from a simple additive migration to a multi-step type migration with dependent objects.

## Action Space

| Action | Params | Description |
| --- | --- | --- |
| `inspect_schema` | `{"table": "all"}` or a table name | Returns structured schema details for all tables or one table. |
| `sample_data` | `{"table": "<table>", "limit": 5}` | Returns the first rows from a table as a markdown table. |
| `run_migration` | `{"sql": "<sql script>"}` | Executes migration SQL against the live episode database. |
| `validate_constraints` | `{}` | Runs `PRAGMA integrity_check` and `PRAGMA foreign_key_check`. |
| `rollback` | `{}` | Restores the seeded database state for the current task. |
| `submit_final` | `{}` | Runs the deterministic grader and ends the episode. |

## Observation Space

| Field | Type | Description |
| --- | --- | --- |
| `task_id` | `string` | Current task identifier. |
| `step` | `integer` | Number of actions taken in the episode. |
| `schema` | `object` | Current table-to-column schema snapshot. |
| `last_action_result` | `string \| null` | Result text from the most recent action. |
| `cumulative_reward` | `number` | Current normalized reward in `[0.0, 1.0]`. |
| `done` | `boolean` | Whether the episode has terminated. |
| `goal` | `string` | Human-readable migration goal for the task. |

## Reward Function

- Successful `run_migration`: `+0.05`
- Successful `validate_constraints`: `+0.05`
- `rollback`: `+0.0`
- `submit_final`: replaces cumulative reward with the grader score
- Max steps reached: episode ends with the current cumulative reward

All rewards are clamped to `[0.0, 1.0]`.

## Tasks

| Task | Difficulty | Goal |
| --- | --- | --- |
| `task1_add_column` | Easy | Add `users.is_verified BOOLEAN NOT NULL DEFAULT 0` without losing rows. |
| `task2_split_table` | Medium | Normalize repeated customer data into `customers` and populate `orders.customer_id`. |
| `task3_type_change` | Hard | Convert `transactions.amount` from noisy text to `REAL` while preserving `summary_views` and `audit_log`. |

## Quick Start

### Reset an episode

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1_add_column"}'
```

### Take a step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"type":"inspect_schema","params":{"table":"all"}}}'
```

### Read current state

```bash
curl http://localhost:7860/state
```

## Local Setup

### Docker

```bash
docker build -t schema-env .
docker run -p 7860:7860 schema-env
```

### Python

```bash
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Baseline Agent

`inference.py` implements a baseline agent that talks to the environment over HTTP and emits the required `[START]`, `[STEP]`, and `[END]` logs for automated evaluation.

### Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `API_BASE_URL` | No | OpenAI-compatible base URL for the action model. |
| `MODEL_NAME` | No | Model identifier for action generation. |
| `HF_TOKEN` | Yes | API key used by the OpenAI client. |
| `LOCAL_IMAGE_NAME` | No | Optional local image name for Docker-based workflows. |
| `ENV_URL` | No | Environment base URL, defaults to `http://localhost:8000`. |

### Baseline scores

Baseline scores are not included yet because running `inference.py` requires a configured `HF_TOKEN` and a reachable OpenAI-compatible endpoint. Once those credentials are available, run:

```bash
HF_TOKEN=... API_BASE_URL=... MODEL_NAME=... python3 inference.py
```

## Tests

```bash
pytest tests/ -v
```
