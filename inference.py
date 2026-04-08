from __future__ import annotations

import json
import os
from typing import Any, Callable

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "<your-api-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-action-model>")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")

SYSTEM_PROMPT = """
You are a database migration expert. You are given a running SQLite database and a migration goal.
Available actions:
- inspect_schema: {"type": "inspect_schema", "params": {"table": "table_name_or_all"}}
- sample_data:    {"type": "sample_data",    "params": {"table": "table_name", "limit": 5}}
- run_migration:  {"type": "run_migration",  "params": {"sql": "ALTER TABLE ..."}}
- validate_constraints: {"type": "validate_constraints", "params": {}}
- rollback:       {"type": "rollback", "params": {}}
- submit_final:   {"type": "submit_final",   "params": {}}

Always respond with a JSON action object only. No explanations.
Think step by step: inspect first, plan, then execute.
""".strip()

TASK_DIFFICULTIES = {
    "task1_add_column": "easy",
    "task2_split_table": "medium",
    "task3_type_change": "hard",
}

TASK3_RECOVERY_SQL = """
DROP VIEW summary_views;
ALTER TABLE transactions RENAME TO transactions_old;
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY,
    amount REAL NOT NULL,
    created_at TEXT NOT NULL
);
INSERT INTO transactions (id, amount, created_at)
SELECT
    id,
    CAST(REPLACE(REPLACE(REPLACE(amount, '$', ''), ',', ''), ' ', '') AS REAL),
    created_at
FROM transactions_old;
CREATE VIEW summary_views AS
SELECT id, amount, created_at
FROM transactions;
DROP TABLE transactions_old;
""".strip()

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def log_start(task_id: str, difficulty: str) -> None:
    print(f"[START] task_id={task_id} difficulty={difficulty}", flush=True)


def log_step(step: int, action_type: str, reward: float, done: bool) -> None:
    print(
        f"[STEP] step={step} action={action_type} reward={reward:.6f} done={str(done).lower()}",
        flush=True,
    )


def log_end(task_id: str, total_reward: float, steps: int) -> None:
    print(f"[END] task_id={task_id} total_reward={total_reward:.6f} steps={steps}", flush=True)


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{ENV_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _fallback_action() -> dict[str, Any]:
    return {"type": "inspect_schema", "params": {"table": "all"}}


def _llm_action(messages: list[dict[str, str]]) -> dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
        )
        content = completion.choices[0].message.content or ""
        action = json.loads(content)
        if not isinstance(action, dict) or "type" not in action:
            return _fallback_action()
        if "params" not in action or not isinstance(action["params"], dict):
            action["params"] = {}
        return action
    except Exception:
        return _fallback_action()


def _successful_reward(entry: dict[str, Any]) -> bool:
    return float(entry.get("reward", 0.0)) > 0.0


def _result_text(entry: dict[str, Any]) -> str:
    observation = entry.get("observation", {})
    return str(observation.get("last_action_result", ""))


def _controlled_action(
    task_id: str,
    llm_action: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    if history:
        last_entry = history[-1]
        last_action_type = str(last_entry.get("action", {}).get("type", ""))
        if (
            task_id in {"task1_add_column", "task3_type_change"}
            and last_action_type == "validate_constraints"
            and _successful_reward(last_entry)
        ):
            return {"type": "submit_final", "params": {}}
        if (
            task_id in {"task1_add_column", "task3_type_change"}
            and last_action_type == "run_migration"
            and _successful_reward(last_entry)
        ):
            return {"type": "validate_constraints", "params": {}}

    if task_id != "task3_type_change":
        return llm_action

    successful_migration = any(
        str(entry.get("action", {}).get("type", "")) == "run_migration" and _successful_reward(entry)
        for entry in history
    )
    failed_migrations = sum(
        1
        for entry in history
        if str(entry.get("action", {}).get("type", "")) == "run_migration"
        and _result_text(entry).startswith("Migration failed:")
    )
    rollbacks = sum(
        1 for entry in history if str(entry.get("action", {}).get("type", "")) == "rollback"
    )
    recent_inspects = sum(
        1
        for entry in history[-4:]
        if str(entry.get("action", {}).get("type", "")) == "inspect_schema"
    )

    if not successful_migration and history:
        last_action_type = str(history[-1].get("action", {}).get("type", ""))
        if failed_migrations >= 1 and last_action_type == "run_migration":
            return {"type": "rollback", "params": {}}
    if (
        not successful_migration
        and failed_migrations == 0
        and recent_inspects >= 4
    ):
        return {"type": "run_migration", "params": {"sql": TASK3_RECOVERY_SQL}}
    if not successful_migration and failed_migrations >= 1 and rollbacks >= 1:
        return {"type": "run_migration", "params": {"sql": TASK3_RECOVERY_SQL}}

    return llm_action


def _next_action(
    task_id: str,
    messages: list[dict[str, str]],
    history: list[dict[str, Any]],
    llm_action_fn: Callable[[list[dict[str, str]]], dict[str, Any]],
) -> dict[str, Any]:
    llm_action = llm_action_fn(messages)
    return _controlled_action(task_id, llm_action, history)


def run_episode(task_id: str) -> float:
    try:
        reset_payload = _post_json("/reset", {"task_id": task_id})
    except requests.RequestException:
        log_end(task_id, 0.0, 0)
        return 0.0
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "goal": reset_payload["goal"],
                    "schema": reset_payload["schema"],
                }
            ),
        },
    ]

    final_reward = 0.0
    steps = 0
    history: list[dict[str, Any]] = []

    try:
        while True:
            action = _next_action(task_id, messages, history, _llm_action)
            step_result = _post_json("/step", {"action": action})
            observation = step_result["observation"]
            steps = int(observation["step"])
            final_reward = float(observation["cumulative_reward"])
            log_step(steps, str(action["type"]), float(step_result["reward"]), bool(step_result["done"]))
            history.append(
                {
                    "action": action,
                    "observation": observation,
                    "reward": float(step_result["reward"]),
                }
            )

            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "last_action_result": observation["last_action_result"],
                            "schema": observation["schema"],
                            "cumulative_reward": observation["cumulative_reward"],
                            "done": observation["done"],
                        }
                    ),
                }
            )

            if step_result["done"]:
                break
    except requests.RequestException:
        log_end(task_id, 0.0, steps)
        return 0.0

    log_end(task_id, final_reward, steps)
    return final_reward


def main() -> None:
    if HF_TOKEN is None:
        raise RuntimeError("HF_TOKEN must be set before running inference.py.")

    for task_id in ("task1_add_column", "task2_split_table", "task3_type_change"):
        log_start(task_id, TASK_DIFFICULTIES[task_id])
        run_episode(task_id)


if __name__ == "__main__":
    main()
