from __future__ import annotations

import json
import os
from typing import Any

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "<your-api-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-action-model>")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

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

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def log_start(task_id: str, difficulty: str) -> None:
    print(f"[START] task_id={task_id} difficulty={difficulty}", flush=True)


def log_step(step: int, action_type: str, reward: float, done: bool) -> None:
    print(
        f"[STEP] step={step} action={action_type} reward={reward:.4f} done={str(done).lower()}",
        flush=True,
    )


def log_end(task_id: str, total_reward: float, steps: int) -> None:
    print(f"[END] task_id={task_id} total_reward={total_reward:.4f} steps={steps}", flush=True)


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


def run_episode(task_id: str) -> float:
    reset_payload = _post_json("/reset", {"task_id": task_id})
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

    while True:
        action = _llm_action(messages)
        step_result = _post_json("/step", {"action": action})
        observation = step_result["observation"]
        steps = int(observation["step"])
        final_reward = float(observation["cumulative_reward"])
        log_step(steps, str(action["type"]), float(step_result["reward"]), bool(step_result["done"]))

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
