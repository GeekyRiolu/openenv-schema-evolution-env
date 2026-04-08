from __future__ import annotations

import os
import requests
from typing import Any

os.environ.setdefault("HF_TOKEN", "test-token")
os.environ.setdefault("API_BASE_URL", "https://example.com/v1")
os.environ.setdefault("MODEL_NAME", "test-model")

import inference
from inference import TASK3_RECOVERY_SQL, _controlled_action, _next_action, run_episode


def _history_entry(
    action_type: str,
    last_action_result: str,
    reward: float,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action": {"type": action_type, "params": params or {}},
        "observation": {"last_action_result": last_action_result},
        "reward": reward,
    }


def test_controlled_action_uses_task3_recovery_after_failed_loop() -> None:
    history = [
        _history_entry(
            "run_migration",
            'Migration failed: near "bad": syntax error',
            0.0,
            {"sql": "bad sql"},
        ),
        _history_entry(
            "rollback",
            "Database rolled back to initial state.",
            0.0,
        ),
    ]

    action = _controlled_action(
        "task3_type_change",
        {"type": "inspect_schema", "params": {"table": "all"}},
        history,
    )

    assert action == {"type": "run_migration", "params": {"sql": TASK3_RECOVERY_SQL}}


def test_controlled_action_submits_after_successful_task3_validation() -> None:
    history = [
        _history_entry(
            "run_migration",
            "Migration applied successfully.",
            0.05,
            {"sql": TASK3_RECOVERY_SQL},
        ),
        _history_entry(
            "validate_constraints",
            "Integrity check OK; foreign key check OK.",
            0.05,
        ),
    ]

    action = _controlled_action(
        "task3_type_change",
        {"type": "inspect_schema", "params": {"table": "all"}},
        history,
    )

    assert action == {"type": "submit_final", "params": {}}


def test_next_action_prefers_controller_rollback_over_llm_loop() -> None:
    history = [
        _history_entry(
            "run_migration",
            "Migration failed: syntax error",
            0.0,
            {"sql": "bad sql"},
        )
    ]

    action = _next_action(
        "task3_type_change",
        [{"role": "system", "content": "x"}],
        history,
        lambda _messages: {"type": "inspect_schema", "params": {"table": "all"}},
    )

    assert action == {"type": "rollback", "params": {}}


def test_controlled_action_keeps_task2_running_after_validation() -> None:
    history = [
        _history_entry(
            "run_migration",
            "Migration applied successfully.",
            0.05,
            {"sql": "partial migration"},
        ),
        _history_entry(
            "validate_constraints",
            "Integrity check OK; foreign key check OK.",
            0.05,
        ),
    ]

    action = _controlled_action(
        "task2_split_table",
        {"type": "inspect_schema", "params": {"table": "orders"}},
        history,
    )

    assert action == {"type": "inspect_schema", "params": {"table": "orders"}}


def test_controlled_action_runs_recovery_sql_after_task3_rollback() -> None:
    history = [
        _history_entry(
            "run_migration",
            "Migration failed: syntax error",
            0.0,
            {"sql": "bad sql"},
        ),
        _history_entry(
            "rollback",
            "Database rolled back to initial state.",
            0.0,
        ),
    ]

    action = _controlled_action(
        "task3_type_change",
        {"type": "run_migration", "params": {"sql": "another bad sql"}},
        history,
    )

    assert action == {"type": "run_migration", "params": {"sql": TASK3_RECOVERY_SQL}}


def test_controlled_action_breaks_task3_inspect_loop_with_recovery_sql() -> None:
    history = [
        _history_entry("inspect_schema", '{"table":"transactions"}', 0.0),
        _history_entry("inspect_schema", '{"table":"transactions"}', 0.0),
        _history_entry("inspect_schema", '{"table":"transactions"}', 0.0),
        _history_entry("inspect_schema", '{"table":"transactions"}', 0.0),
    ]

    action = _controlled_action(
        "task3_type_change",
        {"type": "inspect_schema", "params": {"table": "all"}},
        history,
    )

    assert action == {"type": "run_migration", "params": {"sql": TASK3_RECOVERY_SQL}}


def test_run_episode_handles_reset_connection_error(monkeypatch: Any) -> None:
    def _raise_error(_path: str, _payload: dict[str, Any]) -> dict[str, Any]:
        raise requests.RequestException("connection failed")

    monkeypatch.setattr(inference, "_post_json", _raise_error)

    assert run_episode("task1_add_column") == 0.0
