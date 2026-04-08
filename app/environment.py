from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.graders.grader import Grader
from app.models import Action, ColumnInfo, Observation, SchemaInfo, StepResult
from app.tasks import TASKS, TaskDefinition


_MAX_FINAL_SCORE = 0.9


def _clamp_reward(value: float) -> float:
    rounded = round(value, 4)
    if rounded <= 0.0:
        return 0.0
    if rounded >= 1.0:
        return _MAX_FINAL_SCORE
    return rounded


class SchemaEvolutionEnv:
    def __init__(self) -> None:
        self.task_id: str | None = None
        self.conn: sqlite3.Connection | None = None
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.max_steps = 20
        self._setup_sql: str | None = None
        self._goal = ""
        self._last_action_result: str | None = None
        self._grader = Grader()

    def reset(self, task_id: str) -> Observation:
        task = self._get_task(task_id)
        self._close_connection()
        self.conn = self._create_connection()
        self._apply_setup(task.setup_sql)
        self.task_id = task.spec.id
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self._setup_sql = task.setup_sql
        self._goal = task.spec.goal
        self._last_action_result = None
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        if self.conn is None or self.task_id is None:
            self._last_action_result = "Environment not initialized. Call /reset first."
            return self._build_step_result(0.0, {"error": self._last_action_result})

        if self.done:
            self._last_action_result = "Episode finished. Call /reset first."
            return self._build_step_result(0.0, {"error": self._last_action_result})

        self.step_count += 1

        if action.type == "inspect_schema":
            self._last_action_result = self._handle_inspect_schema(action.params)
            reward = 0.0
            info: dict[str, Any] = {}
        elif action.type == "sample_data":
            self._last_action_result = self._handle_sample_data(action.params)
            reward = 0.0
            info = {}
        elif action.type == "run_migration":
            self._last_action_result = self._handle_run_migration(action.params)
            reward = 0.05 if self._last_action_result == "Migration applied successfully." else 0.0
            info = {}
        elif action.type == "validate_constraints":
            self._last_action_result = self._handle_validate_constraints(action.params)
            reward = 0.05 if self._last_action_result == "Integrity check OK; foreign key check OK." else 0.0
            info = {}
        elif action.type == "rollback":
            self._last_action_result = self._handle_rollback(action.params)
            reward = 0.0
            info = {}
        elif action.type == "submit_final":
            return self._handle_submit_final(action.params)
        else:
            self._last_action_result = f"Unsupported action: {action.type}"
            reward = 0.0
            info = {"error": self._last_action_result}

        self.cumulative_reward = _clamp_reward(self.cumulative_reward + reward)
        if self.step_count >= self.max_steps:
            self.done = True

        return self._build_step_result(reward, info)

    def state(self) -> Observation:
        return self._build_observation()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _close_connection(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _get_task(self, task_id: str) -> TaskDefinition:
        task = TASKS.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")
        return task

    def _build_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id or "",
            step=self.step_count,
            schema_info=self._get_schema(),
            last_action_result=self._last_action_result,
            cumulative_reward=_clamp_reward(self.cumulative_reward),
            done=self.done,
            goal=self._goal,
        )

    def _build_step_result(self, reward: float, info: dict[str, Any]) -> StepResult:
        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=_clamp_reward(reward),
            done=observation.done,
            info=info,
        )

    def _apply_setup(self, setup_sql: str) -> None:
        if self.conn is None:
            raise RuntimeError("No active database connection.")
        self.conn.executescript(setup_sql)
        self.conn.execute("PRAGMA foreign_keys = ON")

    def _get_schema(self) -> SchemaInfo:
        if self.conn is None:
            return SchemaInfo(tables={})

        table_rows = self.conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
            """
        ).fetchall()

        tables: dict[str, list[ColumnInfo]] = {}
        for table_row in table_rows:
            table_name = str(table_row["name"])
            column_rows = self.conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
            tables[table_name] = [
                ColumnInfo(
                    name=str(column_row["name"]),
                    type=str(column_row["type"]),
                    notnull=bool(column_row["notnull"]),
                    default_value=(
                        None
                        if column_row["dflt_value"] is None
                        else str(column_row["dflt_value"])
                    ),
                    primary_key=bool(column_row["pk"]),
                )
                for column_row in column_rows
            ]
        return SchemaInfo(tables=tables)

    def _existing_tables(self) -> set[str]:
        return set(self._get_schema().tables)

    def _validate_table_name(self, table_name: str) -> str | None:
        if table_name not in self._existing_tables():
            return None
        return table_name

    def _handle_inspect_schema(self, params: dict[str, Any]) -> str:
        requested_table = str(params.get("table", "all"))
        schema = self._get_schema()
        if requested_table == "all":
            return schema.model_dump_json(indent=2)
        valid_table = self._validate_table_name(requested_table)
        if valid_table is None:
            return f"Unknown table: {requested_table}"
        return json.dumps(
            {
                "table": valid_table,
                "columns": [column.model_dump() for column in schema.tables[valid_table]],
            },
            indent=2,
        )

    def _handle_sample_data(self, params: dict[str, Any]) -> str:
        if self.conn is None:
            return "Environment not initialized. Call /reset first."

        table_name = str(params.get("table", ""))
        valid_table = self._validate_table_name(table_name)
        if valid_table is None:
            return f"Unknown table: {table_name}"

        limit = params.get("limit", 5)
        try:
            numeric_limit = max(1, min(int(limit), 20))
        except (TypeError, ValueError):
            return "Invalid limit parameter."

        rows = self.conn.execute(
            f'SELECT * FROM "{valid_table}" LIMIT ?;',
            (numeric_limit,),
        ).fetchall()
        if not rows:
            return f"No rows found in {valid_table}."

        headers = list(rows[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
        return "\n".join(lines)

    def _handle_run_migration(self, params: dict[str, Any]) -> str:
        if self.conn is None:
            return "Environment not initialized. Call /reset first."

        sql = params.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return "Missing sql parameter."

        try:
            self.conn.executescript(sql)
            return "Migration applied successfully."
        except sqlite3.Error as exc:
            self.conn.rollback()
            return f"Migration failed: {exc}"

    def _handle_validate_constraints(self, params: dict[str, Any]) -> str:
        if self.conn is None:
            return "Environment not initialized. Call /reset first."

        integrity_result = self.conn.execute("PRAGMA integrity_check;").fetchone()[0]
        foreign_key_rows = self.conn.execute("PRAGMA foreign_key_check;").fetchall()
        if integrity_result == "ok" and not foreign_key_rows:
            return "Integrity check OK; foreign key check OK."
        return (
            f"Integrity check: {integrity_result}; "
            f"foreign key issues: {len(foreign_key_rows)}"
        )

    def _handle_rollback(self, params: dict[str, Any]) -> str:
        if self._setup_sql is None:
            return "No setup SQL available for rollback."
        self._close_connection()
        self.conn = self._create_connection()
        self._apply_setup(self._setup_sql)
        return "Database rolled back to initial state."

    def _handle_submit_final(self, params: dict[str, Any]) -> StepResult:
        if self.conn is None or self.task_id is None:
            self._last_action_result = "Environment not initialized. Call /reset first."
            return self._build_step_result(0.0, {"error": self._last_action_result})

        grade = self._grader.grade(self.conn, self.task_id)
        self.done = True
        self.cumulative_reward = _clamp_reward(float(grade["total_reward"]))
        self._last_action_result = str(grade["message"])
        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=self.cumulative_reward,
            done=True,
            info=grade,
        )
