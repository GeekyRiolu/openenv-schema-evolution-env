from __future__ import annotations

import tomllib
from pathlib import Path

from app.environment import SchemaEvolutionEnv
from app.graders.grader import Grader
from app.main import health, index, list_tasks, reset, state, step
from app.models import Action
from app.models import ResetRequest, StepRequest
from app.tasks.task1_add_column import EXPECTED_ROW_COUNT
from app.tasks.task2_split_table import ORDER_ROW_COUNT, UNIQUE_CUSTOMER_COUNT
from app.tasks.task3_type_change import AUDIT_LOG_ROW_COUNT, EXPECTED_SUM, TRANSACTION_ROW_COUNT


def _solve_task2(env: SchemaEvolutionEnv) -> None:
    migration_sql = """
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        phone TEXT NOT NULL
    );
    INSERT INTO customers (name, email, phone)
    SELECT customer_name, customer_email, customer_phone
    FROM orders
    GROUP BY customer_email;
    ALTER TABLE orders ADD COLUMN customer_id INTEGER REFERENCES customers(id);
    UPDATE orders
    SET customer_id = (
        SELECT customers.id
        FROM customers
        WHERE customers.email = orders.customer_email
    );
    """
    env.step(
        Action(type="run_migration", params={"sql": migration_sql})
    )


def _solve_task3(env: SchemaEvolutionEnv) -> None:
    migration_sql = """
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
    """
    env.step(
        Action(type="run_migration", params={"sql": migration_sql})
    )


def test_reset_returns_clean_state() -> None:
    env = SchemaEvolutionEnv()

    first_reset = env.reset("task1_add_column")
    env.step(Action(type="run_migration", params={"sql": "ALTER TABLE users ADD COLUMN is_verified BOOLEAN NOT NULL DEFAULT 0;"}))
    second_reset = env.reset("task1_add_column")

    assert first_reset.task_id == "task1_add_column"
    assert second_reset.step == 0
    assert second_reset.cumulative_reward == 0.0
    assert second_reset.last_action_result is None
    assert "is_verified" not in {
        column.name for column in second_reset.schema_info.tables["users"]
    }


def test_step_inspect_schema() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task1_add_column")

    result = env.step(Action(type="inspect_schema", params={"table": "users"}))

    assert result.reward == 0.0
    assert '"table": "users"' in result.observation.last_action_result


def test_step_run_migration_valid_sql() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task1_add_column")

    result = env.step(
        Action(type="run_migration", params={"sql": "ALTER TABLE users ADD COLUMN is_verified BOOLEAN NOT NULL DEFAULT 0;"})
    )

    assert result.reward == 0.05
    assert result.observation.last_action_result == "Migration applied successfully."


def test_step_run_migration_invalid_sql_no_crash() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task1_add_column")

    result = env.step(
        Action(type="run_migration", params={"sql": "ALTER TABLE missing_table ADD COLUMN broken TEXT;"})
    )

    assert result.reward == 0.0
    assert "Migration failed" in result.observation.last_action_result
    assert not result.done


def test_full_task1_episode_optimal_solution() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task1_add_column")
    env.step(Action(type="run_migration", params={"sql": "ALTER TABLE users ADD COLUMN is_verified BOOLEAN NOT NULL DEFAULT 0;"}))
    env.step(Action(type="validate_constraints", params={}))

    result = env.step(Action(type="submit_final", params={}))

    assert result.done
    assert result.reward == 1.0
    assert result.info["passed"] is True


def test_full_task2_episode() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task2_split_table")
    _solve_task2(env)
    env.step(Action(type="validate_constraints", params={}))

    result = env.step(Action(type="submit_final", params={}))

    assert result.done
    assert result.reward == 1.0
    assert result.info["breakdown"]["deduplicated_customers"] == 0.2
    assert result.info["breakdown"]["customer_links_complete"] == 0.2


def test_full_task3_episode() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task3_type_change")
    _solve_task3(env)
    env.step(Action(type="validate_constraints", params={}))

    result = env.step(Action(type="submit_final", params={}))

    assert result.done
    assert result.reward == 1.0
    assert result.info["breakdown"]["expected_sum"] == 0.2


def test_grader_deterministic() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task1_add_column")
    env.step(Action(type="run_migration", params={"sql": "ALTER TABLE users ADD COLUMN is_verified BOOLEAN NOT NULL DEFAULT 0;"}))
    grader = Grader()

    first_grade = grader.grade(env.conn, "task1_add_column")
    second_grade = grader.grade(env.conn, "task1_add_column")

    assert first_grade == second_grade


def test_reward_range_always_0_to_1() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task1_add_column")

    for _ in range(5):
        result = env.step(Action(type="validate_constraints", params={}))
        assert 0.0 <= result.reward <= 1.0
        assert 0.0 <= result.observation.cumulative_reward <= 1.0


def test_max_steps_terminates_episode() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task1_add_column")

    result = None
    for _ in range(env.max_steps):
        result = env.step(Action(type="inspect_schema", params={"table": "all"}))

    assert result is not None
    assert result.done is True
    assert result.observation.step == env.max_steps


def test_rollback_restores_clean_state() -> None:
    env = SchemaEvolutionEnv()
    env.reset("task1_add_column")
    env.step(Action(type="run_migration", params={"sql": "ALTER TABLE users ADD COLUMN is_verified BOOLEAN NOT NULL DEFAULT 0;"}))

    rollback_result = env.step(Action(type="rollback", params={}))

    assert rollback_result.reward == 0.0
    assert rollback_result.observation.last_action_result == "Database rolled back to initial state."
    assert "is_verified" not in {
        column.name for column in rollback_result.observation.schema_info.tables["users"]
    }
    user_count = env.conn.execute("SELECT COUNT(*) FROM users;").fetchone()[0]
    assert user_count == EXPECTED_ROW_COUNT


def test_api_endpoints_return_expected_shapes() -> None:
    index_response = index()
    health_response = health()
    default_reset_response = reset()
    reset_response = reset(ResetRequest(task_id="task1_add_column"))
    tasks_response = list_tasks()
    step_response = step(
        StepRequest(action=Action(type="inspect_schema", params={"table": "all"}))
    )
    state_response = state()

    assert str(index_response.headers["location"]) == "/docs"
    assert index_response.status_code == 307
    assert health_response == {"status": "ok"}
    assert default_reset_response.task_id == "task1_add_column"
    assert reset_response.task_id == "task1_add_column"
    assert len(tasks_response) == 3
    assert step_response.observation.task_id == "task1_add_column"
    assert state_response.task_id == "task1_add_column"


def test_seeded_task_shapes_match_expected_counts() -> None:
    env = SchemaEvolutionEnv()

    env.reset("task2_split_table")
    order_count = env.conn.execute("SELECT COUNT(*) FROM orders;").fetchone()[0]
    unique_customer_count = env.conn.execute(
        "SELECT COUNT(DISTINCT customer_email) FROM orders;"
    ).fetchone()[0]

    env.reset("task3_type_change")
    transaction_count = env.conn.execute("SELECT COUNT(*) FROM transactions;").fetchone()[0]
    audit_log_count = env.conn.execute("SELECT COUNT(*) FROM audit_log;").fetchone()[0]
    total_amount = env.conn.execute(
        """
        SELECT ROUND(
            SUM(CAST(REPLACE(REPLACE(REPLACE(amount, '$', ''), ',', ''), ' ', '') AS REAL)),
            2
        )
        FROM transactions;
        """
    ).fetchone()[0]

    assert order_count == ORDER_ROW_COUNT
    assert unique_customer_count == UNIQUE_CUSTOMER_COUNT
    assert transaction_count == TRANSACTION_ROW_COUNT
    assert audit_log_count == AUDIT_LOG_ROW_COUNT
    assert float(total_amount) == EXPECTED_SUM


def test_main_entrypoint_exists_for_validator() -> None:
    from app.main import main

    assert callable(main)


def test_pyproject_declares_server_entrypoint_and_openenv_dependency() -> None:
    pyproject_path = Path("pyproject.toml")

    assert pyproject_path.exists()

    data = tomllib.loads(pyproject_path.read_text())
    project = data["project"]

    assert "openenv-core>=0.2.3" in project["dependencies"]
    assert project["scripts"]["server"] == "server.app:main"


def test_server_wrapper_exists_for_validator() -> None:
    server_app_path = Path("server/app.py")

    assert server_app_path.exists()
    content = server_app_path.read_text()

    assert "from app.main import app" in content
    assert "def main() -> None:" in content
    assert "if __name__ == \"__main__\":" in content


def test_server_wrapper_main_is_callable() -> None:
    from server.app import main

    assert callable(main)


def test_uv_lock_exists_for_validator() -> None:
    assert Path("uv.lock").exists()
