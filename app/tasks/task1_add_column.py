from __future__ import annotations

from app.models import TaskSpec


def _build_setup_sql() -> str:
    statements = [
        "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL, created_at TEXT NOT NULL);",
    ]
    for index in range(1, 101):
        statements.append(
            "INSERT INTO users (id, email, created_at) "
            f"VALUES ({index}, 'user{index}@example.com', '2024-01-{((index - 1) % 28) + 1:02d}T00:00:00');"
        )
    return "\n".join(statements)


TASK_SPEC = TaskSpec(
    id="task1_add_column",
    name="Add column with default",
    difficulty="easy",
    description=(
        "Add a verification column to users without losing rows or allowing NULL defaults."
    ),
    goal="Add users.is_verified BOOLEAN NOT NULL DEFAULT 0 without losing any rows.",
)

SETUP_SQL = _build_setup_sql()
EXPECTED_ROW_COUNT = 100
