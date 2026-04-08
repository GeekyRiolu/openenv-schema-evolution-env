from __future__ import annotations

import random

from app.models import TaskSpec


TRANSACTION_ROW_COUNT = 500
AUDIT_LOG_ROW_COUNT = 500

random.seed(42)

EXPECTED_SUM = 0.0
FORMATTED_VALUES: list[str] = []

for index in range(TRANSACTION_ROW_COUNT):
    value = round(random.uniform(1.0, 9999.99), 2)
    EXPECTED_SUM += value
    format_name = random.choice(["plain", "dollar", "comma", "both"])
    if format_name == "plain":
        formatted_value = f"{value:.2f}"
    elif format_name == "dollar":
        formatted_value = f"${value:.2f}"
    elif format_name == "comma":
        formatted_value = f"{value:,.2f}"
    else:
        formatted_value = f"${value:,.2f}"
    FORMATTED_VALUES.append(formatted_value)

EXPECTED_SUM = round(EXPECTED_SUM, 2)


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _build_setup_sql() -> str:
    statements = [
        """
        CREATE TABLE transactions (
            id INTEGER PRIMARY KEY,
            amount TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """.strip(),
    ]

    for index, amount in enumerate(FORMATTED_VALUES, start=1):
        created_at = f"2024-03-{((index - 1) % 28) + 1:02d}T08:30:00"
        statements.append(
            "INSERT INTO transactions (id, amount, created_at) "
            f"VALUES ({index}, '{_sql_quote(amount)}', '{created_at}');"
        )

    statements.extend(
        [
            """
            CREATE VIEW summary_views AS
            SELECT id, amount, created_at
            FROM transactions;
            """.strip(),
            """
            CREATE TABLE audit_log (
                id INTEGER PRIMARY KEY,
                transaction_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """.strip(),
        ]
    )

    for index in range(1, AUDIT_LOG_ROW_COUNT + 1):
        created_at = f"2024-03-{((index - 1) % 28) + 1:02d}T09:00:00"
        statements.append(
            "INSERT INTO audit_log (id, transaction_id, action, created_at) "
            f"VALUES ({index}, {index}, 'seeded', '{created_at}');"
        )

    return "\n".join(statements)


TASK_SPEC = TaskSpec(
    id="task3_type_change",
    name="Zero-downtime column type migration",
    difficulty="hard",
    description=(
        "Migrate transactions.amount from noisy text into numeric storage without breaking "
        "dependent objects."
    ),
    goal=(
        "Migrate transactions.amount from TEXT to REAL, clean formatting noise, preserve "
        "summary_views and audit_log, and match the deterministic expected total."
    ),
)

SETUP_SQL = _build_setup_sql()
