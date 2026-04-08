from __future__ import annotations

from app.models import TaskSpec


UNIQUE_CUSTOMER_COUNT = 40
ORDER_ROW_COUNT = 200


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _build_setup_sql() -> str:
    statements = [
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            customer_email TEXT NOT NULL,
            customer_phone TEXT NOT NULL,
            item TEXT NOT NULL,
            amount REAL NOT NULL,
            ordered_at TEXT NOT NULL
        );
        """.strip(),
    ]

    items = ["keyboard", "monitor", "mouse", "dock", "laptop"]
    for index in range(1, ORDER_ROW_COUNT + 1):
        customer_index = ((index - 1) % UNIQUE_CUSTOMER_COUNT) + 1
        name = f"Customer {customer_index}"
        email = f"customer{customer_index}@example.com"
        phone = f"+1-555-01{customer_index:02d}"
        item = items[(index - 1) % len(items)]
        amount = 25.0 + (index % 17) * 3.5
        ordered_at = f"2024-02-{((index - 1) % 28) + 1:02d}T12:00:00"
        statements.append(
            "INSERT INTO orders "
            "(id, customer_name, customer_email, customer_phone, item, amount, ordered_at) "
            f"VALUES ({index}, '{_sql_quote(name)}', '{_sql_quote(email)}', '{_sql_quote(phone)}', "
            f"'{_sql_quote(item)}', {amount:.2f}, '{ordered_at}');"
        )
    return "\n".join(statements)


TASK_SPEC = TaskSpec(
    id="task2_split_table",
    name="Split table and repair foreign keys",
    difficulty="medium",
    description=(
        "Normalize repeated customer data into a dedicated table and connect orders to it."
    ),
    goal=(
        "Create customers, deduplicate rows from orders, add orders.customer_id, and populate "
        "a valid foreign key for every order."
    ),
)

SETUP_SQL = _build_setup_sql()
