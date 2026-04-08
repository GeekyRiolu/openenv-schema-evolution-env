from __future__ import annotations

import sqlite3
from typing import Any

from app.tasks.task1_add_column import EXPECTED_ROW_COUNT
from app.tasks.task3_type_change import (
    AUDIT_LOG_ROW_COUNT,
    EXPECTED_SUM,
    TRANSACTION_ROW_COUNT,
)


def _clamp_reward(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


class Grader:
    def grade(self, conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
        try:
            if task_id == "task1_add_column":
                return self._grade_task1(conn)
            if task_id == "task2_split_table":
                return self._grade_task2(conn)
            if task_id == "task3_type_change":
                return self._grade_task3(conn)
            return self._build_result({}, "Unknown task.", False)
        except Exception as exc:
            return self._build_result(
                {},
                f"Grader failed safely: {exc}",
                False,
            )

    def _build_result(
        self,
        breakdown: dict[str, float],
        message: str,
        passed: bool,
    ) -> dict[str, Any]:
        total_reward = _clamp_reward(sum(breakdown.values()))
        return {
            "total_reward": total_reward,
            "breakdown": breakdown,
            "passed": passed and total_reward >= 0.9,
            "message": message,
        }

    def _grade_task1(self, conn: sqlite3.Connection) -> dict[str, Any]:
        breakdown = {
            "column_exists_with_correct_type": 0.0,
            "default_zero": 0.0,
            "row_count_unchanged": 0.0,
            "not_null_constraint": 0.0,
        }

        columns = conn.execute("PRAGMA table_info(users);").fetchall()
        row_count = conn.execute("SELECT COUNT(*) FROM users;").fetchone()[0]
        user_column = next((column for column in columns if column[1] == "is_verified"), None)

        if user_column is not None and user_column[2].upper() == "BOOLEAN":
            breakdown["column_exists_with_correct_type"] = 0.4
        if user_column is not None and str(user_column[4]).strip("'") == "0":
            breakdown["default_zero"] = 0.3
        if row_count == EXPECTED_ROW_COUNT:
            breakdown["row_count_unchanged"] = 0.2
        if user_column is not None and bool(user_column[3]):
            breakdown["not_null_constraint"] = 0.1

        return self._build_result(
            breakdown,
            "Task 1 graded successfully.",
            True,
        )

    def _grade_task2(self, conn: sqlite3.Connection) -> dict[str, Any]:
        breakdown = {
            "customers_schema": 0.0,
            "deduplicated_customers": 0.0,
            "orders_customer_id_column": 0.0,
            "customer_links_complete": 0.0,
            "foreign_key_integrity": 0.0,
        }

        customers_exists = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'customers';"
        ).fetchone()[0]
        if customers_exists:
            customers_columns = conn.execute("PRAGMA table_info(customers);").fetchall()
            column_map = {column[1]: column for column in customers_columns}
            unique_indexes = conn.execute("PRAGMA index_list(customers);").fetchall()
            email_unique = False
            for index_row in unique_indexes:
                if not index_row[2]:
                    continue
                indexed_columns = conn.execute(
                    f"PRAGMA index_info('{index_row[1]}');"
                ).fetchall()
                if any(indexed_column[2] == "email" for indexed_column in indexed_columns):
                    email_unique = True
                    break
            if (
                set(column_map) == {"id", "name", "email", "phone"}
                and column_map["id"][5] == 1
                and column_map["email"][2].upper() == "TEXT"
                and email_unique
            ):
                breakdown["customers_schema"] = 0.25

            expected_unique = conn.execute(
                "SELECT COUNT(DISTINCT customer_email) FROM orders;"
            ).fetchone()[0]
            actual_unique = conn.execute("SELECT COUNT(*) FROM customers;").fetchone()[0]
            missing_unique = conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT customer_email AS email
                    FROM orders
                    EXCEPT
                    SELECT email
                    FROM customers
                );
                """
            ).fetchone()[0]
            duplicate_unique = conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT email
                    FROM customers
                    GROUP BY email
                    HAVING COUNT(*) > 1
                );
                """
            ).fetchone()[0]
            if actual_unique == expected_unique and missing_unique == 0 and duplicate_unique == 0:
                breakdown["deduplicated_customers"] = 0.2

        orders_columns = conn.execute("PRAGMA table_info(orders);").fetchall()
        orders_column_map = {column[1]: column for column in orders_columns}
        if "customer_id" in orders_column_map:
            breakdown["orders_customer_id_column"] = 0.25

        if customers_exists and "customer_id" in orders_column_map:
            linked_rows = conn.execute(
                """
                SELECT COUNT(*)
                FROM orders AS o
                JOIN customers AS c
                  ON c.id = o.customer_id
                WHERE o.customer_id IS NOT NULL
                  AND o.customer_email = c.email
                  AND o.customer_name = c.name
                  AND o.customer_phone = c.phone;
                """
            ).fetchone()[0]
            total_rows = conn.execute("SELECT COUNT(*) FROM orders;").fetchone()[0]
            null_customer_ids = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE customer_id IS NULL;"
            ).fetchone()[0]
            if total_rows > 0 and linked_rows == total_rows and null_customer_ids == 0:
                breakdown["customer_links_complete"] = 0.2

        foreign_key_issues = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if not foreign_key_issues:
            breakdown["foreign_key_integrity"] = 0.1

        return self._build_result(
            breakdown,
            "Task 2 graded successfully.",
            True,
        )

    def _grade_task3(self, conn: sqlite3.Connection) -> dict[str, Any]:
        breakdown = {
            "numeric_column_type": 0.0,
            "all_rows_numeric": 0.0,
            "expected_sum": 0.0,
            "summary_view_intact": 0.0,
            "audit_log_intact": 0.0,
        }

        transaction_columns = conn.execute("PRAGMA table_info(transactions);").fetchall()
        amount_column = next(
            (column for column in transaction_columns if column[1] == "amount"),
            None,
        )
        if amount_column is not None and amount_column[2].upper() in {"REAL", "NUMERIC"}:
            breakdown["numeric_column_type"] = 0.2

        numeric_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM transactions
            WHERE amount IS NOT NULL
              AND typeof(amount) IN ('real', 'integer');
            """
        ).fetchone()[0]
        if numeric_rows == TRANSACTION_ROW_COUNT:
            breakdown["all_rows_numeric"] = 0.25

        total_amount = conn.execute("SELECT ROUND(SUM(amount), 2) FROM transactions;").fetchone()[0]
        if total_amount is not None and abs(float(total_amount) - EXPECTED_SUM) <= 0.01:
            breakdown["expected_sum"] = 0.2

        try:
            conn.execute("SELECT COUNT(*) FROM summary_views;").fetchone()[0]
        except sqlite3.Error:
            pass
        else:
            breakdown["summary_view_intact"] = 0.2

        try:
            audit_log_rows = conn.execute("SELECT COUNT(*) FROM audit_log;").fetchone()[0]
        except sqlite3.Error:
            audit_log_rows = None
        if audit_log_rows == AUDIT_LOG_ROW_COUNT:
            breakdown["audit_log_intact"] = 0.15

        return self._build_result(
            breakdown,
            "Task 3 graded successfully.",
            True,
        )
