from __future__ import annotations

import sqlite3
from typing import Any
from app.reward_bounds import safe_reward
from app.reward_bounds import MAX_REPORTED_REWARD, MIN_REPORTED_REWARD, clamp_open_interval
from app.tasks.task1_add_column import EXPECTED_ROW_COUNT
from app.tasks.task3_type_change import (
    AUDIT_LOG_ROW_COUNT,
    EXPECTED_SUM,
    TRANSACTION_ROW_COUNT,
)


def _no_credit() -> float:
    return MIN_REPORTED_REWARD


def _clamp_total(value: float) -> float:
    return safe_reward(value)

class Grader:
    def _build_result(self, breakdown, message, passed):
        raw_sum = sum(breakdown.values())
        if raw_sum >= 1.0 - 1e-9:
            scale = 0.85 / raw_sum  # scale to 0.85, well away from any boundary
            breakdown = {k: safe_reward(round(v * scale, 4)) for k, v in breakdown.items()}
            raw_sum = sum(breakdown.values())
        breakdown = {k: safe_reward(v) for k, v in breakdown.items()}
        total_reward = safe_reward(sum(breakdown.values()))
        return {
            "total_reward": total_reward,
            "breakdown": breakdown,
            "passed": passed and total_reward >= 0.8,
            "message": message,
        }

    def _build_result(
        self,
        breakdown: dict[str, float],
        message: str,
        passed: bool,
    ) -> dict[str, Any]:
        # Validators require every numeric score strictly in (0, 1). A perfect rubric sums to
        # 1.0 before scaling; scale down so breakdown + total never hit 1.0.
        raw_sum = sum(breakdown.values())
        if raw_sum >= 1.0 - 1e-9:
            scale = MAX_REPORTED_REWARD / raw_sum
            breakdown = {k: round(v * scale, 4) for k, v in breakdown.items()}
            raw_sum = sum(breakdown.values())
        breakdown = {k: clamp_open_interval(v) for k, v in breakdown.items()}
        raw_sum = sum(breakdown.values())
        total_reward = _clamp_total(raw_sum)
        return {
            "total_reward": total_reward,
            "breakdown": breakdown,
            "passed": passed and total_reward >= MAX_REPORTED_REWARD,
            "message": message,
        }

    def _grade_task1(self, conn: sqlite3.Connection) -> dict[str, Any]:
        breakdown = {
            "column_exists_with_correct_type": _no_credit(),
            "default_zero": _no_credit(),
            "row_count_unchanged": _no_credit(),
            "not_null_constraint": _no_credit(),
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
            "customers_schema": _no_credit(),
            "deduplicated_customers": _no_credit(),
            "orders_customer_id_column": _no_credit(),
            "customer_links_complete": _no_credit(),
            "foreign_key_integrity": _no_credit(),
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
            "numeric_column_type": _no_credit(),
            "all_rows_numeric": _no_credit(),
            "expected_sum": _no_credit(),
            "summary_view_intact": _no_credit(),
            "audit_log_intact": _no_credit(),
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
