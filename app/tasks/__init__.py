from __future__ import annotations

from dataclasses import dataclass

from app.models import TaskSpec
from app.tasks import task1_add_column, task2_split_table, task3_type_change


@dataclass(frozen=True)
class TaskDefinition:
    spec: TaskSpec
    setup_sql: str


TASKS: dict[str, TaskDefinition] = {
    task1_add_column.TASK_SPEC.id: TaskDefinition(
        spec=task1_add_column.TASK_SPEC,
        setup_sql=task1_add_column.SETUP_SQL,
    ),
    task2_split_table.TASK_SPEC.id: TaskDefinition(
        spec=task2_split_table.TASK_SPEC,
        setup_sql=task2_split_table.SETUP_SQL,
    ),
    task3_type_change.TASK_SPEC.id: TaskDefinition(
        spec=task3_type_change.TASK_SPEC,
        setup_sql=task3_type_change.SETUP_SQL,
    ),
}
