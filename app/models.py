from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ResetRequest(BaseModel):
    task_id: str = "task1_add_column"


class Action(BaseModel):
    type: Literal[
        "inspect_schema",
        "sample_data",
        "run_migration",
        "validate_constraints",
        "rollback",
        "submit_final",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    action: Action


class ColumnInfo(BaseModel):
    name: str
    type: str
    notnull: bool
    default_value: str | None
    primary_key: bool


class SchemaInfo(BaseModel):
    tables: dict[str, list[ColumnInfo]]


class Observation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str
    step: int
    schema_info: SchemaInfo = Field(alias="schema")
    last_action_result: str | None
    cumulative_reward: float
    done: bool
    goal: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    goal: str
