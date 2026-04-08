from __future__ import annotations

import threading

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.environment import SchemaEvolutionEnv
from app.models import Observation, ResetRequest, StepRequest, StepResult, TaskSpec
from app.tasks import TASKS

app = FastAPI(title="SchemaEvolutionEnv", version="1.0.0")
env = SchemaEvolutionEnv()
env_lock = threading.Lock()


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> list[TaskSpec]:
    return [task.spec for task in TASKS.values()]


@app.post("/reset")
def reset(body: ResetRequest | None = None) -> Observation:
    try:
        with env_lock:
            task_id = body.task_id if body is not None else ResetRequest().task_id
            return env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(body: StepRequest) -> StepResult:
    with env_lock:
        if env.done:
            raise HTTPException(status_code=400, detail="Episode finished. Call /reset first.")
        return env.step(body.action)


@app.get("/state")
def state() -> Observation:
    with env_lock:
        return env.state()


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)
