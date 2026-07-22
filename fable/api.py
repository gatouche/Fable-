"""Local REST API — lets any app talk to Fable."""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fable.core import FableRunner

app = FastAPI(title="Fable API", version="0.1.0")
runner = FableRunner(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class TaskRequest(BaseModel):
    task: str
    dry_run: bool = False


@app.post("/run")
def run_task(req: TaskRequest):
    try:
        if req.dry_run:
            steps = runner.plan(req.task)
            return {"steps": steps}
        results = runner.run(req.task)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
