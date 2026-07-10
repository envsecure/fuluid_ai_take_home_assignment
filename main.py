from __future__ import annotations

import logging
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.orchestrator import Orchestrator
from llm import get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Autonomous AI Agent",
    description="Accepts a natural language request, plans tasks, executes them, and returns a .docx document.",
    version="1.0.0",
)

llm_client = get_llm()
orchestrator = Orchestrator(llm=llm_client.llm)


class RequestInput(BaseModel):
    request: str


class StatusResponse(BaseModel):
    task_id: str
    status: str
    tasks: list[dict] = []
    download_url: str = ""
    error: str = ""


@app.post("/agent", status_code=202)
async def submit_request(body: RequestInput, background_tasks: BackgroundTasks):
    """
    Submit a natural language request.

    Returns immediately with a task_id. The agent runs in the background.
    Poll GET /agent/status/{task_id} for completion.
    """
    if not body.request.strip():
        raise HTTPException(status_code=400, detail="Request cannot be empty.")

    task_id = str(uuid.uuid4())
    background_tasks.add_task(orchestrator.run, task_id, body.request)

    logger.info("Accepted request %s: %s", task_id, body.request[:80])

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Agent is working on your request. Poll GET /agent/status/{task_id} for updates.",
    }


@app.get("/agent/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """Poll the status of an agent run."""
    state = orchestrator.get_state(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    tasks_list = [
        {
            "id": t.id,
            "title": t.title,
            "status": t.status.value,
            "retry_count": t.retry_count,
        }
        for t in state.tasks
    ]

    return StatusResponse(
        task_id=task_id,
        status=state.status,
        tasks=tasks_list,
        download_url=f"/agent/download/{task_id}" if state.status == "completed" else "",
        error=state.error,
    )


@app.get("/agent/download/{task_id}")
async def download_docx(task_id: str):
    """Download the generated .docx document."""
    state = orchestrator.get_state(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    if state.status != "completed":
        raise HTTPException(status_code=400, detail="Document is not ready yet.")
    if not os.path.exists(state.docx_path):
        raise HTTPException(status_code=500, detail="Document file not found on disk.")

    filename = f"{state.request[:50].replace(' ', '_')}.docx"
    return FileResponse(
        path=state.docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
