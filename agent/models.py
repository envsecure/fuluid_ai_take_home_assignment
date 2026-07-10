from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    id: int
    title: str
    description: str
    depends_on: Optional[list[int]] = None
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    retry_count: int = 0
    error: str = ""


class AgentState(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: str
    tasks: list[Task] = []
    results: dict[int, str] = {}
    status: str = "pending"
    error: str = ""
    conversation: list[dict] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    docx_path: str = ""
