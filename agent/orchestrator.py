from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from agent.executor import execute_all_tasks
from agent.memory import add_to_memory
from agent.models import AgentState
from agent.planner import decompose_request

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Manages the full agent lifecycle: plan → execute → build.

    Stores AgentState in-memory keyed by task_id.
    """

    def __init__(self, llm):
        self._llm = llm
        self._states: dict[str, AgentState] = {}

    async def run(self, task_id: str, request: str) -> None:
        """Full agent pipeline. Runs as a background task."""
        logger.info("Starting agent run %s: %s", task_id, request[:80])

        state = AgentState(task_id=task_id, request=request)
        self._states[task_id] = state

        try:
            # Phase 1: Plan
            state.status = "planning"
            add_to_memory(state.model_dump(), "user", request)

            tasks = decompose_request(request, self._llm)
            state.tasks = tasks
            state.status = "executing"
            state.updated_at = datetime.now()

            # Phase 2: Execute
            results = await execute_all_tasks(state, self._llm)
            state.results = results

            if state.error:
                state.status = "failed"
                logger.error("Agent run %s failed: %s", task_id, state.error)
                return

            # Phase 3: Build document
            state.status = "building"
            state.updated_at = datetime.now()
            docx_path = await self._build_document(state)
            state.docx_path = docx_path
            state.status = "completed"

        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            logger.exception("Agent run %s failed unexpectedly", task_id)
        finally:
            state.updated_at = datetime.now()

    def get_state(self, task_id: str) -> AgentState | None:
        return self._states.get(task_id)

    async def _build_document(self, state: AgentState) -> str:
        """
        Generate the final .docx document from all task results.
        Uses LLM to polish the compilation, then writes via python-docx.
        """
        import os
        from docbuilder.builder import generate_docx

        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/{state.task_id}.docx"
        generate_docx(state, path)
        return path
