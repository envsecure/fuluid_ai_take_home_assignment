from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent.memory import add_to_memory
from agent.models import Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskSchema(BaseModel):
    id: int = Field(description="Unique task ID, starting from 1")
    title: str = Field(description="Task title, under 10 words")
    description: str = Field(description="What to produce, 1-2 sentences")
    depends_on: Optional[list[int]] = Field(
        None, description="IDs of prerequisite tasks. null for root tasks."
    )


class PlanSchema(BaseModel):
    tasks: list[TaskSchema]


FALLBACK_TASKS = [
    Task(id=1, title="Executive Summary", description="Write a concise executive summary covering the key points of the request."),
    Task(id=2, title="Background & Context", description="Research and describe the background context needed to understand the topic."),
    Task(id=3, title="Main Content", description="Write the main body content with detailed analysis and discussion.", depends_on=[1, 2]),
    Task(id=4, title="Supporting Data", description="Include relevant data, metrics, or examples to support the main content.", depends_on=[3]),
    Task(id=5, title="Conclusion & Next Steps", description="Summarize findings and list actionable next steps.", depends_on=[4]),
]


def decompose_request(request: str, llm, max_retries: int = 2) -> list[Task]:
    """
    Decompose a natural language request into structured tasks.

    Uses Gemini via LangChain's PydanticOutputParser. Falls back to a
    hardcoded 5-task template if all LLM attempts fail.
    """
    parser = PydanticOutputParser(pydantic_object=PlanSchema)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert project planner. Break the user's request into "
         "3-8 concrete, sequential tasks. Each task must be independently "
         "executable.\n\n{format_instructions}\n\n"
         "Rules:\n"
         "- id must be a unique integer starting from 1\n"
         "- depends_on must reference existing task IDs, or be null for root tasks\n"
         "- Return ONLY valid JSON, no markdown fences, no explanation"),
        ("human", "{request}"),
    ])

    chain = prompt | llm | parser

    for attempt in range(max_retries + 1):
        try:
            plan = chain.invoke({
                "request": request,
                "format_instructions": parser.get_format_instructions(),
            })
            tasks = [Task(**t.model_dump()) for t in plan.tasks]
            _validate_dag(tasks)
            logger.info("Successfully decomposed request into %d tasks", len(tasks))
            return tasks
        except Exception as e:
            logger.warning("Planner attempt %d/%d failed: %s", attempt + 1, max_retries + 1, e)
            if attempt < max_retries:
                continue

    logger.warning("All LLM planner attempts failed. Using fallback template.")
    return _get_fallback_tasks(request)


def _validate_dag(tasks: list[Task]) -> None:
    """Validate dependency graph: no cycles, all references are valid."""
    task_ids = {t.id for t in tasks}
    for t in tasks:
        if t.depends_on:
            for dep_id in t.depends_on:
                if dep_id not in task_ids:
                    t.depends_on.remove(dep_id)


def _get_fallback_tasks(request: str) -> list[Task]:
    """Return annotated fallback tasks."""
    return FALLBACK_TASKS
