from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from agent.memory import add_to_memory, build_memory_context
from agent.models import AgentState, Task, TaskStatus

logger = logging.getLogger(__name__)


async def execute_all_tasks(
    state: AgentState,
    llm,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> dict[int, str]:
    """
    Execute all tasks in dependency order (topological DAG).

    For each ready task:
    1. Build prompt with context from completed dependencies
    2. Call LLM with exponential backoff retry
    3. On failure: skip non-critical, fail critical
    4. Store result in state

    Returns state.results dict.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional business writer. Write well-structured "
         "content for the section described.\n\n"
         "{memory_context}"
         "Context from previous sections:\n{dependency_context}"),
        ("human",
         "Task: {task_title}\n{task_description}\n\n"
         "Use markdown formatting with headings, bullet points, and tables "
         "where appropriate. Be specific and detailed."),
    ])

    while True:
        task = _next_ready_task(state.tasks)
        if task is None:
            break

        logger.info("Executing task %d: %s", task.id, task.title)
        task.status = TaskStatus.RUNNING

        dep_context = _build_dependency_context(task, state)
        mem_context = build_memory_context(state.model_dump())

        success = False
        for attempt in range(max_retries):
            mem_context_attempt = (
                f"{mem_context}\n"
                if mem_context else ""
            )
            if attempt > 0:
                mem_context_attempt += (
                    f"Note: Previous attempt {attempt} failed. "
                    f"Error: {task.error}\n"
                    f"Please produce a different, correct response this time."
                )

            try:
                result = await llm.ainvoke(
                    prompt.invoke({
                        "memory_context": mem_context_attempt,
                        "dependency_context": dep_context,
                        "task_title": task.title,
                        "task_description": task.description,
                    })
                )
                task.result = result.content
                task.status = TaskStatus.COMPLETED
                state.results[task.id] = result.content
                add_to_memory(
                    state.model_dump(),
                    "assistant",
                    f"[Task {task.id}: {task.title}]\n{result.content[:500]}",
                )
                success = True
                logger.info("Task %d completed successfully", task.id)
                break
            except Exception as e:
                task.error = str(e)
                task.retry_count += 1
                logger.warning(
                    "Task %d attempt %d/%d failed: %s",
                    task.id, attempt + 1, max_retries, e,
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        if not success:
            if _is_critical(task):
                task.status = TaskStatus.FAILED
                state.error = f"Critical task {task.id} ({task.title}) failed: {task.error}"
                logger.error(state.error)
                break
            else:
                task.status = TaskStatus.SKIPPED
                task.result = f"[{task.title} could not be generated]"
                state.results[task.id] = task.result
                logger.warning("Task %d skipped after %d failures", task.id, max_retries)

    return state.results


def _next_ready_task(tasks: list[Task]) -> Optional[Task]:
    """Find the first pending task whose dependencies are all completed."""
    completed_ids = {t.id for t in tasks if t.status == TaskStatus.COMPLETED}
    for task in tasks:
        if task.status != TaskStatus.PENDING:
            continue
        deps = task.depends_on or []
        if all(d in completed_ids for d in deps):
            return task
    return None


def _build_dependency_context(task: Task, state: AgentState) -> str:
    """Build context string from completed dependency tasks."""
    parts = []
    for dep_id in (task.depends_on or []):
        dep_result = state.results.get(dep_id, "")
        dep_title = ""
        for t in state.tasks:
            if t.id == dep_id:
                dep_title = t.title
                break
        if dep_result:
            parts.append(f"--- {dep_title} ---\n{dep_result[:2000]}")
    return "\n\n".join(parts)


def _is_critical(task: Task) -> bool:
    """Determine if a task is critical (fails the whole agent)."""
    keywords = ["summary", "objective", "scope", "conclusion", "executive"]
    title_lower = task.title.lower()
    return any(kw in title_lower for kw in keywords)
