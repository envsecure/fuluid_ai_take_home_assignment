# Video Script — Autonomous AI Agent System Walkthrough

**Duration:** ~10 minutes
**Audience:** HR (technical background)
**Tone:** Casual conversation, show code on screen

---

## INTRO (30 seconds)

> Hey, I built an autonomous AI agent system. You give it a natural language request like "write me a market analysis report" and it automatically plans the work, executes each step using an LLM, and generates a polished Word document. Let me walk you through how it works.

**Show:** project folder structure

```
agent/          → core logic (planner, executor, orchestrator)
docbuilder/     → markdown to .docx conversion
llm/            → LLM client wrapper
main.py         → FastAPI server
```

---

## 1. SYSTEM OVERVIEW — How it all connects (1 min)

> The system has three phases: Plan, Execute, Build. Think of it like a project manager — it breaks your request into tasks, runs them in order, and compiles the results into a document.

**Show:** `agent/orchestrator.py` lines 26-57

```python
async def run(self, task_id: str, request: str) -> None:
    # Phase 1: Plan
    state.status = "planning"
    tasks = decompose_request(request, self._llm)
    state.tasks = tasks
    state.status = "executing"

    # Phase 2: Execute
    results = await execute_all_tasks(state, self._llm)
    state.results = results

    # Phase 3: Build document
    state.status = "building"
    docx_path = await self._build_document(state)
    state.status = "completed"
```

> This is the orchestrator. It coordinates the whole pipeline. Notice the status tracking — planning, executing, building, completed. This lets the frontend poll progress in real time.

**Key point:** The entire pipeline runs as a background task. The API returns immediately with a task ID, and the client polls for status.

---

## 2. API LAYER — How requests come in (1 min)

**Show:** `main.py` lines 42-62

```python
@app.post("/agent", status_code=202)
async def submit_request(body: RequestInput, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(orchestrator.run, task_id, body.request)
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Agent is working on your request. Poll GET /agent/status/{task_id} for updates.",
    }
```

> It's a FastAPI server. You POST a request, it gives you a task ID back immediately — 202 Accepted, not 200. Then you poll the status endpoint to see progress, and when it's done, you download the .docx file.

**Show:** the three endpoints quickly

| Endpoint | Purpose |
|---|---|
| `POST /agent` | Submit request, get task_id |
| `GET /agent/status/{task_id}` | Poll progress |
| `GET /agent/download/{task_id}` | Download document |

---

## 3. PLANNING PHASE — LLM decomposes the request (1.5 min)

**Show:** `agent/planner.py` lines 39-78

> This is where it gets interesting. The planner takes your natural language request and uses an LLM to break it into 3-8 concrete tasks with dependencies.

```python
def decompose_request(request: str, llm, max_retries: int = 2) -> list[Task]:
    parser = PydanticOutputParser(pydantic_object=PlanSchema)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert project planner. Break the user's request into "
         "3-8 concrete, sequential tasks...\n\n{format_instructions}"),
        ("human", "{request}"),
    ])

    chain = prompt | llm | parser
```

> See this chain? `prompt | llm | parser` — that's LangChain's LCEL (LangChain Expression Language). It's a pipeline: format the prompt, send to LLM, parse the output into a Pydantic model.

**Show:** the Pydantic schema

```python
class TaskSchema(BaseModel):
    id: int
    title: str
    description: str
    depends_on: Optional[list[int]] = None

class PlanSchema(BaseModel):
    tasks: list[TaskSchema]
```

> The LLM returns JSON that matches this schema. PydanticOutputParser validates it automatically. If the LLM returns garbage, we catch it and retry.

**Show:** the fallback

```python
FALLBACK_TASKS = [
    Task(id=1, title="Executive Summary", ...),
    Task(id=2, title="Background & Context", ...),
    Task(id=3, title="Main Content", depends_on=[1, 2]),
    Task(id=4, title="Supporting Data", depends_on=[3]),
    Task(id=5, title="Conclusion & Next Steps", depends_on=[4]),
]
```

> If all LLM attempts fail, we fall back to a hardcoded 5-task template. This is a tradeoff — we'd rather produce a generic document than fail completely.

---

## 4. EXECUTION PHASE — Running tasks in dependency order (2 min)

**Show:** `agent/executor.py` lines 15-50

> Now the executor runs each task. But not in order — it respects dependencies. Task 3 can't start until tasks 1 and 2 are done.

```python
async def execute_all_tasks(state, llm, max_retries=3, base_delay=2.0):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional business writer...\n\n"
         "{memory_context}"
         "Context from previous sections:\n{dependency_context}"),
        ("human",
         "Task: {task_title}\n{task_description}\n\n"
         "Use markdown formatting..."),
    ])

    while True:
        task = _next_ready_task(state.tasks)
        if task is None:
            break
```

> `_next_ready_task` finds the first task whose dependencies are all completed. It's like a topological sort on a DAG (Directed Acyclic Graph).

**Show:** the retry logic

```python
for attempt in range(max_retries):
    try:
        result = await llm.ainvoke(prompt.invoke({...}))
        task.result = result.content
        task.status = TaskStatus.COMPLETED
        state.results[task.id] = result.content
        success = True
        break
    except Exception as e:
        task.error = str(e)
        task.retry_count += 1
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)  # exponential backoff
            await asyncio.sleep(delay)
```

> Each task has up to 3 retries with exponential backoff — 2 seconds, 4 seconds, 8 seconds. This handles rate limits and transient errors.

**Show:** critical vs non-critical tasks

```python
def _is_critical(task: Task) -> bool:
    keywords = ["summary", "objective", "scope", "conclusion", "executive"]
    title_lower = task.title.lower()
    return any(kw in title_lower for kw in keywords)
```

> Not all tasks are equal. If a task with "summary" or "conclusion" in the title fails, we stop the whole pipeline — it's critical. But if "Supporting Data" fails, we just skip it and continue. The document still gets generated, just without that section.

**Tradeoff to mention:** We could let the user configure which tasks are critical, but for simplicity we use keyword matching. It's not perfect but works for 90% of cases.

---

## 5. MEMORY SYSTEM — Context between tasks (1 min)

**Show:** `agent/memory.py` full file

```python
SLIDING_WINDOW_SIZE = 20

def add_to_memory(state, role, content):
    state["conversation"].append({
        "role": role,
        "content": content,
        "timestamp": str(datetime.now()),
    })
    if len(state["conversation"]) > SLIDING_WINDOW_SIZE:
        state["conversation"] = state["conversation"][-SLIDING_WINDOW_SIZE:]

def build_memory_context(state, max_turns=6):
    turns = state.get("conversation", [])
    recent = turns[-max_turns:]
    lines = ["## Prior conversation context:"]
    for t in recent:
        label = "User" if t["role"] == "user" else "Assistant"
        truncated = t["content"][:300]
        lines.append(f"  {label}: {truncated}")
    return "\n".join(lines) + "\n"
```

> Each task gets context from two sources: dependency results (what previous tasks produced) and conversation memory (recent history). The sliding window keeps only the last 20 turns so we don't blow up the token limit. Each turn is truncated to 300 characters.

**Tradeoff:** Sliding window is simple but loses old context. A vector store would be better for large projects, but overkill here.

---

## 6. DOCUMENT GENERATION — Markdown to .docx (1.5 min)

**Show:** `docbuilder/builder.py` — key functions

> The last phase converts markdown results into a styled Word document.

```python
def generate_docx(state: AgentState, output_path: str) -> None:
    doc = Document()
    _set_default_style(doc)
    _add_cover_page(doc, state.request)
    _add_tasks_content(doc, state)
    _add_footer(doc)
    doc.save(output_path)
```

**Show:** the markdown parser

```python
def _render_markdown(doc, markdown_text):
    lines = markdown_text.split("\n")
    for line in lines:
        if stripped.startswith("# "):
            doc.add_heading(text, level=1)
        elif stripped.startswith("## "):
            doc.add_heading(text, level=2)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            p = doc.add_paragraph(style="List Bullet")
            _add_run_with_bold(p, text)
        elif re.match(r"^(\d+)\.\s+(.*)", stripped):
            p = doc.add_paragraph(style="List Number")
            _add_run_with_bold(p, text)
        elif line.strip().startswith("|") and line.strip().endswith("|"):
            # table detection and rendering
```

> It's a simplified markdown parser. It handles headings, bullet lists, numbered lists, bold text, and tables. Not a full markdown parser — just enough for what the LLM typically produces.

**Show:** table styling

```python
def _shade_cell(cell, color):
    shading = cell._element.get_or_add_tcPr()
    shading_elem = shading.makeelement(qn("w:shd"), {})
    shading_elem.set(qn("w:val"), "clear")
    shading_elem.set(qn("w:color"), "auto")
    shading_elem.set(qn("w:fill"), str(color))
    shading.append(shading_elem)
```

> Tables get styled with a dark header row and alternating row shading. This uses low-level XML manipulation because python-docx doesn't expose shading directly.

---

## 7. FAULT TOLERANCE — How it handles failures (1 min)

> Let me quickly summarize how the system handles failures at every level.

| Layer | Failure | Handling |
|---|---|---|
| Planning | LLM returns bad JSON | Retry 2x, then fallback template |
| Execution | LLM call fails | Retry 3x with exponential backoff |
| Execution | Non-critical task fails | Skip, continue with remaining |
| Execution | Critical task fails | Stop entire pipeline |
| Document | Output dir missing | Auto-create with `os.makedirs` |
| LLM | No API key set | Fall back to MockLLM for testing |

**Show:** mock client briefly

```python
class MockLLM:
    def invoke(self, messages):
        return AIMessage(content=json.dumps({
            "tasks": [
                {"id": 1, "title": "Executive Summary", ...},
                ...
            ]
        }))
```

> The MockLLM returns canned responses so you can test the full pipeline without spending API credits.

---

## 8. TRADEOFFS & DESIGN DECISIONS (1 min)

> A few design decisions worth mentioning:

**1. In-memory state vs database**
> State is stored in a Python dict. Simple, fast, no setup. But it's lost on restart and doesn't scale. For production you'd use Redis or a database.

**2. Custom markdown parser vs library**
> I wrote a simplified parser instead of using something like `markdown-it`. Reason: we only need a subset (headings, lists, tables, bold), and a full parser would pull in dependencies and might produce unexpected HTML elements.

**3. Synchronous doc generation**
> `generate_docx` runs synchronously in an async pipeline. python-docx is sync-only. It's fine because doc generation is fast (milliseconds). Not worth wrapping in a thread pool.

**4. LangChain for LLM calls**
> LangChain gives us prompt templates, output parsing, and provider abstraction. Switching from Gemini to OpenRouter was just changing the client — the rest of the code didn't move. That's the value.

**5. Retry with backoff**
> Exponential backoff prevents hammering the API. 3 retries is a balance between reliability and latency. We could make it configurable but 3 works for most cases.

---

## CONCLUSION (30 seconds)

> So that's the system. You give it a request, it plans the work, executes each step with context from previous steps, handles failures gracefully, and produces a styled Word document. The whole thing is async, fault-tolerant, and the LLM provider is swappable. Thanks for watching.

---

## QUICK REFERENCE — Files to show on screen

| File | What to show | When |
|---|---|---|
| `main.py` | POST endpoint + status polling | API layer section |
| `agent/orchestrator.py` | `run()` method, 3 phases | Overview |
| `agent/planner.py` | `decompose_request()`, Pydantic schema | Planning section |
| `agent/executor.py` | `execute_all_tasks()`, retry loop | Execution section |
| `agent/memory.py` | Full file (it's short) | Memory section |
| `docbuilder/builder.py` | `generate_docx()`, `_render_markdown()` | Doc generation |
| `llm/client.py` | `OpenRouterClient` class | Tradeoffs |
| `agent/models.py` | `Task`, `AgentState` Pydantic models | Anytime |
