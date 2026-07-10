# Implementation Plan

## Phase 1 — Project Scaffolding (30 min)

- [ ] Create directory structure: `agent/`, `llm/`, `docbuilder/`, `outputs/`
- [ ] Write `requirements.txt` with all dependencies
- [ ] Write `config.py` — Pydantic `BaseSettings` with `LLM_PROVIDER`, API keys, `OUTPUT_DIR`
- [ ] Write `.env.example`
- [ ] Write stub `main.py` with health-check endpoint only

## Phase 2 — LLM Client Abstraction (45 min)

- [ ] Write `llm/base.py` — abstract `LLMClient` class with `generate(system, prompt, max_tokens) -> str`
- [ ] Write `llm/groq_client.py` — implement using `groq` library, handle 429 rate limits with backoff
- [ ] Write `llm/gemini_client.py` — implement using `google-genai`, handle free-tier quota errors
- [ ] Write `llm/ollama_client.py` — implement using `ollama` library, handle connection errors
- [ ] Write factory `LLMClient.create(provider: str)` in `llm/__init__.py`
- [ ] Test each client with a simple prompt via a throwaway script

## Phase 3 — Agent Core (1.5 hr)

- [ ] Write `agent/models.py`:
  - `Task` dataclass: `id, title, description, depends_on, status, result`
  - `AgentState` dataclass: `task_id, request, tasks, results, status, error`
- [ ] Write `agent/planner.py`:
  - Function `decompose_request(request: str, llm: LLMClient) -> list[Task]`
  - Prompt: "Break this request into 3-8 JSON tasks with id, title, description, depends_on"
  - Parse JSON response, validate schema
  - Error handling: retry → fallback to hardcoded 5-task template
- [ ] Write `agent/executor.py`:
  - Function `execute_tasks(tasks, llm, on_task_complete callback) -> dict`
  - Topological DAG execution — loop until all tasks done
  - Per-task prompt with context from completed dependencies
  - Retry logic (3 attempts per task)
- [ ] Write `agent/orchestrator.py`:
  - Class `Orchestrator` — holds in-memory `dict[task_id, AgentState]`
  - Method `run(task_id, request)` — calls planner → executor → doc builder
  - Method `get_status(task_id) -> AgentState`
  - Background: use `asyncio.to_thread` or `asyncio.create_task` to avoid blocking

## Phase 4 — Document Builder (1 hr)

- [ ] Write `docbuilder/styles.py` — constants for fonts, colors, margins
- [ ] Write `docbuilder/markdown_parser.py`:
  - Parse markdown headings, bold, bullet lists, numbered lists, tables
  - Return a list of structured blocks (paragraph, heading, table, list)
- [ ] Write `docbuilder/builder.py`:
  - Function `build_doc(result: dict, title: str) -> Path`
  - Use `python-docx` to create document
  - Add cover page (title, subtitle, date)
  - Iterate over parsed blocks and add to document
  - Apply styles, table shading, page numbers in footer
  - Save to `outputs/{task_id}.docx`

## Phase 5 — FastAPI Routes (30 min)

- [ ] Write `POST /agent`:
  - Accept `{"request": "..."}`, validate non-empty
  - Generate `uuid` task_id
  - Enqueue background task (orchestrator.run)
  - Return `202 {"task_id": ..., "status": "processing"}`
- [ ] Write `GET /agent/status/{task_id}`:
  - Look up in orchestrator state
  - Return status, task breakdown, download_url if completed
- [ ] Write `GET /agent/download/{task_id}`:
  - Return `StreamingResponse` with `.docx` file
  - Set `Content-Type` and `Content-Disposition` headers
  - Optionally delete file after download

## Phase 6 — Error Handling & Polish (30 min)

- [ ] Global exception handler in FastAPI — return `500 {"error": "..."}`
- [ ] Timeout on LLM calls (30s default)
- [ ] Empty request → `400 {"error": "Please provide a valid request"}`
- [ ] File cleanup: on startup, delete files in `outputs/` older than 1 hour
- [ ] Add `startup` event to create `outputs/` directory if missing

## Phase 7 — Testing (45 min)

- [ ] Unit tests: `test_planner.py` — mock LLM, verify task list parsing
- [ ] Unit tests: `test_markdown_parser.py` — all markdown block types
- [ ] Unit tests: `test_docbuilder.py` — verify .docx is valid zip
- [ ] Integration test: `test_agent.py` — mock LLM responses, full pipeline
- [ ] API tests: `test_api.py` — httpx AsyncClient, status codes, polling flow
- [ ] Run: `pytest -v --cov`

## Phase 8 — Documentation & Handoff (15 min)

- [ ] Write `README.md` with setup, run, example curl commands
- [ ] Add comments to `orchestrator.py` and `executor.py`
- [ ] Final review: verify all imports, edge cases, type hints

---

## Total: ~6 hours

| Phase | Time | Deliverable |
|---|---|---|
| 1. Scaffolding | 30 min | `main.py`, `config.py`, `requirements.txt`, dirs |
| 2. LLM Clients | 45 min | 3 implementations + factory |
| 3. Agent Core | 1.5 hr | Planner, Executor, Orchestrator |
| 4. Doc Builder | 1 hr | Markdown parser + python-docx builder |
| 5. API Routes | 30 min | 3 endpoints |
| 6. Error Handling | 30 min | Edge cases, timeouts, cleanup |
| 7. Testing | 45 min | Unit + integration + API tests |
| 8. Polish | 15 min | README, comments, final review |
