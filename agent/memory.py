from __future__ import annotations

SLIDING_WINDOW_SIZE = 20


def add_to_memory(state: dict, role: str, content: str) -> None:
    """Append a turn to conversation memory, keeping a sliding window."""
    if "conversation" not in state:
        state["conversation"] = []
    state["conversation"].append({
        "role": role,
        "content": content,
        "timestamp": str(__import__("datetime").datetime.now()),
    })
    if len(state["conversation"]) > SLIDING_WINDOW_SIZE:
        state["conversation"] = state["conversation"][-SLIDING_WINDOW_SIZE:]


def build_memory_context(state: dict, max_turns: int = 6) -> str:
    """
    Build a context string from recent conversation history.
    Used when constructing prompts so the LLM knows what happened before.
    """
    turns = state.get("conversation", [])
    if not turns:
        return ""
    recent = turns[-max_turns:]
    lines = ["## Prior conversation context:"]
    for t in recent:
        label = "User" if t["role"] == "user" else "Assistant"
        truncated = t["content"][:300]
        lines.append(f"  {label}: {truncated}")
    return "\n".join(lines) + "\n"
