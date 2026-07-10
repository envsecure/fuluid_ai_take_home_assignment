from __future__ import annotations

import json
from typing import Any
from langchain_core.messages import AIMessage


class MockLLM:
    """
    Returns canned responses when no API key is available.

    Used for testing the planner and executor without Gemini costs.
    """

    def invoke(self, messages: Any) -> AIMessage:
        return self._respond()

    async def ainvoke(self, messages: Any) -> AIMessage:
        return self._respond()

    def _respond(self) -> AIMessage:
        content = json.dumps({
            "tasks": [
                {"id": 1, "title": "Executive Summary",
                 "description": "Write a brief executive summary covering the purpose and key outcomes.",
                 "depends_on": None},
                {"id": 2, "title": "Background Research",
                 "description": "Gather context and background information on the topic.",
                 "depends_on": [1]},
                {"id": 3, "title": "Main Content",
                 "description": "Write the detailed body content with analysis.",
                 "depends_on": [1, 2]},
                {"id": 4, "title": "Supporting Data",
                 "description": "Include relevant metrics, examples, and supporting evidence.",
                 "depends_on": [3]},
                {"id": 5, "title": "Conclusion & Recommendations",
                 "description": "Summarize findings and provide actionable recommendations.",
                 "depends_on": [4]},
            ]
        })
        return AIMessage(content=content)
