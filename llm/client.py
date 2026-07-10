from __future__ import annotations

import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Wrapper around ChatGoogleGenerativeAI with error handling.

    Uses GOOGLE_API_KEY from environment. Falls back to a mock
    if no API key is set (for testing).
    """

    def __init__(self, model: str = "gemini-1.5-flash", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self._llm: ChatGoogleGenerativeAI | None = None

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:

            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                from llm.mock_client import MockLLM
                logger.warning("No GOOGLE_API_KEY set. Using MockLLM.")
                return MockLLM()

            self._llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                google_api_key=api_key,
            )
        return self._llm

    def invoke(self, messages):
        return self.llm.invoke(messages)

    async def ainvoke(self, messages):
        return await self.llm.ainvoke(messages)


def get_llm() -> GeminiClient:
    """Factory: returns a configured GeminiClient."""
    return GeminiClient()
