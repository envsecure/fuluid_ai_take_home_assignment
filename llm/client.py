from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Wrapper around ChatGoogleGenerativeAI with error handling.

    Uses GOOGLE_API_KEY from environment. Falls back to a mock
    if no API key is set (for testing).

    The ChatGoogleGenerativeAI import is lazy — only loaded when
    an API key is actually present, so MockLLM works without
    installing langchain-google-genai.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.7):
        self.model = model or os.environ.get("LLM_MODEL", "gemini-1.5-flash")
        self.temperature = temperature
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:

            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                from llm.mock_client import MockLLM
                logger.warning("No GOOGLE_API_KEY set. Using MockLLM.")
                return MockLLM()

            from langchain_google_genai import ChatGoogleGenerativeAI

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
