from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    Wrapper around ChatOpenAI configured for OpenRouter.

    Uses OPENROUTER_API_KEY from environment. Falls back to a mock
    if no API key is set (for testing).
    """

    def __init__(self, model: str | None = None, temperature: float = 0.7):
        self.model = model or os.environ.get("LLM_MODEL", "tencent/hy3:free")
        self.temperature = temperature
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:

            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                from llm.mock_client import MockLLM
                logger.warning("No OPENROUTER_API_KEY set. Using MockLLM.")
                return MockLLM()

            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=self.temperature,
                max_tokens=16000,
            )
        return self._llm

    def invoke(self, messages):
        return self.llm.invoke(messages)

    async def ainvoke(self, messages):
        return await self.llm.ainvoke(messages)


def get_llm() -> OpenRouterClient:
    """Factory: returns a configured OpenRouterClient."""
    return OpenRouterClient()
