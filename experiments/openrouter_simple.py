"""
Standalone OpenRouter example using LangChain.

Usage:
    set OPENROUTER_API_KEY=your_key_here
    python experiments/openrouter_simple.py

Get a key at: https://openrouter.ai/keys

Requires: pip install langchain-openai
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "openai/gpt-3.5-turbo"


def build_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    return ChatOpenAI(
        model=MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
    )


if __name__ == "__main__":
    llm = build_llm()

    prompt = "What is the capital of France? Answer in one sentence."
    print(f"Prompt: {prompt}")

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"Answer: {response.content}")
