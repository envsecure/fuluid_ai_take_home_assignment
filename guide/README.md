# LangChain & LangGraph Guide

A beginner-friendly guide with simple code examples and Mermaid diagrams. No prior LangChain knowledge needed.

## Contents

| File | Topic | What You'll Learn |
|---|---|---|
| `01_langchain_basics.md` | LangChain fundamentals | What is an LLM wrapper? Prompt chains vs raw API calls |
| `02_prompts_and_chains.md` | Prompts + Chains | ChatPromptTemplate, pipable chains, Runnable interface |
| `03_output_parsers.md` | Structured output | PydanticOutputParser, getting JSON from LLM reliably |
| `04_langgraph_intro.md` | LangGraph basics | StateGraph, nodes, edges, state management |
| `05_agent_with_langgraph.md` | Full agent | Building the autonomous agent with planning + execution |

## How to Use

1. Read in order — each file builds on the previous
2. Each file has mermaid diagrams explaining the flow
3. Code examples are standalone (pip install first)
4. Try running the examples yourself

## Quick Install

```bash
pip install langchain-core langchain-google-genai langgraph pydantic
```
