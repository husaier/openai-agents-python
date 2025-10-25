#!/bin/bash

export OPENAI_BASE_URL="https://api.360.cn/v1"
export OPENAI_API_KEY=""

export TRACING_API_KEY=""

export OPENAI_API_KEY=""

uv run python hse/test.py

export FACT_CHECK_MODEL="gpt-4o-mini"

export MODELSCOPE_API_KEY=""

export OPENAI_BASE_URL=""
export OPENAI_API_KEY="EMPTY"
export FACT_CHECK_MODEL="Qwen3-32B"
PYTHONPATH=src python hse/fact_check.py

export MCP_URL=""
export MCP_API_KEY=""
PYTHONPATH=src python hse/fact_check_mcp.py
