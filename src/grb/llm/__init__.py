"""LLM access layer for the benchmark (Claude Headless via the ``claude`` CLI)."""

from grb.llm.headless import call_claude

__all__ = ["call_claude"]
