"""Utilities to build an agent with search and self reflection.

This module exposes a helper to construct an :class:`Agent` that can query a
knowledge base and optionally critique its own draft answers. The agent uses two
function tools:

* ``search`` – fetches information from an external knowledge source.
* ``self_check`` – allows the model to review a draft answer before finalizing it.

The helper accepts callables implementing these behaviours, making it easy to
plug in custom search backends or critique logic. By default the search tool
performs a simple Wikipedia lookup using ``requests``.
"""

from __future__ import annotations

from typing import Callable, Dict, Any

import requests

from .agent import Agent
from .tool import function_tool


def _default_wikipedia_search(query: str) -> str:
    """Return a short snippet from Wikipedia for ``query``.

    The implementation is intentionally lightweight and only grabs the first
    search result snippet. For production systems you may want a more robust
    retriever with error handling and caching.
    """

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "format": "json",
            "utf8": 1,
            "srlimit": 1,
            "srsearch": query,
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    search = data.get("query", {}).get("search", [])
    if not search:
        return ""
    return search[0].get("snippet", "")


def create_reflection_agent(
    search_fn: Callable[[str], str] | None = None,
    self_check_fn: Callable[[str], str] | None = None,
) -> Agent[Dict[str, Any]]:
    """Create an agent equipped with fact search and self reflection.

    Args:
        search_fn: Callable used by the ``search`` tool. Defaults to a simple
            Wikipedia search.
        self_check_fn: Callable used by the ``self_check`` tool. If ``None`` the
            tool simply echoes its input.

    Returns:
        An :class:`Agent` instance ready to run.
    """

    actual_search = search_fn or _default_wikipedia_search

    @function_tool
    def search(query: str) -> str:  # pragma: no cover - exercised in tests
        """Look up information related to ``query``."""

        return actual_search(query)

    @function_tool
    def self_check(draft: str) -> str:  # pragma: no cover - exercised in tests
        """Critique or verify the ``draft`` answer."""

        if self_check_fn is None:
            return draft
        return self_check_fn(draft)

    instructions = (
        "You are a helpful assistant that answers questions carefully. "
        "Use the `search` tool to gather facts before responding. "
        "After formulating a draft, you may call `self_check` to review it. "
        "Only provide the final answer once you are satisfied."
    )

    return Agent(name="reflection_agent", instructions=instructions, tools=[search, self_check])
