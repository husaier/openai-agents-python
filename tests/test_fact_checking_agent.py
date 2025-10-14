import json

import pytest

from agents import Runner, create_fact_checking_agent

from .fake_model import FakeModel
from .test_responses import get_final_output_message, get_function_tool_call, get_text_message


@pytest.mark.asyncio
async def test_fact_checking_agent_search_and_self_check():
    search_log = []
    check_log = []

    def search_impl(q: str, top_k: int):
        search_log.append((q, top_k))
        # Return list of evidence dicts
        return [
            {"title": "Item1", "snippet": "Snippet1", "source": "S1"},
            {"title": "Item2", "snippet": "Snippet2", "source": "S2"},
        ]

    def check_impl(draft: str):
        check_log.append(draft)
        return "OK"

    model = FakeModel()
    agent = create_fact_checking_agent(search_fn=search_impl, self_check_fn=check_impl)
    agent.model = model

    # Simulate LLM plan: first search, then produce draft & self check, then final answer
    model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("search", json.dumps({"query": "美元 含金量 1792 1946", "top_k": 3}))],
            [
                get_text_message("Draft answer with citations [1]"),
                get_function_tool_call("self_check", json.dumps({"draft": "Draft answer with citations [1]"})),
            ],
            [get_final_output_message("Final polished answer")],
        ]
    )

    result = await Runner.run(agent, input="测试问题")

    assert result.final_output == "Final polished answer"
    assert search_log and search_log[0][0].startswith("美元")
    assert check_log and check_log[0].startswith("Draft answer")
