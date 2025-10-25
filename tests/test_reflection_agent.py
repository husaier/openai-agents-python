import json

import pytest

from agents import Runner, create_reflection_agent

from .fake_model import FakeModel
from .test_responses import get_final_output_message, get_function_tool_call, get_text_message


@pytest.mark.asyncio
async def test_reflection_agent_search_and_self_check():
    search_queries: list[str] = []
    self_check_inputs: list[str] = []

    def search_fn(q: str) -> str:
        search_queries.append(q)
        return "draft answer"

    def check_fn(ans: str) -> str:
        self_check_inputs.append(ans)
        return "ok"

    model = FakeModel()
    agent = create_reflection_agent(search_fn, check_fn)
    agent.model = model

    model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("search", json.dumps({"query": "benefits"}))],
            [
                get_text_message("draft answer"),
                get_function_tool_call("self_check", json.dumps({"draft": "draft answer"})),
            ],
            [get_final_output_message("final answer")],
        ]
    )

    result = await Runner.run(agent, input="question")

    assert result.final_output == "final answer"
    assert search_queries == ["benefits"]
    assert self_check_inputs == ["draft answer"]
