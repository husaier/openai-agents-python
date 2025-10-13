from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from agents import Agent, ItemHelpers, Runner, TResponseInputItem, WebSearchTool, trace

"""Example of self-reflection and fact checking using two cooperating agents.

The research agent answers a question using web search. The critic agent reviews the
answer for unsupported claims. If the critic finds issues, it provides feedback and the
researcher revises the answer. The loop continues until the critic approves the response.
"""

researcher = Agent(
    name="researcher",
    instructions=(
        "You answer the user's question by searching the web for facts."
        "Think step by step and cite your sources."
        "If feedback is provided, use it to revise your answer."
    ),
    tools=[WebSearchTool()],
)


@dataclass
class CriticFeedback:
    verdict: Literal["pass", "revise"]
    feedback: str


critic = Agent[None](
    name="critic",
    instructions=(
        "You check the researcher's answer for unsupported or inaccurate claims."
        "If everything is well supported, respond with verdict 'pass'."
        "Otherwise, respond with verdict 'revise' and explain what needs fixing."
    ),
    output_type=CriticFeedback,
)


async def main() -> None:
    question = input("Question: ")
    conversation: list[TResponseInputItem] = [{"role": "user", "content": question}]

    latest_answer = ""
    with trace("fact_checking_loop"):
        while True:
            research_result = await Runner.run(researcher, conversation)
            conversation = research_result.to_input_list()
            latest_answer = ItemHelpers.text_message_outputs(research_result.new_items)
            print("Researcher answer:\n", latest_answer)

            critic_result = await Runner.run(critic, conversation)
            feedback = critic_result.final_output
            print(f"Critic verdict: {feedback.verdict}")

            if feedback.verdict == "pass":
                break

            print("Revising with critic feedback\n")
            conversation.append({"role": "user", "content": f"Feedback: {feedback.feedback}"})

    print("Final answer:\n", latest_answer)


if __name__ == "__main__":
    asyncio.run(main())
