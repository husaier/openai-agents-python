from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import os


BASE_URL = os.getenv("OPENAI_BASE_URL") or ""
API_KEY = os.getenv("OPENAI_API_KEY") or ""


client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)

MODEL_NAME = "openai/gpt-5-nano"
agent = Agent(
    name="Assistant", 
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client), 
    instructions="You are a helpful assistant"
)

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
