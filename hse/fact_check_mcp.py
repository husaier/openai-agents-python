"""示例：使用 create_fact_checking_agent 执行带检索+自我反思的事实核查问答。

运行方式：
    PYTHONPATH=src python hse/fact_check.py

可自定义：
1. search_impl: 接企业内部检索 / 向量库 / Wikipedia 等
2. self_check_impl: 对草稿答案进行自我批评（检查遗漏、无支撑断言、引用格式）

Agent 将采用 Thought -> Action(search/self_check) -> Observation 循环（模型内部自行生成），
直到准备好最终答案（包含引用 Sources 列表）。
"""

from __future__ import annotations

import asyncio
import os
from agents.mcp import MCPServer, MCPServerStreamableHttp
from openai import AsyncOpenAI
import asyncio
import time

from agents import Runner, trace, ModelSettings, OpenAIChatCompletionsModel, Agent, set_tracing_export_api_key, set_tracing_disabled


def build_agent(mcp_server: MCPServer, max_turns: int = 5):
    instructions = (
        "你是一名一丝不苟的事实核查研究助理。\n"
        "在回答之前，请遵循‘思考（Thought）-> 行动（Action）’的迭代模式来收集证据。\n"
        "流程规则：\n"
        "- 首先将问题分解为子问题（年份、政策变化、实体）。\n"
        "- 对于每个非平凡的声明，确保至少有一个搜索结果支持它。\n"
        "- 在收集到足够的证据后，草拟答案并使用内联数字引用 [1]、[2]……来标记实际使用的搜索的时间顺序（而不是任意的）。\n"
        "- 如果仍然存在不确定性，可以选择调用 self_check；然后进行改进。\n"
        "- 最终答案必须：（a）直接回答，（b）逻辑分组，（c）引用来源，（d）包括“来源：”列表，将 [n] 映射到来源标题。\n"
        "- 如果证据不足，明确说明局限性。\n"
        "避免幻觉；绝不要捏造未见的日期、名称或数字。\n"
        "仅将打磨后的最终答案返回给用户（不包含思考/行动）。\n"
        f"你最多只能执行{max_turns}步行动，你必须在{max_turns}步内返回最终答案。"
    )
    
    model: OpenAIChatCompletionsModel | None = None
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:  # 只有在提供密钥时才配置默认模型，避免测试或本地无密钥时报循环导入
        base_url = os.getenv("OPENAI_BASE_URL") or None
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        # 可选关闭 tracing（避免演示场景上传）
        tracing_api_key = os.environ["TRACING_API_KEY"]
        set_tracing_export_api_key(tracing_api_key)
        set_tracing_disabled(disabled=False)
        # 模型名称可按需调整，示例使用 responses/chat 兼容模型占位
        model_name = os.getenv("FACT_CHECK_MODEL", "gpt-5-nano")
        model = OpenAIChatCompletionsModel(model=model_name, openai_client=client)

    agent = Agent(
        name=f"Fact Checking Agent",
        model=model,
        instructions=instructions,
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(tool_choice="required"),
    )

    return agent


QUESTIONS = [
    "美元自1792年《铸币法案》通过至1946年国际货币基金组织标准期间的关键含金量变化历程有哪些？",
    "严家淦在台湾省政府财政厅长和财政部部长任期内实施了哪些主要财政改革措施？",
    "欧元区采用单一货币有哪些主要经济好处？",
    "中央银行的主要职责包括哪些具体方面？",
    "1998年中国人民银行撤销省级分行后设立的九家跨省区分行名称及其管辖范围是什么？",
]


async def run_demo():
    MAX_TURNS = 10
    mcp_url = os.getenv("MCP_URL")
    api_key = os.getenv("MCP_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    async with MCPServerStreamableHttp(
        name="cn bing搜索",
        params={
            "url": mcp_url,
            "headers": headers,
        }
    ) as server:
        tool_list = await server.list_tools()
        print(f"TOOLS: {[tool.name for tool in tool_list]}")
        
        agent = build_agent(server, max_turns=MAX_TURNS)
        
        for q in QUESTIONS:
            print("\n=== 问题 ===")
            print(q)
            with trace(workflow_name="Fact Checking"):
                result = Runner.run(agent, input=q, max_turns=MAX_TURNS)
                for res in result.trace_events:
                    print(f"[{res}]")
            print("\n--- 最终答案 ---")
            print(result.final_output)


if __name__ == "__main__":
    asyncio.run(run_demo())
