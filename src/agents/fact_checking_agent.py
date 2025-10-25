"""Fact-checking & self-reflective agent factory.

This module provides ``create_fact_checking_agent`` which builds on the
regular :class:`Agent` abstraction adding two function tools:

* ``search`` – retrieve factual evidence snippets (default: Wikipedia API)
* ``self_check`` – critique a draft answer and suggest improvements / missing evidence

Design goals:
1. Encourage *deliberate, multi-step reasoning* (ReAct style) before final answer
2. Gather *structured evidence* tied to each claim; cite sources explicitly
3. Optional *self critique* loop to reduce hallucinations
4. Keep implementation pluggable so you can swap in enterprise KB / RAG backend

The agent instructions enforce an interaction format (Thought / Action / Observation) but
do not require the model to literally echo the format in the final answer—only during its
internal reasoning turns. The final answer must be a clean, user-facing response with
inline citations like [1], [2] and a consolidated Sources section.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List
import re
from html import unescape

import requests

from .agent import Agent, ModelSettings
from .tool import function_tool
from .models.openai_chatcompletions import OpenAIChatCompletionsModel
from .tracing import set_tracing_disabled, set_tracing_export_api_key
from openai import AsyncOpenAI
import os


Evidence = Dict[str, str]
SearchResults = List[Evidence]


def _default_wikipedia_search(query: str, top_k: int = 3, lan: str = 'zh') -> SearchResults:
    """Lightweight Wikipedia search returning top_k snippets.

    Each evidence dict contains: {"title", "snippet", "source"} where ``source`` is a
    canonical string we can cite. We only fetch search listing (no page extracts) to
    keep latency & complexity low. Production systems may want page content, caching,
    disambiguation, language options, and robust error handling.
    """

    try:
        resp = requests.get(
            f"https://{lan}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "format": "json",
                "utf8": 1,
                "srlimit": top_k,
                "srsearch": query,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("query", {}).get("search", [])
    except Exception:  # pragma: no cover - network errors shouldn't fail tests
        return []

    results: SearchResults = []
    tag_re = re.compile(r"<[^>]+>")
    for row in data:
        title = row.get("title", "")
        raw_snippet = row.get("snippet", "")
        # 去除 HTML / XML 标签与转义实体
        cleaned = unescape(tag_re.sub("", raw_snippet))
        source = f"Wikipedia:{title}" if title else "Wikipedia"
        results.append({"title": title, "snippet": cleaned, "source": source})
    return results


def create_fact_checking_agent(
    max_turns: int = 5,
    search_fn: Callable[[str, int], SearchResults] | None = None,
    self_check_fn: Callable[[str], str] | None = None,
) -> Agent[Dict[str, Any]]:
    """Create a fact-checking agent with self-reflection.

    Args:
        search_fn: Function performing retrieval. Input: (query, top_k) -> list of evidence dicts.
            Provide custom implementation to integrate enterprise KB / vector store.
        self_check_fn: Function performing critique. Input: draft_answer -> str returning
            critique suggestions (plain text). If None, tool echoes the draft.

    Returns:
        Configured :class:`Agent` ready to run.
    """

    _search_impl = search_fn or _default_wikipedia_search


    @function_tool()
    def search(query: str, top_k: int = 3) -> SearchResults:
        """在wikipedia上，为特定的缺失事实检索客观证据。

        参数:
            query (str): 聚焦且精炼的检索查询语句，必须是中文，尽量包含 1~3 个关键信息单元（如实体名、年份、政策/制度关键词），避免整句长问题。
            top_k (int, 默认 3): 返回的最大证据片段数；建议 1-5 之间。更大可能增加噪声与延迟。

        返回:
            list[dict]: 证据列表，每个元素包含:
                - title: 资料标题
                - snippet: 片段（可能含 HTML 强调标签，需要后处理）
                - source: 可在引用列表中展示的来源标识

        使用建议:
            - 将复杂问题拆成多个子查询，多次调用获得覆盖面。
            - 若 snippet 暗示新的关键实体或年份，可继续发起针对性补充搜索。
            - 避免一次查询塞入过多关键词，降低召回质量。
        """

        return _search_impl(query, top_k)


    @function_tool()
    def self_check(draft: str) -> str:  # pragma: no cover
        """批判性审阅草稿答案。

            如果需要改进，请以精炼清单形式给出：
            - 缺失的事实（附建议的检索查询）
            - 缺乏支持/支持薄弱的论断（标注引用编号）
            - 存在歧义/需要保留措辞的地方
            如果草稿可靠，直接返回“OK”。
        """

        if self_check_fn is None:
            return draft
        return self_check_fn(draft)

    instructions = (
        "你是一名一丝不苟的事实核查研究助理。\n"
        "在回答之前，请遵循‘思考（Thought）-> 行动（Action）’的迭代模式来收集证据。\n"
        "可用的行动/工具：\n"
        "1) search(query, top_k=3)：输入query必须是中文，检索精炼的证据片段，避免太长。请使用多个聚焦的查询。\n"
        "2) self_check(draft)：在定稿前可选的自我审阅/批注。\n\n"
        "流程规则：\n"
        "- 首先将问题分解为子问题（年份、政策变化、实体）。\n"
        "- 对于每个非平凡的声明，确保至少有一个搜索结果支持它。\n"
        "- 在收集到足够的证据后，草拟答案并使用内联数字引用 [1]、[2]……来标记实际使用的搜索的时间顺序（而不是任意的）。\n"
        "- 如果仍然存在不确定性，可以选择调用 self_check；然后进行改进。\n"
        "- 最终答案必须：（a）直接回答，（b）逻辑分组，（c）引用来源，（d）包括“来源：”列表，将 [n] 映射到来源标题。\n"
        "- 如果证据不足，明确说明局限性。\n"
        "避免幻觉；绝不要捏造未见的日期、名称或数字。\n"
        "仅将打磨后的最终答案返回给用户（不包含思考/行动行）。\n"
        f"你最多只能执行{max_turns}步行动，你必须在{max_turns}步内返回最终答案。"
    )
    

    # 延迟/可选模型配置：如未提供 API Key，将仍然返回 Agent（后续可外部赋值 agent.model）
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

    return Agent(
        name="fact_checking_agent",
        model=model,  # 若为 None，运行前外部可自行指定 FakeModel / 其它模型
        instructions=instructions,
        tools=[search, self_check],
    )


__all__ = ["create_fact_checking_agent", "Evidence", "SearchResults"]
