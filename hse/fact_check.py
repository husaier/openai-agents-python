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

from typing import List, Dict

from agents import Runner, create_fact_checking_agent


# ====== 可替换：简单 Wikipedia 检索（此处直接用默认实现, 仅做包装便于后续替换） ======
def search_impl(query: str, top_k: int):  # noqa: D401 - 简洁说明
	"""自定义检索逻辑占位。当前直接返回空列表交给默认工具实现（示例）。"""
	# 返回 None / [] 让模型继续尝试其他查询；若需要接企业 API，可在此返回结构：
	# [ {"title": "条目标题", "snippet": "证据片段", "source": "SourceID或URL"}, ... ]
	# 这里返回 [] 使得我们改用 agent 内部默认 wikipedia 实现（通过不传 search_fn 更简单）。
	raise NotImplementedError("示例中直接使用内置默认; 如需自定义请将 create_fact_checking_agent(search_impl, ...) 传入")


# ====== 自我检视函数：简易策略 ======
def self_check_impl(draft: str) -> str:
	issues: List[str] = []
	if "Sources:" not in draft:
		issues.append("缺少 Sources 源列表")
	if "[1]" not in draft:
		issues.append("似乎没有任何引用标号 [n]")
	if len(draft) < 50:
		issues.append("答案过短，可能缺少分段逻辑与充分论证")
	return "OK" if not issues else "改进建议:\n- " + "\n- ".join(issues)


def build_agent(max_turns: int = 5):
	# 使用默认 wikipedia 搜索: 不传 search_impl; 仅提供自检函数
	agent = create_fact_checking_agent(max_turns=max_turns, self_check_fn=self_check_impl)
	# 若想固定模型： agent.model = "gpt-4o"  (或其他已配置别名)
	return agent


QUESTIONS = [
	"美元自1792年《铸币法案》通过至1946年国际货币基金组织标准期间的关键含金量变化历程有哪些？",
	"严家淦在台湾省政府财政厅长和财政部部长任期内实施了哪些主要财政改革措施？",
	"欧元区采用单一货币有哪些主要经济好处？",
	"中央银行的主要职责包括哪些具体方面？",
	"1998年中国人民银行撤销省级分行后设立的九家跨省区分行名称及其管辖范围是什么？",
]


async def run_demo():  # pragma: no cover - 手动运行示例
	MAX_TURNS = 5

	agent = build_agent(max_turns=MAX_TURNS)
	for q in QUESTIONS:
		print("\n=== 问题 ===")
		print(q)
		result = await Runner.run(agent, input=q, max_turns=MAX_TURNS)
		print("\n--- 最终答案 ---")
		print(result.final_output)


if __name__ == "__main__":  # pragma: no cover
	import asyncio

	asyncio.run(run_demo())

