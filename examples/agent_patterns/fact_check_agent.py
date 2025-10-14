from __future__ import annotations

"""Example showing how to build a strict fact-checking agent template."""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from agents import Agent, AgentOutputSchema, Runner, WebSearchTool


class EvidenceEntry(BaseModel):
    """Evidence collected from a search result snippet or title."""

    ref: str = Field(
        description="Search reference identifier (for example '1.3').",
    )
    span: list[int] = Field(
        description="Unicode character index range [start, end) within the cited snippet/title.",
        min_length=2,
        max_length=2,
    )
    quote: str = Field(
        description="Exact quoted text (no more than 120 Chinese characters) from the snippet/title.",
    )
    stance: Literal["support", "refute"] = Field(
        description="Whether the evidence supports or refutes the claim.",
    )


class FactCheckFinding(BaseModel):
    """A single fact-check finding covering one verifiable claim."""

    hallu_span: list[int] = Field(
        description="Unicode index range [start, end) of the problematic text in the response.",
        min_length=2,
        max_length=2,
    )
    hallu_text: str = Field(
        description="Exact text excerpt from the response that is being evaluated.",
    )
    verdict: Literal["contradiction", "unsupported", "controversial"] = Field(
        description="Overall judgement for the claim.",
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Impact assessment of the issue.",
    )
    reason: str = Field(
        description="Short Chinese explanation referencing the evidence identifiers.",
    )
    evidence: list[EvidenceEntry] = Field(
        description="List of evidence snippets backing the judgement.",
    )


class FactCheckSearchTool(WebSearchTool):
    """Rename the hosted web search tool so the agent calls `search(...)`."""

    @property
    def name(self) -> str:  # noqa: D401 - simple property override
        return "search"


FACT_CHECK_PROMPT_INSTRUCTIONS = """# 角色
你是一名**严格、可复核的事实核查员**，并且**必须主动使用“search”工具**检索证据。你的唯一职责：依据你用“search”获得的**搜索结果摘要（snippets）**对 *response* 中的**可核查事实**逐条判定，并**只输出 JSON**。

# 工具与编号规范（只有 search）
* 你仅能使用：`search(query, recency?, domains?)`

  * `query`：查询语句；可多轮、多条。
  * `recency`：可选，按需要限制时效（如最近30/90天）。
  * `domains`：可选，限制域名（如官方、学术）。
* **禁止**使用任何打开网页、页面内查找或 PDF 截图等功能；你的一切证据**只能来自搜索结果返回的“snippet 文本”**（以及必要时的“title 文本”，见下条）。
* **证据来源**：优先从 `snippet` 截取。若 `snippet` 不含关键信息而 `title` 含有，可使用 `title`；**不可**从 URL 或域名名称推断事实。
* **参考编号**：第 *k* 次搜索的第 *i* 条结果记为 **`#k.i`**（例如：第一次搜索的第3条 ⇒ `#1.3`）。

  * 在输出的 `evidence[].ref` 中使用**字符串**形式 `"k.i"`（如 `"1.3"`），避免小数歧义。

# 任务

对 *response* 抽取**最小粒度**可核查事实（实体、时间、数量、排序、因果、直引、链接对象等），通过多轮 `search` 为每条事实寻找**明确支持或否定**的**可引用文本子串**（来自搜索结果的 `snippet` 或必要时的 `title`），据此判定结论。

# 判定标准（四选一）

* **supported（支持）**：至少一条参考**明确支持**，且无参考与之矛盾。
* **contradiction（矛盾）**：至少一条参考**直接否定/相反**，且无任何支持性参考。
* **unsupported（无法证实）**：所有参考的可见 `snippet/title` 均未提供支持或否定。
* **controversial（相互冲突）**：同时存在**至少一条支持**与**至少一条否定**证据。

> **仅**当事实被判为 `contradiction` / `unsupported` / `controversial` 时纳入输出；纯 `supported` 的事实不输出。若 *response* 全部为 `supported`，输出空数组 `[]`。

# 检索与取证流程（必须执行，过程不外显）

1. **枚举事实**：将 *response* 拆成最小核查单元（精确到关键实体/数字/日期/结论词）。
2. **生成查询**：为每条事实设计 1–3 条高召回查询（中英/同义词/别名），必要时加时效限定（如 `recency=90`）与权威域（`domains`）。
3. **发起搜索**：调用 `search`（可多轮），优先权威来源（官方/学术/主流媒体/数据库），兼顾多样性。
4. **抽取证据**：仅从返回的 `snippet`（必要时 `title`）截取**最小充分**证据子串；**不得**凭常识或 URL 推断。
5. **判定**：若既有支持又有否定 ⇒ `controversial`；仅有否定 ⇒ `contradiction`；均无 ⇒ `unsupported`；仅有支持 ⇒ 不输出。
6. **终止条件**：每条事实至多 3 轮搜索；若仍无可引用子串，判 `unsupported` 并在 `reason` 标注已覆盖的参考范围（如 `#1.*—#3.*`）。

# span 标注（response 侧）

* 使用 **Unicode 字符索引**，从 0 起，左闭右开 `[start, end)`；`\n` 计 1 个字符；不做归一化。
* `hallu_span` **精确覆盖**最小问题片段（关键实体、数字或结论短语）。
* 多条按 `start` 升序；必要时将重叠事实拆分。

# 证据定位（references 侧，精确到“哪里”）

* `evidence` 为数组；元素结构：
  `{"ref": "1.3", "span": [s, e], "quote": "与 span 完全一致（≤120字）", "stance": "support|refute"}`
* `span` 相对于**对应搜索结果文本**本身：

  * 若取自 `snippet`：以 `snippet` 字符串为基准标注。
  * 若取自 `title`：以 `title` 字符串为基准标注，并在 `reason` 明示“证据来自 title”。
* `contradiction`：至少 1 条 `refute`；可附加 `support` 作背景。
* `controversial`：至少 1 条 `support` 与 1 条 `refute`。
* `unsupported`：`evidence` 置为 `[]`，并在 `reason` 写明“参考未覆盖（已检查参考 #1.*—#M.*；仅 snippet/title 可见，无法佐证）”。

# 严重性（severity）

* `"low"`：轻微偏差/措辞/边缘细节（日期≤1天、数字误差≤5%）。
* `"medium"`：对部分结论有实质影响（日期≤7天、数字误差≤10%等）。
* `"high"`：关键事实错误（核心主体/结论/主要时间地点错误，或数字误差>10%）。
* `"critical"`：颠覆主要结论或涉医疗/安全/金融/法律等高风险的重大误导。
* 对 `unsupported`：按影响评估（细节型 `"low"`，一般性 `"medium"`，核心/高风险 `"high"`/`"critical"`）。

# 输出格式（**必须严格一致；仅 JSON；布尔为 true/false**）

当存在任一 `contradiction` / `unsupported` / `controversial` 时，输出为**数组**，每个元素形如：

```
[
  {
    "hallu_span": [start, end],
    "hallu_text": "从 response 中截取的该事实原文",
    "verdict": "contradiction" | "unsupported" | "controversial",
    "severity": "low" | "medium" | "high" | "critical",
    "reason": "用中文精炼说明问题与参考编号及冲突类型（数值/日期/主体/地点/因果等）。若为未覆盖，写明已检查范围（如参考 #1.*—#3.*），并注明“仅 snippet/title 可见”。",
    "evidence": [
      {"ref": "1.3", "span": [s1, e1], "quote": "与 snippet 中该区间完全一致…", "stance": "refute"},
      {"ref": "1.1", "span": [s2, e2], "quote": "与 snippet 中该区间完全一致…", "stance": "support"}
    ]
  }
]
```

若全部为 `supported`，输出 `[]`（空数组）。

# 质量与合规约束

* **必须检索**；**禁止**引用未检索来源或模型记忆。
* **证据仅限**搜索结果可见文本（`snippet` 优先，`title` 兜底）；**不得**以域名/URL/机构名替代内容证据。
* 每条事实 1–4 条证据；`quote` ≤120字且与 `span` 完全一致。
* 涉隐私/付费墙/不可见内容：若 `snippet/title` 无法给出可引用子串，则判 `unsupported` 并在 `reason` 说明原因与检索覆盖范围。

# 输入模板

```
====================  输入  ====================
question:
{question}

response:
{response}
==============================================
```

用户消息会完全按照上述模板提供 `question` 与 `response` 字段；请在最终 JSON 中仅给出核查结果。
"""


def format_fact_check_prompt(*, question: str, response: str) -> str:
    """Fill the shared prompt template with the question and model response."""

    return (
        "====================  输入  ====================\n"
        "question:\n"
        f"{question}\n\n"
        "response:\n"
        f"{response}\n"
        "=============================================="
    )


def build_fact_check_agent() -> Agent[list[FactCheckFinding]]:
    """Construct the fact-checking agent with strict JSON output."""

    return Agent(
        name="FactCheckAgent",
        instructions=FACT_CHECK_PROMPT_INSTRUCTIONS,
        tools=[FactCheckSearchTool(search_context_size="high")],
        output_type=AgentOutputSchema(list[FactCheckFinding]),
    )


async def main() -> None:
    agent = build_fact_check_agent()

    prompt = format_fact_check_prompt(
        question="请核查回答中的陈述是否准确。",
        response=(
            "乔·拜登在2020年赢得美国总统大选，并于2021年1月20日就职。"
            "他现在仍担任副总统，而美国首都位于纽约。"
        ),
    )

    result = await Runner.run(agent, prompt)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
