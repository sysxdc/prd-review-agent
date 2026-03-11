"""
agent.py（Reflection + LangSmith 升级版）

改造重点：
1. 加入 reflection 节点：对 assistant 输出做质量验证，不合格则重试（最多2次）
2. 接入 LangSmith：3行代码实现全链路追踪，可视化每次 token 消耗和调用链
3. 评审完成后自动写入历史库，供下次 RAG 召回
4. state 中新增 retry_count，防止无限循环
"""

import os
import json
import sqlite3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import tools_condition, ToolNode

from state import PRDState
from tools import check_completeness, extract_user_stories, identify_risks
from rag_store import save_review_to_history

load_dotenv()

# ── LangSmith 接入（3行，ENV里配好就自动追踪）──────────────────────────────
# 在 .env 里加入：
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=your_langsmith_key
#   LANGCHAIN_PROJECT=prd-review-agent
# 不配置也不影响运行，只是没有追踪面板
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
# ────────────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

tools = [check_completeness, extract_user_stories, identify_risks]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """你是一个专业的产品需求评审助手，拥有丰富的互联网产品经验。

分析PRD文档时，你必须：
1. 调用 check_completeness 工具检查完整性（工具内部已集成行业规范知识库）
2. 调用 extract_user_stories 工具提取用户故事
3. 调用 identify_risks 工具识别技术风险
4. 综合三个工具的结构化结果，输出最终评审报告

最终报告必须按以下格式输出：

## 📊 完整性评分：X/100
> 评分依据：（来自工具分析结果）

## ❌ 缺失字段
- **字段名**：缺失原因和改进建议

## 📖 用户故事（P0优先）
- **[P0]** 作为[用户]，我希望[功能]，以便[价值]
  - AC：验收标准

## ⚠️ 风险识别
- **[高危]** 风险描述 → 应对建议

## 🐛 需求缺陷
- 缺陷类型：位置和改进建议

## 💡 综合建议
- 优先级最高的3条可执行建议

回答追问时正常对话即可，不需要固定格式。请用中文回答。"""


# ── 节点1：assistant（主分析节点）─────────────────────────────────────────
def assistant(state: PRDState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ── 节点2：reflection（质量验证节点）──────────────────────────────────────
# Reflection Judge Prompt：用 LLM 评估自己的输出是否合格
REFLECTION_PROMPT = """你是一个严格的质量审核员。请检查以下PRD评审报告是否合格。

评审报告：
{report}

检查标准：
1. 是否包含完整性评分（0-100的数字）
2. 是否列出了至少1个缺失字段（如果PRD真的很完整则可以没有）
3. 是否包含至少1条用户故事
4. 是否包含风险识别
5. 内容是否具体，而非泛泛而谈（如"需要改进"这类没有信息量的表述不合格）

输出JSON（不要输出其他内容）：
{{
  "passed": true/false,
  "score": <0-100>,
  "issues": ["不合格原因1", "不合格原因2"],
  "suggestion": "给assistant的改进提示"
}}"""


def reflection(state: PRDState):
    """
    质量验证节点：
    - 找到最近一条 AIMessage，提取报告内容
    - 用 LLM-as-Judge 评分
    - 不合格且重试次数<2：返回 retry 信号，让 assistant 重新生成
    - 合格或达到重试上限：通过，触发历史入库
    """
    retry_count = state.get("retry_count", 0)

    # 找最后一条 AI 消息（tool call 除外）
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            last_ai_message = msg
            break

    if not last_ai_message:
        # 还没有正式报告（可能还在 tool 调用中），直接放行
        return {"current_step": "waiting_for_report"}

    # 跳过追问场景：如果最后的 AI 消息很短，说明是对话回复，不需要 reflection
    if len(last_ai_message.content) < 200:
        return {"current_step": "chat_response"}

    # 执行 Reflection 评估
    judge_prompt = REFLECTION_PROMPT.format(report=last_ai_message.content)
    judge_response = llm.invoke([HumanMessage(content=judge_prompt)])

    try:
        raw = judge_response.content.strip().removeprefix("```json").removesuffix("```").strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        # 解析失败，默认通过（避免无限循环）
        result = {"passed": True, "score": 70, "issues": [], "suggestion": ""}

    passed = result.get("passed", True)
    issues = result.get("issues", [])
    suggestion = result.get("suggestion", "")

    print(f"[Reflection] passed={passed}, score={result.get('score')}, retry={retry_count}")

    # 不合格且还有重试次数
    if not passed and retry_count < 2:
        retry_msg = HumanMessage(
            content=f"你的评审报告需要改进，请重新生成。\n问题：{'; '.join(issues)}\n建议：{suggestion}"
        )
        return {
            "messages": [retry_msg],
            "retry_count": retry_count + 1,
            "current_step": "retrying",
        }

    # 合格或达到重试上限 → 写入历史库
    _save_to_history(state, last_ai_message.content)
    return {
        "retry_count": 0,
        "current_step": "completed",
    }


def _save_to_history(state: PRDState, report_content: str):
    """从 state 中提取结果，写入历史评审库"""
    prd_content = state.get("prd_content", "")
    if not prd_content:
        return
    prd_summary = prd_content[:150]  # 取前150字作为向量化摘要

    review_result = {
        "score": state.get("completeness_score", 0),
        "missing_fields": state.get("missing_fields", []),
        "risks": state.get("risks", []),
        "user_stories": state.get("user_stories", []),
        "conclusion": report_content[:300],
    }
    save_review_to_history(prd_summary, review_result)


# ── 路由函数：reflection 后决定继续还是结束 ───────────────────────────────
def should_retry(state: PRDState):
    step = state.get("current_step", "")
    if step == "retrying":
        return "assistant"   # 重新生成
    return END               # 完成或对话回复


# ── 构建图 ────────────────────────────────────────────────────────────────
def build_graph():
    os.makedirs("state_db", exist_ok=True)
    conn = sqlite3.connect("state_db/prd_agent.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    builder = StateGraph(PRDState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("reflection", reflection)

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)  # assistant → tools or END
    builder.add_edge("tools", "assistant")                        # tools → assistant
    # assistant 完成 tool 调用、生成最终报告后 → reflection
    # 注意：tools_condition 会在没有 tool_call 时路由到 END
    # 我们需要在没有 tool_call 时路由到 reflection，而不是 END
    # 因此覆盖 tools_condition 的 END 路径
    builder.add_conditional_edges(
        "assistant",
        _route_assistant,
        {"tools": "tools", "reflection": "reflection", END: END},
    )
    builder.add_conditional_edges("reflection", should_retry, {"assistant": "assistant", END: END})

    return builder.compile(checkpointer=memory)


def _route_assistant(state: PRDState):
    """
    覆盖默认的 tools_condition：
    - 有 tool_calls → 去 tools
    - 无 tool_calls 且消息够长（是报告） → 去 reflection
    - 无 tool_calls 且消息很短（是追问回复） → END
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 报告足够长才做 reflection
    if len(last_message.content) >= 200:
        return "reflection"
    return END


graph = build_graph()