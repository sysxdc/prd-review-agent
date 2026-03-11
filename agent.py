"""
agent.py — 并行工具调用 + 流式输出 + Reflection + LangSmith
"""

import os
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import PRDState
from tools import check_completeness, extract_user_stories, identify_risks
from rag_store import save_review_to_history

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    streaming=True,
)

tools = [check_completeness, extract_user_stories, identify_risks]
llm_with_tools = llm.bind_tools(tools)

TOOL_MAP = {
    "check_completeness": check_completeness,
    "extract_user_stories": extract_user_stories,
    "identify_risks": identify_risks,
}

SYSTEM_PROMPT = """你是一个专业的产品需求评审助手，拥有丰富的互联网产品经验。

分析PRD文档时，你必须同时调用以下三个工具（它们会并行执行）：
1. check_completeness —— 检查完整性，工具内部已集成行业规范知识库
2. extract_user_stories —— 提取并规范化用户故事
3. identify_risks —— 识别技术风险和需求缺陷

综合三个工具的结构化结果，输出最终评审报告，格式如下：

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


# ── 节点1：assistant ──────────────────────────────────────────────────────
def assistant(state: PRDState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ── 节点2：parallel_tools ─────────────────────────────────────────────────
def parallel_tools(state: PRDState):
    """并发执行所有工具，响应时间从 sum(各工具) 降为 max(各工具)"""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    if not tool_calls:
        return {"messages": []}

    results = {}
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        future_to_call = {
            executor.submit(_run_tool, call["name"], call["args"]): call
            for call in tool_calls
        }
        for future in as_completed(future_to_call):
            call = future_to_call[future]
            try:
                output = future.result()
            except Exception as e:
                output = f"工具执行出错：{str(e)}"
            results[call["id"]] = output

    tool_messages = [
        ToolMessage(
            content=str(results.get(call["id"], "执行失败")),
            tool_call_id=call["id"],
            name=call["name"],
        )
        for call in tool_calls
    ]
    return {"messages": tool_messages}


def _run_tool(tool_name: str, args: dict) -> str:
    tool_fn = TOOL_MAP.get(tool_name)
    if not tool_fn:
        return f"未知工具：{tool_name}"
    return tool_fn.invoke(args)


# ── 节点3：reflection ─────────────────────────────────────────────────────
REFLECTION_PROMPT = """你是一个严格但公正的质量审核员。请检查以下PRD评审报告是否合格。

评审报告：
{report}

检查标准：
1. 是否包含完整性评分（0-100的数字）
2. 是否列出了缺失字段（低质量PRD应有多个，高质量PRD可以较少）
3. 是否包含至少1条格式规范的用户故事
4. 是否包含风险识别内容
5. 内容是否具体，而非泛泛而谈

评分参考（判断报告中的分数是否合理）：
- 高质量PRD（模块齐全、有AC、有风险应对）：55-90分
- 中等质量PRD（核心模块存在但不完整）：30-65分
- 低质量PRD（大量模块缺失）：0-40分

输出JSON（不要输出其他内容）：
{{
  "passed": true/false,
  "score": <0-100>,
  "issues": ["不合格原因，没有则为空列表"],
  "suggestion": "给assistant的改进提示，通过则留空"
}}"""

_judge_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    streaming=False,
)


def reflection(state: PRDState):
    retry_count = state.get("retry_count", 0)

    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            last_ai_message = msg
            break

    if not last_ai_message or len(last_ai_message.content) < 200:
        return {"current_step": "chat_response"}

    judge_response = _judge_llm.invoke(
        [HumanMessage(content=REFLECTION_PROMPT.format(report=last_ai_message.content))]
    )

    try:
        raw = judge_response.content.strip().removeprefix("```json").removesuffix("```").strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"passed": True, "score": 70, "issues": [], "suggestion": ""}

    passed = result.get("passed", True)
    issues = result.get("issues", [])
    suggestion = result.get("suggestion", "")
    print(f"[Reflection] passed={passed}, score={result.get('score')}, retry={retry_count}")

    if not passed and retry_count < 2:
        return {
            "messages": [HumanMessage(
                content=f"评审报告需要改进，请重新生成。\n问题：{'; '.join(issues)}\n建议：{suggestion}"
            )],
            "retry_count": retry_count + 1,
            "current_step": "retrying",
        }

    _save_to_history(state, last_ai_message.content)
    return {"retry_count": 0, "current_step": "completed"}


def _save_to_history(state: PRDState, report_content: str):
    """
    Bug fix：从消息历史提取PRD内容，而不是依赖 state["prd_content"]
    state["prd_content"] 从未被写入，原来的实现导致历史库永远为空
    """
    prd_summary = ""
    prefix = "请分析这份PRD文档：\n\n"
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and prefix in msg.content:
            prd_summary = msg.content[len(prefix):len(prefix)+150]
            break

    if not prd_summary:
        return

    # 从工具结果中提取结构化数据写入历史库
    score = 0
    missing_fields = []
    risks = []
    user_stories = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if "score" in data:
                    score = data["score"]
                    missing_fields = data.get("missing_fields", [])
                if "risks" in data:
                    risks = [r.get("description", "") for r in data.get("risks", [])]
                if "user_stories" in data:
                    user_stories = [s.get("story", "") for s in data.get("user_stories", [])]
            except (json.JSONDecodeError, AttributeError):
                pass

    save_review_to_history(
        prd_summary,
        {
            "score": score,
            "missing_fields": missing_fields,
            "risks": risks,
            "user_stories": user_stories,
            "conclusion": report_content[:300],
        },
    )


# ── 路由 ──────────────────────────────────────────────────────────────────
def _route_assistant(state: PRDState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "parallel_tools"
    if len(last.content) >= 200:
        return "reflection"
    return END


def _route_reflection(state: PRDState):
    return "assistant" if state.get("current_step") == "retrying" else END


# ── 构建图 ────────────────────────────────────────────────────────────────
def build_graph():
    os.makedirs("state_db", exist_ok=True)
    conn = sqlite3.connect("state_db/prd_agent.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    builder = StateGraph(PRDState)
    builder.add_node("assistant", assistant)
    builder.add_node("parallel_tools", parallel_tools)
    builder.add_node("reflection", reflection)

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant", _route_assistant,
        {"parallel_tools": "parallel_tools", "reflection": "reflection", END: END},
    )
    builder.add_edge("parallel_tools", "assistant")
    builder.add_conditional_edges(
        "reflection", _route_reflection,
        {"assistant": "assistant", END: END},
    )

    return builder.compile(checkpointer=memory)


graph = build_graph()


# ── 流式输出（供 app.py 调用）────────────────────────────────────────────
def stream_graph_updates(user_message: str, config: dict):
    """
    流式执行 graph，yield 最终报告的每个 token。
    Reflection 重试时只流式输出最后一次生成的报告，避免重复内容。
    """
    input_state = {"messages": [HumanMessage(content=user_message)]}
    in_final_report = False  # 标记是否进入最终报告生成阶段

    for event in graph.stream(input_state, config, stream_mode="messages"):
        if not isinstance(event, tuple):
            continue
        msg_chunk, metadata = event
        node = metadata.get("langgraph_node", "")

        # reflection 节点执行完后，下一个 assistant 才是最终报告
        if node == "reflection":
            in_final_report = True
            continue

        if node == "assistant":
            # 跳过 tool_call 阶段的 assistant（第一次调用工具时）
            if getattr(msg_chunk, "tool_calls", None):
                continue
            if hasattr(msg_chunk, "content") and msg_chunk.content:
                yield msg_chunk.content