import os
import sqlite3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import tools_condition, ToolNode

from state import PRDState
from tools import check_completeness, extract_user_stories, identify_risks

load_dotenv()

# 初始化LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)

tools = [check_completeness, extract_user_stories, identify_risks]
llm_with_tools = llm.bind_tools(tools)

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的产品需求评审助手。

分析PRD文档时，必须按以下格式输出：

## 📊 完整性评分：X/100

## ❌ 缺失字段
- 字段名：缺失原因

## 📖 用户故事
- 作为[用户]，我希望[功能]，以便[价值]

## ⚠️ 风险识别
- 风险类型：具体描述

## 💡 改进建议
- 具体可执行的建议

回答追问时正常对话即可，不需要固定格式。
请用中文回答。"""

# 核心节点
def assistant(state: PRDState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": response}

# 构建图
def build_graph():
    conn = sqlite3.connect("state_db/prd_agent.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    builder = StateGraph(PRDState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    
    return builder.compile(checkpointer=memory)

graph = build_graph()