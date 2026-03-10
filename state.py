from typing import Annotated
from pydantic import BaseModel
from langgraph.graph import MessagesState

class PRDState(MessagesState):
    # PRD原文
    prd_content: str
    # 完整性评分
    completeness_score: int
    # 缺失的字段
    missing_fields: list
    # 拆解出的用户故事
    user_stories: list
    # 识别出的风险
    risks: list
    # 对话摘要
    summary: str
    # 当前执行步骤
    current_step: str