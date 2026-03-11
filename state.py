from typing_extensions import NotRequired
from langgraph.graph import MessagesState

class PRDState(MessagesState):
    # PRD原文
    prd_content: NotRequired[str]
    # 完整性评分
    completeness_score: NotRequired[int]
    # 缺失的字段
    missing_fields: NotRequired[list[str]]
    # 拆解出的用户故事
    user_stories: NotRequired[list[str]]
    # 识别出的风险
    risks: NotRequired[list[str]]
    # 对话摘要
    summary: NotRequired[str]
    # 当前执行步骤
    current_step: NotRequired[str]
    # Reflection 重试次数（防止无限循环，上限2次）
    retry_count: NotRequired[int]