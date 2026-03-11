"""
tools.py（RAG增强版）

改造重点：
- 每个工具不再是空壳，而是先用 RAG 召回相关规范，再拼入 LLM prompt
- LLM 有了「规范锚点」，输出更准确、更稳定，不靠自由发挥
- 结果写回 state，供 reflection 节点做质量验证
"""

import os
import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from rag_store import retrieve_standards, retrieve_similar_reviews

load_dotenv()

# 工具内部用的 LLM（同一个 DeepSeek，不依赖外部传入）
_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


@tool
def check_completeness(prd_content: str) -> str:
    """
    检查PRD完整性，给出0-100评分和缺失字段列表。
    内部使用 RAG 召回行业完整性规范作为评判依据。
    """
    # Step 1: 召回相关规范
    standards = retrieve_standards(
        query="PRD完整性检查 必须包含哪些模块",
        k=2,
    )
    scoring_std = retrieve_standards(
        query="PRD评分标准 100分",
        k=1,
    )
    all_standards = standards + scoring_std

    standards_text = "\n\n---\n\n".join(all_standards)

    # Step 2: 召回历史相似案例（如果有）
    similar_cases = retrieve_similar_reviews(prd_content[:200], k=2)
    history_text = ""
    if similar_cases:
        history_text = "\n\n【历史相似案例参考】\n" + "\n---\n".join(similar_cases)

    # Step 3: 带着规范上下文请求 LLM
    prompt = f"""你是一个专业PRD评审专家。请根据以下行业规范，对PRD进行完整性评估。

【行业评审规范】
{standards_text}
{history_text}

【待评审PRD内容】
{prd_content}

请严格按照规范，输出以下JSON格式（不要输出任何其他内容）：
{{
  "score": <0-100的整数>,
  "missing_fields": ["缺失字段1（原因）", "缺失字段2（原因）"],
  "present_fields": ["已有字段1", "已有字段2"],
  "score_rationale": "评分依据，2-3句话"
}}"""

    response = _llm.invoke([HumanMessage(content=prompt)])
    
    # 健壮解析
    try:
        # 去除可能的 markdown 代码块
        raw = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        result = json.loads(raw)
        return json.dumps(result, ensure_ascii=False)
    except json.JSONDecodeError:
        # 解析失败时返回原文，让 agent 自己处理
        return response.content


@tool
def extract_user_stories(prd_content: str) -> str:
    """
    从PRD中提取并规范化用户故事，确保符合INVEST原则。
    内部使用 RAG 召回用户故事写作规范。
    """
    standards = retrieve_standards(
        query="用户故事 INVEST原则 写作规范 格式",
        k=2,
    )
    standards_text = "\n\n".join(standards)

    prompt = f"""你是一个敏捷需求专家。请根据以下用户故事规范，从PRD中提取并规范化用户故事。

【用户故事规范】
{standards_text}

【待分析PRD内容】
{prd_content}

要求：
1. 每条故事必须严格遵循"作为[角色]，我希望[行为]，以便[价值]"格式
2. 每条故事附上简短的验收标准（AC）
3. 按优先级排序（P0最高）

输出JSON格式：
{{
  "user_stories": [
    {{
      "priority": "P0",
      "story": "作为[角色]，我希望[行为]，以便[价值]",
      "acceptance_criteria": "验收标准描述"
    }}
  ],
  "total_count": <数量>,
  "quality_note": "整体质量说明"
}}"""

    response = _llm.invoke([HumanMessage(content=prompt)])
    try:
        raw = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        result = json.loads(raw)
        return json.dumps(result, ensure_ascii=False)
    except json.JSONDecodeError:
        return response.content


@tool
def identify_risks(prd_content: str) -> str:
    """
    识别PRD中的技术风险、边界问题和常见缺陷。
    内部使用 RAG 召回风险识别清单和常见缺陷模式。
    """
    risk_standards = retrieve_standards(
        query="技术风险识别 第三方依赖 数据安全 高并发",
        k=2,
    )
    defect_standards = retrieve_standards(
        query="PRD常见缺陷 需求不可测试 边界条件缺失",
        k=1,
    )
    standards_text = "\n\n".join(risk_standards + defect_standards)

    prompt = f"""你是一个技术架构师兼需求分析专家。请根据以下风险清单，对PRD进行风险识别。

【风险识别规范和常见缺陷】
{standards_text}

【待分析PRD内容】
{prd_content}

请输出JSON格式：
{{
  "risks": [
    {{
      "level": "高危/中危/低危",
      "category": "技术风险/需求缺陷/依赖风险",
      "description": "风险描述",
      "suggestion": "应对建议"
    }}
  ],
  "defects": [
    {{
      "defect_type": "缺陷类型名称",
      "location": "出现在PRD哪个部分",
      "suggestion": "改进建议"
    }}
  ],
  "overall_risk_level": "高/中/低"
}}"""

    response = _llm.invoke([HumanMessage(content=prompt)])
    try:
        raw = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        result = json.loads(raw)
        return json.dumps(result, ensure_ascii=False)
    except json.JSONDecodeError:
        return response.content