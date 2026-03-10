from langchain_core.tools import tool

@tool
def check_completeness(prd_content: str) -> str:
    """检查PRD完整性，识别缺失字段"""
    required_fields = [
        "产品背景", "目标用户", "功能需求", 
        "非功能需求", "验收标准", "上线计划"
    ]
    # 这里后续用LLM来判断
    return f"需要检查的字段：{required_fields}"

@tool  
def extract_user_stories(prd_content: str) -> str:
    """从PRD中提取用户故事"""
    return "提取用户故事中..."

@tool
def identify_risks(prd_content: str) -> str:
    """识别PRD中的技术风险和边界问题"""
    return "识别风险中..."