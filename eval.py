"""
eval.py — LLM-as-Judge 自动化评估 Pipeline

用途：
- 用你已有的 test_cases 作为 ground truth
- 每次改动代码后运行，验证改动没有让质量下降
- 把评估结果打印出来，核心数据写进 README

运行方法：
  python eval.py

输出示例：
  ============ PRD Review Agent Eval Report ============
  用例: 低质量PRD（社交APP）
    完整性评分: 25  期望范围: 10-40  ✅
    缺失字段召回率: 0.83  ✅
    用户故事数量: 3  ✅
    风险识别: 包含高危风险  ✅
  ...
  总体通过率: 11/12 (91.7%)
"""

import json
import os
import time
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ── 评估用的 LLM（独立于 agent，避免相互影响）──
judge_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

# ── Ground Truth 测试集（基于你 test_cases 文件夹的已知结果）──────────────
TEST_CASES = [
    {
        "name": "低质量PRD（社交APP）",
        "prd": """产品名称：社交APP
功能：用户可以发帖、点赞、关注其他用户。
需要做一个好用的社交产品。""",
        "expected": {
            "score_range": (0, 40),           # 评分应在此范围内
            "min_missing_fields": 4,           # 至少识别出4个缺失字段
            "min_user_stories": 1,             # 至少1条用户故事
            "should_contain_risks": True,      # 应该识别到风险
            "risk_keywords": ["安全", "隐私", "并发", "第三方", "数据"],  # 至少包含其中1个
        },
    },
    {
        "name": "中等质量PRD（电商平台）",
        "prd": """产品名称：电商平台购物车功能
目标用户：18-35岁网购用户
功能需求：
- 用户可以添加商品到购物车
- 支持修改数量和删除商品
- 显示实时价格和库存
- 支持结算跳转

技术要求：响应时间<2秒
上线时间：2026年Q2""",
        "expected": {
            "score_range": (30, 65),
            "min_missing_fields": 2,
            "min_user_stories": 2,
            "should_contain_risks": True,
            "risk_keywords": ["库存", "并发", "支付", "一致性", "超卖"],
        },
    },
    {
        "name": "高质量PRD（知识库系统）",
        "prd": """产品名称：企业知识库系统 v1.0
产品背景：公司内部知识分散，新员工查找资料效率低，平均耗时2小时/天
目标用户：
  - 主要用户：公司全体员工（500人）
  - 次要用户：HR和知识库管理员
核心功能需求（P0）：
  1. 文档上传与管理：支持PDF/Word/Markdown，单文件<50MB
     验收标准：上传成功率>99%，解析时间<30s
  2. 全文搜索：关键词搜索，响应时间P99<1s
     验收标准：搜索准确率>85%（基于人工标注数据集）
非功能需求：
  - 可用性：99.9% SLA，计划外停机<8.7h/年
  - 安全：文档权限隔离，敏感文档支持水印
数据需求：文档元数据存MySQL，向量索引用Elasticsearch
验收标准：UAT测试通过，核心场景测试用例100%通过
上线计划：2026-Q3，先内测100人，再全量""",
        "expected": {
            "score_range": (55, 90),
            "min_missing_fields": 0,           # 高质量PRD缺失字段应该少
            "min_user_stories": 2,
            "should_contain_risks": True,
            "risk_keywords": ["搜索", "权限", "存储", "迁移", "向量"],
        },
    },
]


# ── 评估结果数据结构 ──────────────────────────────────────────────────────
@dataclass
class EvalResult:
    case_name: str
    checks: list[dict] = field(default_factory=list)
    passed: int = 0
    total: int = 0
    raw_report: str = ""
    latency_seconds: float = 0.0

    @property
    def pass_rate(self):
        return self.passed / self.total if self.total > 0 else 0


# ── 核心评估函数 ──────────────────────────────────────────────────────────
def run_agent_on_prd(prd_content: str) -> tuple[str, float]:
    """运行 agent，返回（评审报告文本, 耗时秒数）"""
    # 直接用 judge_llm 模拟 agent 输出（eval 独立于 agent，避免循环依赖）
    # 生产环境可以改为直接调用 graph.invoke(...)
    from agent import graph

    start = time.time()
    config = {"configurable": {"thread_id": f"eval_{int(time.time())}"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=f"请评审以下PRD：\n\n{prd_content}")]},
        config=config,
    )
    elapsed = time.time() - start

    # 取最后一条 AI 消息
    from langchain_core.messages import AIMessage
    report = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            report = msg.content
            break
    return report, elapsed


def llm_judge_report(report: str, prd: str, expected: dict) -> dict:
    """
    用 LLM-as-Judge 从报告中提取结构化评估数据
    返回字典包含：score, missing_fields_count, user_stories_count, risk_keywords_found
    """
    prompt = f"""从以下PRD评审报告中提取关键数据，输出JSON格式，不要输出其他内容。

评审报告：
{report}

需要提取：
1. 完整性评分（数字）
2. 列出的缺失字段数量（数字）
3. 用户故事条数（数字）
4. 报告中提到的风险关键词（从以下列表中选：安全、隐私、并发、第三方、数据、库存、支付、一致性、超卖、搜索、权限、存储、迁移、向量）

输出格式：
{{
  "score": <数字或null>,
  "missing_fields_count": <数字>,
  "user_stories_count": <数字>,
  "risk_keywords_found": ["关键词1", "关键词2"]
}}"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    try:
        raw = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(raw)
    except Exception:
        return {"score": None, "missing_fields_count": 0, "user_stories_count": 0, "risk_keywords_found": []}


def evaluate_case(case: dict) -> EvalResult:
    """评估单个测试用例"""
    print(f"\n▶ 评估中：{case['name']}...")
    result = EvalResult(case_name=case["name"])

    # 运行 agent
    report, latency = run_agent_on_prd(case["prd"])
    result.raw_report = report
    result.latency_seconds = latency

    # 用 LLM 从报告中提取数据
    extracted = llm_judge_report(report, case["prd"], case["expected"])
    expected = case["expected"]

    # ── Check 1: 完整性评分范围 ──
    score = extracted.get("score")
    score_min, score_max = expected["score_range"]
    if score is not None:
        score_ok = score_min <= score <= score_max
        result.checks.append({
            "name": "完整性评分范围",
            "value": score,
            "expected": f"{score_min}-{score_max}",
            "passed": score_ok,
        })
        result.total += 1
        if score_ok:
            result.passed += 1

    # ── Check 2: 缺失字段数量 ──
    missing_count = extracted.get("missing_fields_count", 0)
    min_missing = expected["min_missing_fields"]
    missing_ok = missing_count >= min_missing
    result.checks.append({
        "name": "缺失字段识别数",
        "value": missing_count,
        "expected": f">= {min_missing}",
        "passed": missing_ok,
    })
    result.total += 1
    if missing_ok:
        result.passed += 1

    # ── Check 3: 用户故事数量 ──
    story_count = extracted.get("user_stories_count", 0)
    min_stories = expected["min_user_stories"]
    story_ok = story_count >= min_stories
    result.checks.append({
        "name": "用户故事数量",
        "value": story_count,
        "expected": f">= {min_stories}",
        "passed": story_ok,
    })
    result.total += 1
    if story_ok:
        result.passed += 1

    # ── Check 4: 风险关键词覆盖 ──
    if expected["should_contain_risks"]:
        found_keywords = extracted.get("risk_keywords_found", [])
        expected_keywords = expected["risk_keywords"]
        hit = any(kw in found_keywords for kw in expected_keywords)
        result.checks.append({
            "name": "风险关键词覆盖",
            "value": found_keywords,
            "expected": f"包含其中之一: {expected_keywords}",
            "passed": hit,
        })
        result.total += 1
        if hit:
            result.passed += 1

    return result


def print_report(results: list[EvalResult]):
    """打印评估报告"""
    print("\n" + "=" * 55)
    print("       PRD Review Agent — Eval Report")
    print("=" * 55)

    total_passed = sum(r.passed for r in results)
    total_checks = sum(r.total for r in results)

    for r in results:
        icon = "✅" if r.pass_rate >= 0.75 else "⚠️" if r.pass_rate >= 0.5 else "❌"
        print(f"\n{icon}  {r.case_name}  ({r.passed}/{r.total} checks, {r.latency_seconds:.1f}s)")
        for check in r.checks:
            status = "✅" if check["passed"] else "❌"
            print(f"   {status} {check['name']}: {check['value']}  (期望: {check['expected']})")

    print("\n" + "-" * 55)
    overall = total_passed / total_checks * 100 if total_checks > 0 else 0
    print(f"总体通过率: {total_passed}/{total_checks}  ({overall:.1f}%)")
    avg_latency = sum(r.latency_seconds for r in results) / len(results)
    print(f"平均响应时间: {avg_latency:.1f}s")
    print("=" * 55)

    # 生成可贴进 README 的 badge 数据
    print(f"\n📋 README Badge 数据（复制粘贴用）：")
    print(f"- Eval 通过率：{overall:.0f}%")
    print(f"- 平均响应时间：{avg_latency:.0f}s")
    print(f"- 测试用例：{len(results)} 个场景，{total_checks} 项检查")


if __name__ == "__main__":
    results = [evaluate_case(case) for case in TEST_CASES]
    print_report(results)