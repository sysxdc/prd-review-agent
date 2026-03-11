"""
RAG 知识库模块
- 行业规范库：预置 PRD 标准规范文档
- 历史评审库：每次评审完自动入库，下次相似 PRD 召回参考案例
"""

import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("EMBED_BASE_URL"),
)

CHROMA_DIR = "chroma_db"

# ─────────────────────────────────────────────
# 行业规范库（Industry Standards）
# ─────────────────────────────────────────────
INDUSTRY_STANDARDS = [
    {
        "content": """PRD完整性标准：一份合格的PRD必须包含以下8个核心模块：
1. 产品背景与目标：说明为什么做这个产品，解决什么问题，OKR是什么
2. 目标用户画像：用户分群、核心痛点、使用场景描述
3. 功能需求清单：按优先级（P0/P1/P2）列出所有功能点，每条需求需明确输入/输出/异常处理
4. 非功能需求：性能指标（响应时间<500ms）、安全要求、兼容性、可扩展性
5. 数据需求：涉及的数据模型、数据流向、埋点方案
6. 交互与UI说明：关键流程的流程图或原型图链接
7. 验收标准：每个功能点对应的测试用例或AC（Acceptance Criteria）
8. 上线计划：版本规划、灰度策略、回滚方案""",
        "metadata": {"type": "standard", "category": "completeness", "source": "PRD规范指南v2.0"},
    },
    {
        "content": """用户故事写作规范：
标准格式：作为[用户角色]，我希望[完成某件事]，以便[获得某种价值]
质量标准（INVEST原则）：
- Independent（独立性）：故事之间尽量不相互依赖
- Negotiable（可协商）：细节可以讨论，不是固定合同
- Valuable（有价值）：对用户或业务有明确价值
- Estimable（可估算）：开发团队能估算工作量
- Small（小）：能在一个Sprint内完成
- Testable（可测试）：有明确的验收条件
反例（不合格）：作为用户，我希望系统运行更快。（缺少具体场景和可测试标准）
正例（合格）：作为电商买家，我希望商品搜索结果在2秒内返回，以便我能快速找到目标商品，验收标准：P99延迟<2s，空结果率<5%""",
        "metadata": {"type": "standard", "category": "user_story", "source": "敏捷需求规范"},
    },
    {
        "content": """技术风险识别清单（互联网产品常见风险）：
高危风险（必须在PRD阶段识别）：
- 第三方依赖风险：支付、短信、地图等第三方服务的SLA，是否有降级方案
- 数据安全与隐私：涉及用户个人信息（手机号/位置/行为数据）需符合GDPR/个人信息保护法
- 高并发场景：活动秒杀、春节流量峰值，需明确限流/熔断策略
- 数据一致性：分布式系统中的最终一致性问题，如库存扣减
中危风险（建议在PRD阶段标注）：
- 兼容性风险：iOS/Android最低版本要求，老旧机型适配
- 迁移风险：老数据迁移方案，用户习惯改变带来的流失
- 依赖排期风险：需要其他团队配合的模块，排期是否对齐""",
        "metadata": {"type": "standard", "category": "risk", "source": "互联网产品风险管理规范"},
    },
    {
        "content": """PRD评分标准（100分制）：
90-100分（优秀）：8个模块全部完整，用户故事符合INVEST原则，有完整AC，技术风险已识别并有应对方案
70-89分（良好）：核心模块（功能需求/验收标准/上线计划）完整，有2个以下模块缺失，风险部分识别
50-69分（及格）：功能需求清单存在，但缺少AC或优先级，用户故事不规范，风险未充分识别
30-49分（不及格）：超过4个核心模块缺失，功能描述模糊，无法直接进入开发
0-29分（极差）：仅有功能列表或概念描述，无法作为开发依据""",
        "metadata": {"type": "standard", "category": "scoring", "source": "PRD质量评分体系"},
    },
    {
        "content": """常见PRD缺陷模式及改进建议：
缺陷1「需求不可测试」：功能描述中出现"支持"、"优化"、"提升体验"等模糊词汇，无量化指标
  → 改进：每条功能需求附上可量化的验收标准，如"加载时间<3秒"
缺陷2「边界条件缺失」：只描述正常流程（Happy Path），未考虑异常输入、网络中断、并发冲突
  → 改进：为每个关键功能补充异常流程和边界条件说明
缺陷3「假设未明确」：隐含了大量假设但未写出，如"假设用户已登录"、"假设库存系统已就绪"
  → 改进：在PRD开头明确列出所有前提假设和依赖项
缺陷4「优先级缺失」：所有功能平铺，无P0/P1/P2区分，开发团队无法决策取舍
  → 改进：用MoSCoW法则（Must/Should/Could/Won't）标注每条需求""",
        "metadata": {"type": "standard", "category": "defects", "source": "PRD常见问题手册"},
    },
]


def _get_standards_store() -> Chroma:
    """获取或创建行业规范向量库"""
    standards_dir = os.path.join(CHROMA_DIR, "standards")
    store = Chroma(
        collection_name="industry_standards",
        embedding_function=EMBED_MODEL,
        persist_directory=standards_dir,
    )
    # 如果库是空的，初始化预置文档
    if store._collection.count() == 0:  # langchain-chroma 1.x 仍支持此写法
        docs = [
            Document(page_content=item["content"], metadata=item["metadata"])
            for item in INDUSTRY_STANDARDS
        ]
        store.add_documents(docs)
        print(f"[RAG] 行业规范库初始化完成，写入 {len(docs)} 条文档")
    return store


def _get_history_store() -> Chroma:
    """获取或创建历史评审向量库"""
    history_dir = os.path.join(CHROMA_DIR, "history")
    store = Chroma(
        collection_name="review_history",
        embedding_function=EMBED_MODEL,
        persist_directory=history_dir,
    )
    return store


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def retrieve_standards(query: str, category: str = None, k: int = 3) -> list[str]:
    """
    从行业规范库检索相关规范
    :param query: 检索查询
    :param category: 可选过滤，如 "completeness" / "risk" / "user_story" / "scoring"
    :param k: 返回条数
    """
    store = _get_standards_store()
    # langchain-chroma 单条件直接传字段名即可，多条件才需要 $and
    if category:
        filter_dict = {"category": category}
    else:
        filter_dict = None

    results = store.similarity_search(query, k=k, filter=filter_dict)
    return [doc.page_content for doc in results]


def retrieve_similar_reviews(prd_summary: str, k: int = 2) -> list[str]:
    """
    从历史评审库检索相似案例
    :param prd_summary: 当前PRD的摘要描述
    :param k: 返回条数
    """
    store = _get_history_store()
    try:
        if store._collection.count() == 0:
            return []
    except Exception:
        return []
    results = store.similarity_search(prd_summary, k=k)
    return [doc.page_content for doc in results]


def save_review_to_history(prd_summary: str, review_result: dict):
    """
    将评审结果写入历史库，供后续 RAG 召回
    :param prd_summary: PRD摘要（用于向量化）
    :param review_result: 包含 score/missing_fields/risks/user_stories 的字典
    """
    store = _get_history_store()
    content = f"""历史评审案例：
PRD类型：{prd_summary}
完整性评分：{review_result.get('score', 'N/A')}/100
缺失字段：{', '.join(review_result.get('missing_fields', []))}
主要风险：{', '.join(review_result.get('risks', [])[:3])}
用户故事数量：{len(review_result.get('user_stories', []))}条
评审结论：{review_result.get('conclusion', '')}"""

    doc = Document(
        page_content=content,
        metadata={"type": "history", "prd_type": prd_summary[:50]},
    )
    store.add_documents([doc])
    print(f"[RAG] 历史评审已入库：{prd_summary[:30]}...")