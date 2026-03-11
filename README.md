# PRD需求评审Agent

> 上传产品需求文档，60秒获得专业评审报告

## 在线Demo
🔗 [立即体验](https://prd-review-agent-hsw3pyajndptpvzuacvxsd.streamlit.app/)

![Demo演示](动画.gif)

## 解决什么问题

产品评审会议平均返工2-3次，核心原因是需求描述不清晰、
验收标准缺失、技术风险未识别。

本项目用AI Agent自动完成评审工作，输出结构化报告。

## 核心功能

- ✅ 需求完整性检测（8个维度）
- ✅ PRD质量评分（0-100分）
- ✅ 自动拆解用户故事
- ✅ 技术风险识别
- ✅ 多轮追问，持续对话
- ✅ 对话历史持久化（SQLite）

## 测试数据

测试文档见 [test_cases](./test_cases) 文件夹，可直接下载上传体验。

| 文档类型 | 完整性评分 | 识别缺失字段数 | 生成用户故事数 |
|--------|---------|------------|------------|
| 低质量PRD（社交APP） | 25/100 | 6个 | 3条 |
| 中等质量PRD（电商平台） | 40/100 | 6个 | 4条 |
| 高质量PRD（知识库系统） | 65/100 | 10个 | 3条 |

评分梯度清晰，能准确区分PRD质量高低。

## 系统架构

![架构图](architecture.png)

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent框架 | LangGraph |
| 大模型 | DeepSeek API |
| 持久化 | SQLite |
| 界面 | Streamlit |
| 文档解析 | PyPDF |

## 快速开始

\```bash
# 1. 克隆项目
git clone https://github.com/sysxdc/prd-review-agent.git
cd prd-review-agent

# 2. 配置环境变量
cp .env.example .env
# 填入你的 DEEPSEEK_API_KEY

# 3. 安装依赖并运行
pip install -r requirements.txt
streamlit run app.py
\```

## 踩坑记录

**坑1：PDF解析乱码**
直接用decode()读取PDF会乱码，改用pypdf库解析后解决。

**坑2：对话历史重复输出**
Agent返回所有消息记录，只取最后一条AIMessage解决。

**坑3：SQLite路径报错**
state_db文件夹不存在导致报错，手动创建文件夹解决。

**坑4：Streamlit Cloud部署失败**
requirements.txt包含整个Python环境导致安装失败，精简为只保留项目依赖解决。