# PRD需求评审Agent

> 上传产品需求文档，60秒获得专业评审报告

## 解决什么问题
产品评审会议平均返工2-3次，核心原因是需求描述不清晰、
验收标准缺失、技术风险未识别。本项目用AI自动完成评审工作。

## 在线Demo
[待补充]

## 核心功能
- ✅ 需求完整性检测
- ✅ 自动拆解用户故事  
- ✅ 技术风险识别
- ✅ PRD质量评分
- ✅ 支持多轮追问

## 技术栈
LangGraph + DeepSeek API + SQLite + Streamlit

## 架构图
[待补充]

## 快速开始
pip install -r requirements.txt
streamlit run app.py
```

---

现在的项目已经完成了约40%：
```
✅ 环境搭建
✅ Agent核心逻辑
✅ PDF解析
✅ 对话记忆
✅ 基础界面

待完成：
⬜ Prompt优化（结构化输出）
⬜ 评分系统
⬜ README完善
⬜ 部署上线
⬜ 录制Demo GIF
