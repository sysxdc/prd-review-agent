import streamlit as st
import pypdf
import io
import uuid
from langchain_core.messages import HumanMessage
from agent import graph, stream_graph_updates

st.title("PRD 需求评审 Agent")
st.caption("上传 PRD 文档，获得 RAG 增强的结构化评审报告")

# ── 会话状态初始化 ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False   # 防止上传后重复触发分析

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── 清除对话 ───────────────────────────────────────────────────────────────
if st.button("🗑️ 清除对话，重新开始"):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.analyzed = False
    st.rerun()

# ── 历史消息 ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 文件上传 ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "上传 PRD 文档（支持 PDF / TXT）",
    type=["txt", "pdf"],
    # key 绑定 thread_id，清除对话后自动重置上传区
    key=f"uploader_{st.session_state.thread_id}",
)

if uploaded_file and not st.session_state.analyzed:
    # 解析文件内容
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        content = "\n".join(
            page.extract_text() or "" for page in pdf_reader.pages
        )
    else:
        content = uploaded_file.read().decode("utf-8", errors="ignore")

    if content.strip():
        st.success(f"✅ 文档解析成功，共 {len(content)} 字符")

        if st.button("🚀 开始分析"):
            # 标记已分析，防止 rerun 重复触发
            st.session_state.analyzed = True

            user_msg = "已上传 PRD 文档，请开始评审"
            st.session_state.messages.append({"role": "user", "content": user_msg})

            prompt = f"请分析这份PRD文档：\n\n{content}"

            # 流式输出
            with st.chat_message("assistant"):
                response = st.write_stream(
                    stream_graph_updates(prompt, config)
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })
            st.rerun()
    else:
        st.error("文档内容为空，请检查文件")

# ── 追问输入（始终显示，分析前也可以直接提问）──────────────────────────────
user_input = st.chat_input("上传文档后可继续追问...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 追问也用流式
    with st.chat_message("assistant"):
        response = st.write_stream(
            stream_graph_updates(user_input, config)
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
    })
    st.rerun()