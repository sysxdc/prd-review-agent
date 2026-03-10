import streamlit as st
import pypdf
import io
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agent import graph

st.title("PRD需求评审Agent")
st.caption("上传你的PRD文档，获得专业评审报告")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 顶部放清除按钮
if st.button("清除对话，重新开始"):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 上传文件
uploaded_file = st.file_uploader("上传PRD文档", type=["txt", "pdf"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text()
    else:
        content = uploaded_file.read().decode("utf-8", errors="ignore")

    st.success("文档上传成功")

    if st.button("开始分析"):
        with st.spinner("分析中..."):
            message = HumanMessage(
                content=f"请分析这份PRD文档：\n\n{content}"
            )
            result = graph.invoke({"messages": [message]}, config)

            last_ai_message = None
            for m in reversed(result["messages"]):
                if isinstance(m, AIMessage) and m.content:
                    last_ai_message = m.content
                    break

            if last_ai_message:
                st.session_state.messages.append({
                    "role": "user",
                    "content": "已上传PRD文档，请分析"
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": last_ai_message
                })
                st.rerun()

# 追问功能
user_input = st.chat_input("继续追问...")
if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("思考中..."):
        message = HumanMessage(content=user_input)
        result = graph.invoke({"messages": [message]}, config)

        last_ai_message = None
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage) and m.content:
                last_ai_message = m.content
                break

        if last_ai_message:
            st.session_state.messages.append({
                "role": "assistant",
                "content": last_ai_message
            })

    st.rerun()