# streamlit_app.py
"""
传感器知识库问答机器人（Web 交互版）
运行：streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import require_api_key
from src.logging_config import setup_logging
from src.rag_chat import SensorRAGChat
from langchain_core.messages import HumanMessage, AIMessage

setup_logging(debug=False)

st.set_page_config(page_title="传感器问答助手", page_icon="🤖", layout="wide")

st.title("🤖 传感器知识库问答助手")
st.caption("基于 RAG 架构的工业传感器监控与选型知识库")

@st.cache_resource
def init_chat():
    try:
        require_api_key()
        return SensorRAGChat()
    except Exception as e:
        st.error(f"初始化失败：{e}")
        return None

chat = init_chat()

# 初始化对话历史状态（保存底层 LLM 消息对象）
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ 控制面板")
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("##### ℹ️ 使用说明")
    st.caption("1. 提问尽量具体。\n2. 支持多轮无主语追问（如'那它的封装呢'）。\n3. 展开底部可查看溯源与消耗。")

# 展示历史对话记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "meta" in message and message["meta"]:
            with st.expander("📊 详细信息 & 引用溯源"):
                cost_data = message["meta"].get("cost", "")
                sources_data = message["meta"].get("sources", "")
                if cost_data:
                    st.caption(cost_data)
                if sources_data:
                    st.markdown("**📚 参考来源文件：**")
                    st.markdown(sources_data)

if prompt := st.chat_input("请输入关于传感器的问题..."):
    if not chat:
        st.warning("系统未初始化，请检查 API Key 或向量库配置。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            meta_info = {}
            
            start_time = time.time()  # 掐表开始
            try:
                for event in chat.ask_stream(prompt, history=st.session_state.history):
                    if not event.done:
                        full_response += event.delta
                        response_placeholder.markdown(full_response + "▌")
                    else:
                        result = event.result
                        st.session_state.history = result.history
                        end_time = time.time()  # 掐表结束
                        elapsed_time = end_time - start_time
                        
                        meta_info["cost"] = f"⏱️ 端到端耗时: {elapsed_time:.2f}s | 💰 本次消耗：{result.usage.total_tokens} tokens | 估算成本：¥{result.usage.estimated_cost_cny:.6f}"
                        if result.sources:
                            sources_md = "\n".join([f"- 📄 `{src}`" for src in result.sources])
                            meta_info["sources"] = sources_md
                        
                response_placeholder.markdown(full_response)
                with st.expander("📊 详细信息 & 引用溯源"):
                    st.caption(meta_info.get("cost", ""))
                    if meta_info.get("sources"):
                        st.markdown("**📚 参考来源文件：**")
                        st.markdown(meta_info["sources"])
                        
            except Exception as e:
                full_response = f"❌ 发生错误：{e}"
                response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "meta": meta_info
            })