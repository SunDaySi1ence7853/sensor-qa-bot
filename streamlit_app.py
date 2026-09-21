import streamlit as st
import pandas as pd
import json
from src.agent import get_sensor_agent
from src.tools.sensor_tools import get_sensor_data, query_history
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ----------------- 页面基础配置 -----------------
st.set_page_config(page_title="传感器 QA Bot", page_icon="🤖", layout="wide")

# 缓存加载 Agent，避免每次对话都重新初始化
@st.cache_resource
def load_agent():
    return get_sensor_agent()

agent = load_agent()

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 工业传感器监控与诊断助手")
st.caption("基于 LangGraph + DeepSeek + ESP32 实时串口数据")

# ----------------- 页面布局：左 60% 聊天，右 40% 图表 -----------------
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("💬 Agent 对话区")
    
    # 展示历史对话
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                # 显示历史 Token 与费用
                if msg.get("tokens", 0) > 0:
                    st.caption(f"🔍 Token消耗: {msg['tokens']} | 💰 估算费用: ¥{msg['cost']:.4f}")
                # 显示历史知识引用
                if msg.get("refs"):
                    with st.expander("📚 知识库引用来源"):
                        for i, ref in enumerate(msg["refs"]):
                            st.markdown(f"**片段 {i+1}：**")
                            st.text(ref[:500] + "..." if len(ref) > 500 else ref)

    # 接收用户输入
    if prompt := st.chat_input("问问 Agent：现在温度正常吗？或者问硬件规格"):
        # 展示并记录用户输入
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 Agent 获取回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Agent 正在思考并调用工具..."):
                # 将 session_state 里的历史转换为 LangChain 消息格式
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in st.session_state.messages
                ]
                
                # 调用 ReAct Agent
                response = agent.invoke({"messages": history})
                
                # 提取最后一条 AI 消息
                ai_msg = response["messages"][-1]
                ai_content = ai_msg.content
                
                # 1. 提取 Token 统计与计算费用
                total_tokens = 0
                cost = 0.0
                if hasattr(ai_msg, 'usage_metadata') and ai_msg.usage_metadata:
                    usage = ai_msg.usage_metadata
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    # 估算费用：假设 DeepSeek 输入 0.001元/千 tokens，输出 0.002元/千 tokens
                    cost = (input_tokens * 0.001 + output_tokens * 0.002) / 1000
                
                # 2. 提取知识库引用（遍历消息找 ToolMessage）
                references = []
                for msg in response["messages"]:
                    # 知识库工具名如果是 search_manual
                    if isinstance(msg, ToolMessage) and msg.name == "search_manual":
                        try:
                            # 尝试解析工具返回的 JSON，提取具体的文档片段
                            ref_data = json.loads(msg.content)
                            if "content" in ref_data:
                                references.append(ref_data["content"])
                            elif "documents" in ref_data:
                                references.extend(ref_data["documents"])
                        except:
                            references.append(msg.content)

                # 输出主体回复
                message_placeholder.markdown(ai_content)
                
                # 输出 Token 统计与引用
                if total_tokens > 0:
                    st.caption(f"🔍 Token消耗: {total_tokens} | 💰 估算费用: ¥{cost:.4f}")
                
                if references:
                    with st.expander("📚 知识库引用来源"):
                        for i, ref in enumerate(references):
                            st.markdown(f"**片段 {i+1}：**")
                            st.text(ref[:500] + "..." if len(ref) > 500 else ref)
                
        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_content, 
            "tokens": total_tokens, 
            "cost": cost,
            "refs": references
        })

with col2:
    st.subheader("📊 实时监控面板")
    
    # 手动刷新数据的按钮
    if st.button("🔄 刷新实时与历史数据", use_container_width=True):
        st.rerun()

    try:
        # 1. 获取并展示实时数据卡片
        temp_data = get_sensor_data.invoke({"sensor_type": "temperature"})
        
        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                label="🌡️ 实时温度", 
                value=f"{temp_data.get('value', 'N/A')} {temp_data.get('unit', '')}"
            )
        with m2:
            try:
                humi_data = get_sensor_data.invoke({"sensor_type": "humidity"})
                st.metric(
                    label="💧 实时湿度", 
                    value=f"{humi_data.get('value', 'N/A')} {humi_data.get('unit', '')}"
                )
            except Exception:
                st.metric(label="💧 实时湿度", value="N/A")
        
        st.write("---")
        
        # 2. 获取历史数据并画图
        st.write("### 近1小时温度趋势摘要")
        hist_temp = query_history.invoke({"sensor_type": "temperature", "hours": 1})
        
        if hist_temp.get("data_points", 0) > 0:
            df_temp = pd.DataFrame({
                "指标": ["最大值", "平均值", "最小值"],
                "温度 (°C)": [hist_temp["max"], hist_temp["mean"], hist_temp["min"]]
            }).set_index("指标")
            
            st.bar_chart(df_temp, horizontal=False)
            
            if hist_temp.get("is_alert"):
                st.error(f"⚠️ 温度异常！当前区间: {hist_temp['min']} - {hist_temp['max']} °C")
            else:
                st.success(f"✅ 温度正常，平均: {hist_temp['mean']} °C")
        else:
            st.info("暂无足够的历史数据生成图表，多发几次数据即可。")

    except Exception as e:
        st.error(f"右侧面板获取数据失败: {e}")