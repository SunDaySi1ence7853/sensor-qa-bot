import streamlit as st
import pandas as pd
import json
from src.agent import get_sensor_agent
from src.tools.sensor_tools import get_sensor_data, query_history
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ----------------- 页面基础配置 -----------------
st.set_page_config(page_title="传感器 QA Bot", page_icon="🤖", layout="wide")

@st.cache_resource
def load_agent():
    return get_sensor_agent()

agent = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 工业传感器监控与诊断助手")
st.caption("基于 LangGraph + DeepSeek + ESP32 实时串口数据")

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
                if msg.get("tokens", 0) > 0:
                    st.caption(f"🔍 Token消耗: {msg['tokens']} | 💰 估算费用: ¥{msg['cost']:.4f}")
                if msg.get("refs"):
                    with st.expander("📚 知识库引用来源"):
                        for i, ref in enumerate(msg["refs"]):
                            st.markdown(f"**片段 {i+1}：**")
                            st.text(ref[:500] + "..." if len(ref) > 500 else ref)

    if prompt := st.chat_input("问问 Agent：现在温度正常吗？或者问硬件规格"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.status("Agent 正在思考与调用工具...", expanded=True) as status:
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in st.session_state.messages
                ]
                
                response = agent.invoke({"messages": history})
                
                # --- 核心修改：ReAct 过程可视化 ---
                st.write("### 🔄 推理与行动 过程")
                # 只遍历新产生的消息（跳过传进去的 history）
                new_messages = response["messages"][len(history):]
                
                for msg in new_messages:
                    # 1. 如果是 AI 决定调用工具
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc.get("name")
                            args = tc.get("args")
                            st.write(f"**🧠 决策：调用工具 `{tool_name}`**")
                            with st.expander(f"传入参数"):
                                st.json(args)
                                
                    # 2. 如果是工具返回了结果
                    elif isinstance(msg, ToolMessage):
                        st.write(f"**📋 结果：`{msg.name}` 返回**")
                        try:
                            # 尝试把 string 解析成 json 好看一点
                            content_json = json.loads(msg.content)
                            with st.expander("工具返回数据"):
                                st.json(content_json)
                        except:
                            with st.expander("工具返回数据"):
                                st.text(msg.content[:500])
                                
                status.update(label="Agent 执行完成！", state="complete", expanded=False)
            
            # 提取最后回复与统计信息
            ai_msg = response["messages"][-1]
            ai_content = ai_msg.content
            
            total_tokens = 0
            cost = 0.0
            if hasattr(ai_msg, 'usage_metadata') and ai_msg.usage_metadata:
                usage = ai_msg.usage_metadata
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                cost = (input_tokens * 0.001 + output_tokens * 0.002) / 1000
            
            references = []
            for msg in response["messages"]:
                if isinstance(msg, ToolMessage) and msg.name == "search_manual":
                    try:
                        ref_data = json.loads(msg.content)
                        if "content" in ref_data:
                            references.append(ref_data["content"])
                        elif "documents" in ref_data:
                            references.extend(ref_data["documents"])
                    except:
                        references.append(msg.content)

            # 输出最终回复
            message_placeholder.markdown(ai_content)
            
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