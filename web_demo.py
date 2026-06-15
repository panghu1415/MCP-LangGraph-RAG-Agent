from __future__ import annotations

import requests
import streamlit as st


st.set_page_config(page_title="智能出行 Agent", layout="wide")

st.title("智能出行规划 Agent")
st.caption("MCP + LangGraph + RAG：天气、路线、攻略、行程、预算、保存一体化演示")

api_base = st.sidebar.text_input("FastAPI 地址", "http://127.0.0.1:8001")
thread_id = st.sidebar.text_input("thread_id", "demo-user")
user_id = st.sidebar.text_input("user_id", "demo-user")
auto_save = st.sidebar.checkbox("自动保存行程", value=False)
request_timeout = st.sidebar.slider("请求超时秒数", min_value=60, max_value=300, value=180, step=30)

default_prompt = "我周末想从杭州去上海迪士尼玩两天，帮我规划路线、预算和注意事项，并结合天气给建议。"
message = st.text_area("输入出行需求", default_prompt, height=120)

if st.button("生成方案", type="primary"):
    with st.spinner("Agent 正在拆解任务并调用工具..."):
        try:
            response = requests.post(
                f"{api_base.rstrip('/')}/chat",
                json={"message": message, "thread_id": thread_id, "user_id": user_id, "auto_save": auto_save},
                timeout=request_timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("无法连接 FastAPI 后端。请先启动：python -m uvicorn app:app --host 127.0.0.1 --port 8001")
            st.stop()
        except requests.exceptions.Timeout:
            st.error(f"后端处理超过 {request_timeout} 秒。可以调大左侧超时秒数，或先关闭其他旧后端进程后重试。")
            st.stop()
        except requests.exceptions.RequestException as exc:
            st.error(f"后端请求失败：{exc}")
            st.stop()

    st.markdown(data["answer"])
    if data.get("saved_file"):
        st.success(f"已保存：{data['saved_file']}")

    with st.expander("工具调用 Trace"):
        st.json(data["tool_traces"])
