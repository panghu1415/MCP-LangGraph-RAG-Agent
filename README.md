# MCP + LangGraph + RAG 智能出行规划 Agent

这是一个适合简历和面试演示的 AI 应用项目：用户输入自然语言出行需求后，系统会完成意图抽取、天气查询、路线规划、旅游知识库检索、行程生成、预算估算和结果保存。

项目重点不是单点问答，而是展示 Agent 如何把复杂任务拆成多步，并通过 MCP 工具和 RAG 知识库协同完成。

## 技术栈

- Python 3.10+
- FastAPI：后端服务接口
- Streamlit：前端演示页面
- LangGraph：Agent 工作流编排
- MCP：天气、路线、RAG、行程、预算、文件保存工具服务
- RAG：本地轻量检索，预留 Milvus + DashScope Embedding 生产化扩展
- OpenWeather / AMap：真实外部 API，可选
- 12306 / 航班 / 酒店 API：可选，支持配置真实服务，默认 fallback 演示
- SQLite：用户偏好和对话状态持久化

## 项目结构

```text
.
├── app.py
├── agent_service.py
├── web_demo.py
├── config.py
├── schemas.py
├── servers_config.json
├── agent_prompts.txt
├── requirements.txt
├── docker-compose.yml
├── data/
│   └── travel_knowledge.md
├── rag/
│   └── vector_store.py
├── tools/
│   └── local_tools.py
└── mcp_servers/
    ├── weather_server.py
    ├── route_server.py
    ├── rag_server.py
    ├── planner_server.py
    ├── budget_server.py
    └── write_server.py
```

## 快速启动

1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. 配置环境变量：

```bash
copy .env.example .env
```

在 `.env` 中填写：

```env
DASHSCOPE_API_KEY=你的DashScope Key
DASHSCOPE_BASE_HTTP_API_URL=https://dashscope.aliyuncs.com/api/v1
MODEL=qwen-plus
USE_LLM=true
OPENWEATHER_API_KEY=你的OpenWeather Key
AMAP_API_KEY=你的高德Web服务Key
TRAIN_API_URL=可选的火车票查询服务地址
TRAIN_API_KEY=可选
FLIGHT_API_URL=可选的航班查询服务地址
FLIGHT_API_KEY=可选
HOTEL_API_URL=可选的酒店价格服务地址
HOTEL_API_KEY=可选
```

没有 API Key 也可以运行，系统会使用 mock/fallback 数据完成演示。有 DashScope Key 时，最终回答会调用 `qwen-plus` 生成。

3. 启动 FastAPI：

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

4. 另开一个终端启动 Streamlit：

```bash
python -m streamlit run web_demo.py --server.address 127.0.0.1 --server.port 8501
```

5. 打开前端页面：

```text
http://127.0.0.1:8501
```

6. 或者直接调用接口：

```bash
curl -X POST http://127.0.0.1:8001/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"我周末想从杭州去上海迪士尼玩两天，帮我规划路线、预算和注意事项\",\"thread_id\":\"demo\",\"auto_save\":true}"
```

## 新增接口

SSE 流式接口：

```bash
curl -N -X POST http://127.0.0.1:8001/chat/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"明天从杭州去上海迪士尼，公司报销，优先高铁\",\"thread_id\":\"demo\",\"user_id\":\"u1\"}"
```

查询用户上下文：

```bash
curl "http://127.0.0.1:8001/chat/context/u1?thread_id=demo"
```

重置对话：

```bash
curl -X POST http://127.0.0.1:8001/chat/reset ^
  -H "Content-Type: application/json" ^
  -d "{\"thread_id\":\"demo\"}"
```

## 智能交通调度

系统会按下面规则选择交通方式：

```text
用户明确自驾 -> 高德自驾路线
用户明确飞机 / 偏好飞机 -> 航班查询
用户明确高铁/火车 / 偏好火车 -> 12306
距离 > 800km -> 航班
其他 -> 12306 / 高铁
```

当前 `TRAIN_API_URL`、`FLIGHT_API_URL`、`HOTEL_API_URL` 为空时会返回结构化 fallback 数据，并明确标注来源。填入你自己的服务地址和 Key 后，会优先调用真实 API。

## 真实 RAG：Milvus + DashScope Embedding

当前项目支持三种 RAG 模式：

```text
RAG_BACKEND=auto    # 默认，优先 Milvus，失败后回退本地关键词检索
RAG_BACKEND=milvus  # 强制使用 Milvus + DashScope Embedding
RAG_BACKEND=local   # 强制使用本地轻量检索
```

1. 启动 Milvus：

```bash
docker compose up -d
```

如果提示 `docker` 不是可识别命令，说明当前机器还没有安装 Docker Desktop 或 Docker 没加入 PATH。安装并启动 Docker Desktop 后再执行上面的命令。

2. 在 `.env` 中配置：

```env
DASHSCOPE_API_KEY=你的通义千问DashScope Key
EMBEDDING_MODEL=text-embedding-v1
EMBEDDING_DIM=1536
RAG_BACKEND=milvus
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION=travel_knowledge
```

3. 将示例知识库写入 Milvus：

```bash
python scripts/ingest_knowledge.py --drop-old
```

4. 启动后端后，`travel_rag_search` 会从 Milvus 做语义检索；如果使用 `RAG_BACKEND=auto`，Milvus 不可用时会自动回退到本地检索，保证演示不中断。

## MCP 工具发现

项目已经提供 `MultiServerMCPClient` 工具发现入口：

```bash
curl http://127.0.0.1:8001/mcp/tools
```

该接口会读取 `servers_config.json`，通过 `MultiServerMCPClient` 启动本地 MCP Servers 并列出工具。当前可发现天气、路线、RAG、POI、交通、酒店、预算、写文件等工具。

如果希望让模型自主选择 MCP 工具，而不是走固定 LangGraph 节点顺序，可以在 `.env` 中开启：

```env
USE_MCP=true
```

开启后，`agent_service.py` 会优先使用 `create_react_agent + MultiServerMCPClient`。如果本地依赖、模型或 MCP 子进程异常，系统会在 `tool_traces` 中记录失败原因，并自动回退到稳定工作流。

## 真实天气和地图 API

`.env` 中配置下面两个 Key 后，工具会优先调用真实服务：

```env
OPENWEATHER_API_KEY=你的OpenWeather Key
AMAP_API_KEY=你的高德Web服务Key
```

天气工具流程：

```text
城市名 -> OpenWeather Geocoding -> 经纬度 -> Current Weather -> 天气建议
```

路线工具流程：

```text
出发地/目的地 -> 高德地理编码 -> 自驾路线或公共交通规则建议
```

如果 API 失败、Key 缺失或没有检索到结果，系统会返回 fallback 结果，并在 `source` 字段中标记失败原因。

## 简历写法

项目名称：基于 MCP + LangGraph + RAG 的多工具协同智能出行 Agent

项目描述：构建面向智能出行场景的多工具协同 Agent 系统，基于 LangGraph 实现任务编排，通过 MCP 协议封装天气、路线、RAG、行程、预算和文件保存等工具，并使用 FastAPI 提供服务化接口，实现从自然语言需求到完整出行方案生成的一体化流程。

核心职责：

- 设计 Agent 工作流，将复杂出行需求拆解为意图抽取、天气查询、路线规划、知识检索、行程生成和预算估算。
- 使用 MCP Server 标准化封装外部工具能力，便于工具独立扩展和跨 Agent 复用。
- 构建旅游 RAG 知识库，将景点攻略、出行政策和预算说明作为 Agent 的知识增强工具。
- 使用 FastAPI 和 Streamlit 完成服务化接口与可视化演示，支持 thread_id 会话隔离和行程文件保存。

## 面试讲解重点

- Agent 不是直接生成答案，而是先拆任务，再调用多个工具，最后整合结果。
- RAG 不是独立问答系统，而是 Agent 可调用的知识增强工具。
- MCP Server 把工具协议化，后续可以替换为真实天气、地图、酒店、机票、公司差旅系统。
- 没有外部 Key 时能演示完整流程，有 Key 后能逐步替换 mock 能力。


cd "D:\python object\AgentDemo"
python -m uvicorn app:app --host 127.0.0.1 --port 8001
