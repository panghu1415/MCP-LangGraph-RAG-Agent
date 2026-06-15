from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(override=True)


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_http_api_url: str = os.getenv(
        "DASHSCOPE_BASE_HTTP_API_URL",
        "https://dashscope.aliyuncs.com/api/v1",
    )
    model: str = os.getenv("MODEL", "qwen-plus")
    use_llm: bool = os.getenv("USE_LLM", "true").lower() == "true"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    openweather_api_key: str = os.getenv("OPENWEATHER_API_KEY", "")
    amap_api_key: str = os.getenv("AMAP_API_KEY", "")
    train_api_url: str = os.getenv("TRAIN_API_URL", "")
    train_api_key: str = os.getenv("TRAIN_API_KEY", "")
    flight_api_url: str = os.getenv("FLIGHT_API_URL", "")
    flight_api_key: str = os.getenv("FLIGHT_API_KEY", "")
    hotel_api_url: str = os.getenv("HOTEL_API_URL", "")
    hotel_api_key: str = os.getenv("HOTEL_API_KEY", "")

    rag_backend: str = os.getenv("RAG_BACKEND", "auto").lower()
    milvus_host: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "travel_knowledge")
    milvus_metric_type: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
    milvus_index_type: str = os.getenv("MILVUS_INDEX_TYPE", "HNSW")

    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "output"))
    sqlite_db_path: Path = Path(os.getenv("SQLITE_DB_PATH", "data/travel_agent.db"))
    knowledge_file: Path = Path(os.getenv("KNOWLEDGE_FILE", "data/travel_knowledge.md"))
    use_mcp: bool = os.getenv("USE_MCP", "false").lower() == "true"
    mcp_servers_config: Path = Path(os.getenv("MCP_SERVERS_CONFIG", "servers_config.json"))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "12"))


settings = Settings()
