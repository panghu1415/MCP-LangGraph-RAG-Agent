from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from config import settings


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower())
    return set(words)


CITY_TERMS = ["上海迪士尼", "上海", "北京", "杭州", "成都", "广州", "深圳", "南京", "苏州", "西安", "重庆"]


def split_markdown(text: str, max_chars: int = 700) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n(?=# )", text) if block.strip()]
    chunks: list[str] = []
    for block in blocks:
        paragraphs = [p.strip() for p in block.split("\n\n") if p.strip()]
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) < max_chars:
                current = f"{current}\n\n{paragraph}".strip()
            else:
                if current:
                    chunks.append(current)
                current = paragraph
        if current:
            chunks.append(current)
    return chunks


def load_knowledge_chunks(path: Path | None = None) -> list[str]:
    knowledge_file = path or settings.knowledge_file
    if not knowledge_file.exists():
        return []
    return split_markdown(knowledge_file.read_text(encoding="utf-8"))


@dataclass
class SearchResult:
    content: str
    score: float
    source: str = "unknown"


class TravelKnowledgeBase(Protocol):
    backend_name: str

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        ...

    def answer_context(self, query: str, top_k: int = 3) -> str:
        ...


class LocalTravelKnowledgeBase:
    """Dependency-free retriever used as fallback and offline demo mode."""

    backend_name = "local_keyword"

    def __init__(self, knowledge_file: Path | None = None) -> None:
        self.knowledge_file = knowledge_file or settings.knowledge_file
        self._chunks = load_knowledge_chunks(self.knowledge_file)
        self._chunk_tokens = [tokenize(chunk) for chunk in self._chunks]

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        preferred_match = re.search(r"目的地[:：]\s*([\u4e00-\u9fff]+)", query)
        preferred_city = preferred_match.group(1) if preferred_match else ""

        results: list[SearchResult] = []
        for chunk, tokens in zip(self._chunks, self._chunk_tokens):
            overlap = len(query_tokens & tokens)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(max(len(tokens), 1))
            title = chunk.splitlines()[0] if chunk.splitlines() else ""
            for city in CITY_TERMS:
                if preferred_city and city != preferred_city:
                    continue
                if city in query and city in title:
                    score += 4.0
                elif city in query and city in chunk:
                    score += 1.5
            results.append(SearchResult(content=chunk, score=round(score, 4), source=str(self.knowledge_file)))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def answer_context(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "知识库没有检索到直接相关资料，可根据通用出行经验回答，并提示用户核验官方信息。"
        return format_search_results(results)


class DashScopeEmbeddingClient:
    """DashScope text embedding wrapper.

    The SDK response format has changed across versions, so extraction is kept
    defensive. The project only needs a tiny interface: embed one text or a
    batch of texts.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.dashscope_api_key
        self.model = model or settings.embedding_model
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for DashScope embedding.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import dashscope

        dashscope.api_key = self.api_key
        response = dashscope.TextEmbedding.call(model=self.model, input=texts)
        embeddings = self._extract_embeddings(response)
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @staticmethod
    def _extract_embeddings(response: object) -> list[list[float]]:
        if hasattr(response, "output"):
            output = response.output
        else:
            output = response.get("output", {})  # type: ignore[union-attr]

        embeddings = output.get("embeddings", []) if isinstance(output, dict) else getattr(output, "embeddings", [])
        vectors: list[list[float]] = []
        for item in embeddings:
            if isinstance(item, dict):
                vectors.append(item["embedding"])
            else:
                vectors.append(item.embedding)
        return vectors


class MilvusTravelKnowledgeBase:
    """Milvus-backed semantic retriever for production-like RAG."""

    backend_name = "milvus_dashscope"

    def __init__(self, embedding_client: DashScopeEmbeddingClient | None = None) -> None:
        self.embedding_client = embedding_client or DashScopeEmbeddingClient()
        self.collection = self._connect_collection()

    def _connect_collection(self):
        from pymilvus import Collection, connections, utility

        connections.connect(alias="default", host=settings.milvus_host, port=settings.milvus_port)
        if not utility.has_collection(settings.milvus_collection):
            raise RuntimeError(
                f"Milvus collection '{settings.milvus_collection}' does not exist. "
                "Run: python scripts/ingest_knowledge.py"
            )
        collection = Collection(settings.milvus_collection)
        collection.load()
        return collection

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        vector = self.embedding_client.embed_query(query)
        search_params = {"metric_type": settings.milvus_metric_type, "params": {"ef": 64}}
        hits = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "source"],
        )

        results: list[SearchResult] = []
        for hit in hits[0]:
            entity = hit.entity
            results.append(
                SearchResult(
                    content=entity.get("content"),
                    source=entity.get("source"),
                    score=round(float(hit.score), 4),
                )
            )
        return results

    def answer_context(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "Milvus 知识库没有检索到直接相关资料，请补充旅游资料后重新入库。"
        return format_search_results(results)


def format_search_results(results: list[SearchResult]) -> str:
    return "\n\n---\n\n".join(
        f"[score={result.score} source={result.source}]\n{result.content}" for result in results
    )


def create_milvus_collection(drop_old: bool = False):
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

    connections.connect(alias="default", host=settings.milvus_host, port=settings.milvus_port)
    if utility.has_collection(settings.milvus_collection):
        if drop_old:
            utility.drop_collection(settings.milvus_collection)
        else:
            collection = Collection(settings.milvus_collection)
            collection.load()
            return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
    ]
    schema = CollectionSchema(fields=fields, description="Travel knowledge chunks for Agent RAG")
    collection = Collection(settings.milvus_collection, schema=schema)

    index_params = {
        "index_type": settings.milvus_index_type,
        "metric_type": settings.milvus_metric_type,
        "params": {"M": 16, "efConstruction": 200},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def ingest_chunks_to_milvus(chunks: list[str], source: str, drop_old: bool = False) -> int:
    if not chunks:
        return 0
    collection = create_milvus_collection(drop_old=drop_old)
    embedding_client = DashScopeEmbeddingClient()
    vectors = embedding_client.embed_documents(chunks)

    ids = [
        hashlib.sha1(f"{source}:{index}:{chunk}".encode("utf-8")).hexdigest()
        for index, chunk in enumerate(chunks)
    ]
    sources = [source for _ in chunks]
    collection.insert([ids, chunks, sources, vectors])
    collection.flush()
    collection.load()
    return len(chunks)


def get_knowledge_base() -> TravelKnowledgeBase:
    if settings.rag_backend == "local":
        return LocalTravelKnowledgeBase()
    if settings.rag_backend == "milvus":
        return MilvusTravelKnowledgeBase()

    try:
        if settings.dashscope_api_key:
            return MilvusTravelKnowledgeBase()
    except Exception:
        pass
    return LocalTravelKnowledgeBase()
