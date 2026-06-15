from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agent_service import agent_service
from mcp_client_service import list_mcp_tools
from persistence import preference_store
from schemas import ChatRequest, ChatResponse, ContextResponse, ResetRequest


app = FastAPI(
    title="MCP + LangGraph + RAG 智能出行规划 Agent",
    description="面向出行场景的多工具协同 Agent：天气、路线、攻略 RAG、行程、预算和保存。",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    result = await agent_service.chat(
        message=request.message,
        thread_id=request.thread_id,
        auto_save=request.auto_save,
        user_id=request.user_id,
    )
    return ChatResponse(
        answer=result["answer"],
        thread_id=request.thread_id,
        user_id=request.user_id,
        tool_traces=result.get("tool_traces", []),
        saved_file=result.get("saved_file"),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        yield _sse("step", {"message": "Agent 已接收请求，开始规划。"})
        task = asyncio.create_task(
            agent_service.chat(
                message=request.message,
                thread_id=request.thread_id,
                auto_save=request.auto_save,
                user_id=request.user_id,
            )
        )
        while not task.done():
            yield _sse("step", {"message": "正在调用天气、交通、POI、RAG 和大模型工具..."})
            await asyncio.sleep(3)
        result = await task
        for trace in result.get("tool_traces", []):
            yield _sse("tool", trace.model_dump())
        yield _sse(
            "message",
            {
                "answer": result["answer"],
                "thread_id": request.thread_id,
                "user_id": request.user_id,
                "saved_file": result.get("saved_file"),
            },
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/reset")
def reset_chat(request: ResetRequest) -> dict[str, str | bool]:
    preference_store.reset_conversation(request.thread_id)
    return {"success": True, "message": "对话已重置", "thread_id": request.thread_id}


@app.get("/chat/context/{user_id}", response_model=ContextResponse)
def get_context(user_id: str, thread_id: str = "default") -> ContextResponse:
    pref = preference_store.get_user_pref(user_id)
    state = preference_store.get_conversation_state(thread_id)
    return ContextResponse(user_id=user_id, user_preferences=pref.to_dict(), conversation_state=state)


@app.get("/mcp/tools")
async def mcp_tools() -> dict:
    tools = await list_mcp_tools()
    return {"count": len(tools), "tools": tools}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
