from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from config import settings
from persistence import preference_store
from schemas import ToolTrace
from tools.local_tools import (
    budget_estimate,
    hotel_price_search,
    infer_trip_slots,
    itinerary_plan,
    route_plan,
    save_itinerary,
    smart_transport_dispatch,
    to_pretty_json,
    travel_rag_search,
    travel_poi_search,
    weather_query,
)


class AgentState(TypedDict, total=False):
    message: str
    thread_id: str
    user_id: str
    auto_save: bool
    preference: dict[str, Any]
    slots: dict[str, Any]
    weather: dict[str, Any]
    route: dict[str, Any]
    transport: dict[str, Any]
    rag: dict[str, Any]
    poi: dict[str, Any]
    hotel: dict[str, Any]
    itinerary: dict[str, Any]
    budget: dict[str, Any]
    answer: str
    saved_file: str | None
    tool_traces: list[ToolTrace]


@dataclass
class TravelAgentService:
    """LangGraph-style travel planner Agent.

    The MVP uses deterministic tool orchestration so it is demoable without an
    LLM key. The graph boundary is where a ReAct Agent or MCP client can be
    swapped in for production.
    """

    prompt_file: Path = Path("agent_prompts.txt")
    sessions: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.system_prompt = self.prompt_file.read_text(encoding="utf-8")
        self.graph = self._build_graph()

    def _trace(self, state: AgentState, name: str, tool_input: dict[str, Any], output: Any) -> None:
        state.setdefault("tool_traces", []).append(ToolTrace(name=name, input=tool_input, output=output))

    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        graph = StateGraph(AgentState)
        graph.add_node("understand", self._understand)
        graph.add_node("preferences", self._preferences)
        graph.add_node("weather", self._weather)
        graph.add_node("route", self._route)
        graph.add_node("transport", self._transport)
        graph.add_node("rag", self._rag)
        graph.add_node("poi", self._poi)
        graph.add_node("hotel", self._hotel)
        graph.add_node("planner", self._planner)
        graph.add_node("budget", self._budget)
        graph.add_node("compose", self._compose)
        graph.add_node("save", self._save)

        graph.set_entry_point("understand")
        graph.add_edge("understand", "preferences")
        graph.add_edge("preferences", "weather")
        graph.add_edge("weather", "route")
        graph.add_edge("route", "transport")
        graph.add_edge("transport", "rag")
        graph.add_edge("rag", "poi")
        graph.add_edge("poi", "hotel")
        graph.add_edge("hotel", "planner")
        graph.add_edge("planner", "budget")
        graph.add_edge("budget", "compose")
        graph.add_conditional_edges(
            "compose",
            lambda state: "save" if state.get("auto_save") else "end",
            {"save": "save", "end": END},
        )
        graph.add_edge("save", END)
        return graph.compile()

    async def _understand(self, state: AgentState) -> AgentState:
        slots = infer_trip_slots(state["message"])
        state["slots"] = slots
        self._trace(state, "intent_extract", {"message": state["message"]}, slots)
        return state

    async def _preferences(self, state: AgentState) -> AgentState:
        output = preference_store.update_from_message(state["user_id"], state["message"])
        state["preference"] = output["preference"]
        self._trace(
            state,
            "preference_update",
            {"user_id": state["user_id"], "message": state["message"]},
            output,
        )
        if state["slots"].get("origin") == "出发地" and output["preference"].get("home_location"):
            state["slots"]["origin"] = output["preference"]["home_location"]
        return state

    async def _weather(self, state: AgentState) -> AgentState:
        city = state["slots"]["destination"].replace("迪士尼", "")
        output = await weather_query(city)
        state["weather"] = output
        self._trace(state, "weather_query", {"city": city}, output)
        return state

    async def _route(self, state: AgentState) -> AgentState:
        slots = state["slots"]
        output = await route_plan(slots["origin"], slots["destination"])
        state["route"] = output
        self._trace(
            state,
            "route_plan",
            {"origin": slots["origin"], "destination": slots["destination"]},
            output,
        )
        return state

    async def _transport(self, state: AgentState) -> AgentState:
        slots = state["slots"]
        output = await smart_transport_dispatch(
            slots["origin"],
            slots["destination"],
            state["message"],
            state.get("preference"),
        )
        state["transport"] = output
        self._trace(
            state,
            "smart_transport_dispatch",
            {
                "origin": slots["origin"],
                "destination": slots["destination"],
                "preference": state.get("preference"),
            },
            output,
        )
        return state

    async def _rag(self, state: AgentState) -> AgentState:
        output = travel_rag_search(state["message"])
        state["rag"] = output
        self._trace(state, "travel_rag_search", {"query": state["message"], "top_k": 3}, output)
        return state

    async def _hotel(self, state: AgentState) -> AgentState:
        slots = state["slots"]
        hotel_style = (state.get("preference") or {}).get("hotel_prefer", slots["style"]).lower()
        output = await hotel_price_search(slots["destination"], slots["days"], hotel_style, state.get("poi"))
        state["hotel"] = output
        self._trace(
            state,
            "hotel_price_search",
            {"destination": slots["destination"], "days": slots["days"], "style": hotel_style},
            output,
        )
        return state

    async def _poi(self, state: AgentState) -> AgentState:
        slots = state["slots"]
        output = await travel_poi_search(slots["destination"], state["message"])
        state["poi"] = output
        self._trace(
            state,
            "travel_poi_search",
            {"destination": slots["destination"], "interests": state["message"]},
            output,
        )
        return state

    async def _planner(self, state: AgentState) -> AgentState:
        slots = state["slots"]
        output = itinerary_plan(
            slots["destination"],
            slots["days"],
            slots["interests"],
            state["rag"]["context"],
            state.get("poi"),
        )
        state["itinerary"] = output
        self._trace(
            state,
            "itinerary_plan",
            {
                "destination": slots["destination"],
                "days": slots["days"],
                "interests": slots["interests"],
            },
            output,
        )
        return state

    async def _budget(self, state: AgentState) -> AgentState:
        slots = state["slots"]
        output = budget_estimate(
            slots["destination"],
            slots["days"],
            slots["people"],
            slots["style"],
            state.get("transport"),
            state.get("hotel"),
        )
        state["budget"] = output
        self._trace(
            state,
            "budget_estimate",
            {
                "destination": slots["destination"],
                "days": slots["days"],
                "people": slots["people"],
                "style": slots["style"],
            },
            output,
        )
        return state

    async def _compose(self, state: AgentState) -> AgentState:
        template_answer = self._format_answer(state)
        if not settings.use_llm or not settings.dashscope_api_key:
            state["answer"] = template_answer
            self._trace(
                state,
                "llm_compose",
                {"model": settings.model, "enabled": settings.use_llm},
                {"source": "template_fallback", "reason": "USE_LLM=false or DASHSCOPE_API_KEY is empty"},
            )
            return state

        try:
            state["answer"] = await self._generate_with_qwen(state, template_answer)
            self._trace(
                state,
                "llm_compose",
                {"model": settings.model, "enabled": True},
                {"source": "DashScope", "model": settings.model},
            )
        except Exception as exc:
            state["answer"] = template_answer
            self._trace(
                state,
                "llm_compose",
                {"model": settings.model, "enabled": True},
                {"source": "template_fallback", "reason": str(exc)},
            )
        return state

    async def _save(self, state: AgentState) -> AgentState:
        output = save_itinerary(state["answer"])
        state["saved_file"] = output["path"]
        self._trace(state, "save_itinerary", {"filename": output["filename"]}, output)
        return state

    async def chat(
        self,
        message: str,
        thread_id: str = "default",
        auto_save: bool = False,
        user_id: str = "default-user",
    ) -> AgentState:
        self.sessions.setdefault(thread_id, []).append({"role": "user", "content": message})
        initial: AgentState = {
            "message": message,
            "thread_id": thread_id,
            "user_id": user_id,
            "auto_save": auto_save or ("保存" in message),
            "tool_traces": [],
            "saved_file": None,
        }
        if settings.use_mcp:
            result = await self._run_with_mcp_react(initial)
        elif self.graph is not None:
            result = await self.graph.ainvoke(initial)
        else:
            result = await self._run_without_langgraph(initial)
        self.sessions[thread_id].append({"role": "assistant", "content": result["answer"]})
        preference_store.save_conversation_state(
            thread_id,
            user_id,
            {
                "slots": result.get("slots", {}),
                "preference": result.get("preference", {}),
                "transport": result.get("transport", {}),
                "saved_file": result.get("saved_file"),
            },
        )
        return result

    async def _run_with_mcp_react(self, state: AgentState) -> AgentState:
        """Run a model-driven ReAct Agent over MCP tools.

        The deterministic workflow remains the fallback because MCP stdio
        servers, LangGraph prebuilt APIs, or local model adapters may differ by
        environment during demos.
        """
        try:
            if not settings.dashscope_api_key:
                raise RuntimeError("DASHSCOPE_API_KEY is empty; MCP ReAct needs an LLM.")
            slots = infer_trip_slots(state["message"])
            state["slots"] = slots
            self._trace(state, "intent_extract", {"message": state["message"]}, slots)

            pref_output = preference_store.update_from_message(state["user_id"], state["message"])
            state["preference"] = pref_output["preference"]
            self._trace(
                state,
                "preference_update",
                {"user_id": state["user_id"], "message": state["message"]},
                pref_output,
            )
            if slots.get("origin") == "出发地" and state["preference"].get("home_location"):
                slots["origin"] = state["preference"]["home_location"]

            agent = await self._build_mcp_react_agent()
            response = await agent.ainvoke(
                {
                    "messages": [
                        (
                            "user",
                            self._build_mcp_react_prompt(
                                state["message"],
                                slots,
                                state.get("preference", {}),
                                state.get("auto_save", False),
                            ),
                        )
                    ]
                }
            )
            self._apply_react_response(state, response)
            if not state.get("answer"):
                raise RuntimeError("MCP ReAct returned empty answer.")
            self._trace(
                state,
                "mcp_react_agent",
                {"enabled": True, "model": settings.model, "tools": "MultiServerMCPClient"},
                {"source": "create_react_agent", "status": "success"},
            )
            if state.get("auto_save") and not state.get("saved_file"):
                state = await self._save(state)
            return state
        except Exception as exc:
            self._trace(
                state,
                "mcp_react_agent",
                {"enabled": True, "model": settings.model, "tools": "MultiServerMCPClient"},
                {"source": "deterministic_fallback", "reason": str(exc)},
            )
            return await self._run_without_langgraph(state)

    async def _build_mcp_react_agent(self) -> Any:
        import dashscope

        from langchain_community.chat_models import ChatTongyi
        from langgraph.prebuilt import create_react_agent
        from mcp_client_service import load_mcp_tools

        tools = await load_mcp_tools()
        if not tools:
            raise RuntimeError("No MCP tools discovered from servers_config.json.")
        dashscope.base_http_api_url = settings.dashscope_base_http_api_url
        model = ChatTongyi(
            model=settings.model,
            dashscope_api_key=settings.dashscope_api_key,
            temperature=0.3,
        )
        return create_react_agent(model, tools, prompt=self._build_mcp_system_prompt())

    def _build_mcp_system_prompt(self) -> str:
        return (
            self.system_prompt
            + "\n\n你现在运行在 ReAct + MCP tools 模式。"
            "你可以自主选择 MCP 工具，但最终答案必须基于工具结果。"
            "优先调用 query_weather、plan_route、dispatch_transport、search_travel_knowledge、"
            "search_travel_poi、search_hotel_price、create_itinerary、estimate_budget。"
            "如果用户要求保存，最后调用 write_itinerary。"
            "不要编造实时票价、开放时间、天气或酒店价格；fallback 数据必须说明是估算。"
        )

    def _build_mcp_react_prompt(
        self,
        message: str,
        slots: dict[str, Any],
        preference: dict[str, Any],
        auto_save: bool,
    ) -> str:
        return (
            "用户原始需求：\n"
            f"{message}\n\n"
            "已抽取槽位和偏好如下，可作为工具参数参考；如果你判断需要修正，可以按用户原意修正后调用工具。\n"
            f"槽位：{to_pretty_json(slots)}\n"
            f"用户偏好：{to_pretty_json(preference)}\n"
            f"是否需要保存：{auto_save}\n\n"
            "请自主决定工具调用顺序，生成中文 Markdown 出行方案。"
            "答案需要包含：总体建议、天气提醒、城际交通选择、分日行程、POI/酒店候选、预算拆分、注意事项。"
        )

    def _apply_react_response(self, state: AgentState, response: dict[str, Any]) -> None:
        messages = response.get("messages", []) if isinstance(response, dict) else []
        tool_calls: dict[str, dict[str, Any]] = {}
        for message in messages:
            for call in getattr(message, "tool_calls", []) or []:
                call_id = call.get("id", "")
                tool_calls[call_id] = {
                    "name": call.get("name", "mcp_tool"),
                    "input": call.get("args", {}),
                }

            if getattr(message, "type", "") == "tool":
                call_id = getattr(message, "tool_call_id", "")
                call = tool_calls.get(call_id, {"name": getattr(message, "name", "mcp_tool"), "input": {}})
                output = self._parse_tool_message_content(getattr(message, "content", ""))
                self._trace(state, call["name"], call["input"], output)
                if call["name"] == "write_itinerary" and isinstance(output, dict):
                    state["saved_file"] = output.get("path")

        for message in reversed(messages):
            content = getattr(message, "content", "")
            if getattr(message, "type", "") == "ai" and isinstance(content, str) and content.strip():
                state["answer"] = content.strip()
                return

    @staticmethod
    def _parse_tool_message_content(content: Any) -> Any:
        if not isinstance(content, str):
            return content
        text = content.strip()
        if not text:
            return ""
        try:
            import json

            return json.loads(text)
        except Exception:
            return text

    async def _run_without_langgraph(self, state: AgentState) -> AgentState:
        for step in [
            self._understand,
            self._preferences,
            self._weather,
            self._route,
            self._transport,
            self._rag,
            self._poi,
            self._hotel,
            self._planner,
            self._budget,
            self._compose,
        ]:
            state = await step(state)
        if state.get("auto_save"):
            state = await self._save(state)
        return state

    def _format_answer(self, state: AgentState) -> str:
        slots = state["slots"]
        budget = state["budget"]
        lines = [
            f"# {slots['origin']} 到 {slots['destination']} {slots['days']} 天游玩方案",
            "",
            "## 总体建议",
            f"- 目的地：{slots['destination']}",
            f"- 人数：{slots['people']} 人",
            f"- 预算档位：{slots['style']}",
            f"- 预计总费用：约 {budget['total']} 元",
            "- 实时票价、开放时间和天气需以官方平台为准。",
            "",
            "## 工具结果整合",
            f"- 天气：{state['weather']['weather']}，温度/体感：{state['weather']['temperature']}，来源：{state['weather']['source']}",
            f"- 路线：{state['route']['duration']}，来源：{state['route']['source']}",
            f"- 城际交通：推荐 {state.get('transport', {}).get('mode', '未判断')}，来源：{state.get('transport', {}).get('detail', {}).get('source', '未调用')}",
            f"- 攻略：已从 {state['rag']['source']} 检索相关资料。",
            f"- POI：已从 {state.get('poi', {}).get('source', '未调用')} 获取景点/酒店/餐厅候选。",
            f"- 酒店：已从 {state.get('hotel', {}).get('source', '未调用')} 获取候选价格。",
            "",
            "## 分日行程",
        ]

        for day in state["itinerary"]["plan"]:
            lines.append(f"### Day {day['day']}：{day['theme']}")
            lines.extend(f"- {item}" for item in day["items"])

        lines.extend(
            [
                "",
                "## 预算拆分",
                *[f"- {name}：约 {value} 元" for name, value in budget["items"].items()],
                "",
                "## 交通与酒店候选",
                f"- 交通模式：{state.get('transport', {}).get('mode', '未判断')}",
                f"- 酒店候选：{', '.join([item.get('name', '') for item in state.get('hotel', {}).get('hotels', [])[:3]]) or '暂无'}",
                "",
                "## 知识库参考片段",
                state["rag"]["context"],
                "",
                "## Agent 调用链",
                "intent_extract -> preference_update -> weather_query -> route_plan -> smart_transport_dispatch -> travel_rag_search -> travel_poi_search -> hotel_price_search -> itinerary_plan -> budget_estimate",
            ]
        )
        if state.get("auto_save"):
            lines[-1] += " -> save_itinerary"
        lines.extend(["", "<details><summary>调试用工具输出</summary>", "", "```json"])
        lines.append(to_pretty_json([trace.model_dump() for trace in state.get("tool_traces", [])]))
        lines.extend(["```", "", "</details>"])
        return "\n".join(lines)

    async def _generate_with_qwen(self, state: AgentState, template_answer: str) -> str:
        import asyncio

        return await asyncio.to_thread(self._call_qwen, state, template_answer)

    def _call_qwen(self, state: AgentState, template_answer: str) -> str:
        import dashscope

        dashscope.api_key = settings.dashscope_api_key
        dashscope.base_http_api_url = settings.dashscope_base_http_api_url
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
                + "\n\n你现在负责最终答案生成。必须基于工具结果，不要编造实时票价、开放时间或天气。输出 Markdown。",
            },
            {
                "role": "user",
                "content": self._build_llm_user_prompt(state, template_answer),
            },
        ]
        try:
            response = dashscope.Generation.call(
                model=settings.model,
                messages=messages,
                result_format="message",
                temperature=0.4,
            )
            content = self._extract_dashscope_content(response)
            if content:
                return content
            raise RuntimeError(f"DashScope returned empty content: {response}")
        except Exception as exc:
            content = self._call_qwen_compatible(messages)
            if content:
                return content
            raise RuntimeError(f"DashScope SDK failed and compatible mode returned empty content: {exc}") from exc

    def _call_qwen_compatible(self, messages: list[dict[str, str]]) -> str:
        import httpx

        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.model,
            "messages": messages,
            "temperature": 0.4,
        }
        with httpx.Client(timeout=60, trust_env=False) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()

    def _build_llm_user_prompt(self, state: AgentState, template_answer: str) -> str:
        compact_state = {
            "user_message": state["message"],
            "slots": state["slots"],
            "preference": state.get("preference"),
            "weather": state["weather"],
            "route": state["route"],
            "transport": state.get("transport"),
            "rag": state["rag"],
            "poi": state.get("poi"),
            "hotel": state.get("hotel"),
            "itinerary": state["itinerary"],
            "budget": state["budget"],
        }
        return (
            "请根据下面的 Agent 工具调用结果，生成一份面向用户的智能出行规划方案。\n"
            "要求：\n"
            "1. 使用中文 Markdown。\n"
            "2. 先给结论，再给分日行程、路线、天气提醒、预算、注意事项。\n"
            "3. 明确说明天气、路线、攻略、预算来自工具结果。\n"
            "4. 不确定或实时变化的信息必须提示用户以官方平台为准。\n"
            "5. 不要输出调试 JSON。\n\n"
            "工具结果 JSON：\n"
            f"{to_pretty_json(compact_state)}\n\n"
            "原模板答案，仅作为结构参考：\n"
            f"{template_answer}"
        )

    @staticmethod
    def _extract_dashscope_content(response: object) -> str:
        if hasattr(response, "output"):
            output = response.output
        else:
            output = response.get("output", {})  # type: ignore[union-attr]

        if isinstance(output, dict):
            choices = output.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "").strip()
            return str(output.get("text", "")).strip()

        choices = getattr(output, "choices", [])
        if choices:
            message = getattr(choices[0], "message", None)
            if isinstance(message, dict):
                return message.get("content", "").strip()
            return str(getattr(message, "content", "")).strip()
        return str(getattr(output, "text", "")).strip()


agent_service = TravelAgentService()
