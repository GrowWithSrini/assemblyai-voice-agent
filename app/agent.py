"""The conversational agent, built as a LangGraph state graph.

Graph (one LLM node today):

    START ─▶ call_model ─▶ END

Conversation memory is held server-side by a checkpointer keyed on the browser's
``session_id`` (the ``thread_id``), so the client only sends the new user message
each turn — not the whole history.

Tool calling: the free model (``qwen3.5-4b-32k-fast``) reports ``tools=False``, so
there is no ToolNode yet. On a tool-capable model (``claude-haiku-4-5-20251001``,
``gemini-2.5-flash-lite``, ...) add one where marked below and the graph becomes a
real ReAct agent.

``MemorySaver`` is in-process: fine for a single instance, but a multi-instance
deployment needs a shared checkpointer (e.g. ``langgraph-checkpoint-postgres``) or
sticky sessions.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI

from .assemblyai import AssemblyAIError
from .settings import Settings, get_settings

SYSTEM_PROMPT = (
    "You are a helpful voice assistant in a live spoken conversation. "
    "Answer in one or two short sentences of plain spoken language. "
    "No markdown, no lists, no emoji. If the user's message seems cut off "
    "or unclear, ask a brief clarifying question."
)


@lru_cache
def _build_graph(settings: Settings):
    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=f"{settings.llm_base}/v1",  # LLM Gateway is OpenAI-compatible
        api_key=settings.api_key or "missing",
        max_tokens=settings.llm_max_tokens,
        timeout=30,
        max_retries=0,  # ret/backoff is handled here so we can surface retry_after
    )

    def call_model(state: MessagesState) -> dict:
        reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    # ── tool calling (needs a tool-capable model) ─────────────────────────────
    # from langgraph.prebuilt import ToolNode, tools_condition
    # llm = llm.bind_tools(TOOLS)
    # builder.add_node("tools", ToolNode(TOOLS))
    # builder.add_conditional_edges("call_model", tools_condition)
    # builder.add_edge("tools", "call_model")
    # ─────────────────────────────────────────────────────────────────────────
    return builder.compile(checkpointer=MemorySaver())


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "rate limit" in str(exc).lower()


async def run_agent(session_id: str, message: str) -> str:
    """Advance the conversation ``session_id`` by one user turn; return the reply text."""
    settings = get_settings()
    if not settings.api_key:
        raise AssemblyAIError(500, "ASSEMBLYAI_API_KEY is not set on the server")

    graph = _build_graph(settings)
    config = {"configurable": {"thread_id": session_id}}
    try:
        result = await graph.ainvoke({"messages": [HumanMessage(message)]}, config)
    except Exception as exc:  # noqa: BLE001 — normalize upstream failures for the client
        if _is_rate_limit(exc):
            raise AssemblyAIError(
                429,
                "Free LLM Gateway model is rate limited (~2 req / 50 s). "
                "Wait ~50s or enable paid model access.",
                retry_after=50,
            ) from exc
        raise AssemblyAIError(502, f"agent error: {exc}") from exc

    last = result["messages"][-1]
    if isinstance(last, AIMessage) and isinstance(last.content, str):
        return last.content.strip()
    return str(getattr(last, "content", last)).strip()
