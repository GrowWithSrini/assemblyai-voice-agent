"""The conversational agent, built as a LangGraph state graph.

Without tools:

    START ─▶ call_model ─▶ END

With tools (``settings.llm_enable_tools``) it is a ReAct loop:

    START ─▶ call_model ─┬─(tool call?)─▶ tools ─▶ call_model
                         └─(else)───────▶ END

`llm.bind_tools(TOOLS)` serializes each tool in ``app/tools.py`` to a JSON-Schema
function definition and sends it in the OpenAI ``tools`` parameter; ``ToolNode`` runs
the calls the model asks for and feeds results back. The free model
(``qwen3.5-4b-32k-fast``) rejects a ``tools`` payload with HTTP 400, so tools default
off for it and on for every other model — override with ``LLM_ENABLE_TOOLS``.

Conversation memory is held server-side by a checkpointer keyed on the browser's
``session_id`` (the ``thread_id``), so the client only sends the new user message
each turn — not the whole history. ``MemorySaver`` is in-process: fine for a single
instance, but a multi-instance deployment needs a shared checkpointer (e.g.
``langgraph-checkpoint-postgres``) or sticky sessions.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .assemblyai import AssemblyAIError
from .settings import Settings, get_settings
from .tools import TOOLS

SYSTEM_PROMPT = (
    "You are a helpful voice assistant in a live spoken conversation. "
    "Answer in one or two short sentences of plain spoken language. "
    "No markdown, no lists, no emoji. "
    "Use a tool when the user asks for the current time or the weather; "
    "otherwise just answer. If the user's message seems cut off or unclear, "
    "ask a brief clarifying question."
)


@lru_cache
def _build_graph(settings: Settings):
    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=f"{settings.llm_base}/v1",  # LLM Gateway is OpenAI-compatible
        api_key=settings.api_key or "missing",
        max_tokens=settings.llm_max_tokens,
        timeout=30,
        max_retries=0,  # retry/backoff handled here so we can surface retry_after
    )

    use_tools = settings.llm_enable_tools
    model = llm.bind_tools(TOOLS) if use_tools else llm

    def call_model(state: MessagesState) -> dict:
        reply = model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")

    if use_tools:
        builder.add_node("tools", ToolNode(TOOLS))
        builder.add_conditional_edges("call_model", tools_condition)  # → "tools" or END
        builder.add_edge("tools", "call_model")

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
