import pytest

from app import agent
from app.assemblyai import AssemblyAIError


class _FakeGraph:
    """Stand-in for a compiled LangGraph; records the last invocation."""

    def __init__(self, reply="  hi there  ", raises: Exception | None = None):
        self.reply = reply
        self.raises = raises
        self.state = None
        self.config = None

    async def ainvoke(self, state, config):
        self.state, self.config = state, config
        if self.raises:
            raise self.raises
        return {"messages": [agent.AIMessage(content=self.reply)]}


@pytest.fixture
def fake_graph(monkeypatch):
    graph = _FakeGraph()
    monkeypatch.setattr(agent, "_build_graph", lambda settings: graph)
    return graph


async def test_run_agent_returns_stripped_reply(fake_graph):
    out = await agent.run_agent("sess-1", "hello")
    assert out == "hi there"


async def test_run_agent_uses_session_id_as_thread_id(fake_graph):
    await agent.run_agent("sess-42", "hello")
    assert fake_graph.config == {"configurable": {"thread_id": "sess-42"}}


async def test_run_agent_passes_user_message(fake_graph):
    await agent.run_agent("s", "what is the weather")
    assert fake_graph.state["messages"][0].content == "what is the weather"


async def test_run_agent_rate_limit_becomes_429(monkeypatch):
    monkeypatch.setattr(
        agent, "_build_graph",
        lambda s: _FakeGraph(raises=RuntimeError("Error code: 429 - too many requests")),
    )
    with pytest.raises(AssemblyAIError) as excinfo:
        await agent.run_agent("s", "hi")
    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 50


async def test_run_agent_other_error_becomes_502(monkeypatch):
    monkeypatch.setattr(agent, "_build_graph", lambda s: _FakeGraph(raises=RuntimeError("boom")))
    with pytest.raises(AssemblyAIError) as excinfo:
        await agent.run_agent("s", "hi")
    assert excinfo.value.status_code == 502


async def test_run_agent_without_key_raises_500(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    agent.get_settings.cache_clear()
    with pytest.raises(AssemblyAIError) as excinfo:
        await agent.run_agent("s", "hi")
    assert excinfo.value.status_code == 500


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Error code: 429", True),
        ("rate limit exceeded", True),
        ("Too Many Requests", False),  # no digits, no 'rate limit' phrase
        ("connection reset by peer", False),
    ],
)
def test_is_rate_limit(text, expected):
    assert agent._is_rate_limit(Exception(text)) is expected


# --- ReAct tool loop (real graph, fake LLM) ---------------------------------------


class _ScriptedLLM:
    """Fake ChatOpenAI: bind_tools is a no-op, invoke replays a script of messages."""

    def __init__(self, script):
        self._script = list(script)
        self.invocations = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        msg = self._script[min(self.invocations, len(self._script) - 1)]
        self.invocations += 1
        return msg


async def test_tool_calling_runs_the_react_loop(monkeypatch):
    monkeypatch.setenv("LLM_ENABLE_TOOLS", "true")
    agent.get_settings.cache_clear()
    agent._build_graph.cache_clear()

    scripted = _ScriptedLLM(
        [
            agent.AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "Paris"}, "id": "c1", "type": "tool_call"}
                ],
            ),
            agent.AIMessage(content="It's clear in Paris right now."),
        ]
    )
    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kwargs: scripted)

    reply = await agent.run_agent("sess-tools", "what's the weather in Paris?")

    assert reply == "It's clear in Paris right now."
    assert scripted.invocations == 2  # model → tool → model


async def test_no_tool_node_when_tools_disabled(monkeypatch):
    monkeypatch.setenv("LLM_ENABLE_TOOLS", "false")
    agent.get_settings.cache_clear()
    agent._build_graph.cache_clear()

    scripted = _ScriptedLLM([agent.AIMessage(content="plain answer")])
    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kwargs: scripted)

    graph = agent._build_graph(agent.get_settings())
    assert "tools" not in graph.get_graph().nodes
    assert await agent.run_agent("s", "hi") == "plain answer"
