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
