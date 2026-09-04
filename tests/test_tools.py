from app.tools import TOOLS, get_current_time, get_weather


def test_tools_expose_json_schema():
    # each tool advertises a name, description, and a JSON-Schema arg spec
    names = {t.name for t in TOOLS}
    assert names == {"get_current_time", "get_weather"}
    for t in TOOLS:
        assert t.description
        schema = t.args_schema.model_json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema


def test_get_current_time_ok():
    out = get_current_time.invoke({"timezone": "Asia/Tokyo"})
    assert "Asia/Tokyo" in out


def test_get_current_time_bad_timezone():
    out = get_current_time.invoke({"timezone": "Mars/Olympus"})
    assert "Unknown timezone" in out


def test_get_weather_is_deterministic():
    a = get_weather.invoke({"city": "Paris"})
    b = get_weather.invoke({"city": "  paris "})
    assert a == b
    assert a.startswith("Paris:")
    assert "demo data" in a
