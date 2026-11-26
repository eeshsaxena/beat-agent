"""The provider plumbing that doesn't need a network: Gemini schema conversion, model ranking,
failure classes, and which errors switch a session to the offline grammar for good."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from beatagent import llm
from beatagent.llm import LLMUnavailable, gemini_failure, is_permanent, to_gemini_schema


def _schema_keywords(s):
    """Every JSON Schema keyword used, skipping the user-chosen names under 'properties'."""
    if isinstance(s, list):
        return set().union(*map(_schema_keywords, s)) if s else set()
    if not isinstance(s, dict):
        return set()
    out = set(s)
    for k, v in s.items():
        if k == "properties":
            for sub in v.values():
                out |= _schema_keywords(sub)
        else:
            out |= _schema_keywords(v)
    return out


def test_edit_tool_schema_converts_to_gemini_subset():
    from beatagent.agent import _DEFS, _READING, _ops_schema

    full = llm.schema_with_defs({"reading": _READING, "ops": _ops_schema}, ["reading", "ops"], _DEFS)
    g = to_gemini_schema(full)
    assert _schema_keywords(g) <= llm._KEEP
    # all eight op variants survive, each with its discriminator as a one-value enum
    variants = g["properties"]["ops"]["items"]["anyOf"]
    assert len(variants) == 8
    assert sorted(v["properties"]["op"]["enum"][0] for v in variants) == sorted(
        ["remove", "keep_only", "repeat", "move", "insert_copy", "fade", "gain", "mute"])


def test_const_and_oneof_and_ref():
    s = {"$defs": {"A": {"type": "object", "properties": {"k": {"const": "a", "title": "K"}}, "required": ["k"]}},
         "type": "object", "properties": {"x": {"oneOf": [{"$ref": "#/$defs/A"}, {"type": "integer", "minimum": 0}]}},
         "additionalProperties": False, "title": "T"}
    g = to_gemini_schema(s)
    assert g == {"type": "object", "properties": {"x": {"anyOf": [
        {"type": "object", "properties": {"k": {"enum": ["a"], "type": "string"}}, "required": ["k"]},
        {"type": "integer", "minimum": 0}]}}}


class _Client:
    def __init__(self, names):
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"])
                                                    for n in names])


def test_model_ranking(monkeypatch):
    monkeypatch.setattr(llm, "_GEMINI_CANDIDATES", [])
    monkeypatch.delenv("BEAT_AGENT_GEMINI_MODEL", raising=False)
    names = ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-2.5-pro",
             "gemini-3.1-flash-image", "gemini-omni-flash-preview", "gemini-3.1-flash-lite", "gemma-4-31b-it"]
    assert llm.gemini_candidates(_Client(names)) == [
        "gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
        "gemini-2.5-pro"]
    monkeypatch.setattr(llm, "_GEMINI_CANDIDATES", [])
    monkeypatch.setenv("BEAT_AGENT_GEMINI_MODEL", "gemini-3.5-flash-lite")
    assert llm.gemini_candidates(_Client(names))[:2] == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]


def _wrapped(cause):
    try:
        raise LLMUnavailable("x") from cause
    except LLMUnavailable as e:
        return e


@pytest.mark.parametrize("code,kind", [(503, "retry"), (500, "retry"), (504, "skip"), (404, "skip"), (429, "skip"),
                                       (400, "fatal"), (401, "fatal"), (403, "fatal")])
def test_failure_classes(code, kind):
    from google.genai import errors

    cls = errors.ServerError if code >= 500 else errors.ClientError
    err = cls(code, {"error": {"code": code, "message": "m", "status": "S"}})
    assert gemini_failure(_wrapped(err)) == kind


def test_timeouts_skip_but_no_network_stops():
    class ReadTimeout(Exception):
        pass

    class ConnectError(Exception):
        pass

    assert gemini_failure(_wrapped(ReadTimeout())) == "skip"
    assert gemini_failure(_wrapped(ConnectError())) == "fatal"


@pytest.mark.parametrize("msg,perm", [
    ("The Gemini API key was rejected (401).", True),
    ("No GEMINI_API_KEY is set.", True),
    ("Every Gemini model was busy or unavailable (last: 404 NOT_FOUND. No such model).", False),
    ("Gemini is rate-limited right now; try again in a moment.", False),
    (None, False)])
def test_only_key_problems_are_permanent(msg, perm):
    assert is_permanent(msg) is perm


def test_a_model_name_only_goes_to_its_own_provider():
    loop = llm.AutoLoop("sys", [], model="claude-sonnet-5-5")
    assert loop._loop("gemini")._model is None
    assert llm.AutoLoop("sys", [], model="gemini-3.5-flash-lite")._loop("gemini")._model == "gemini-3.5-flash-lite"


def _calls_then_fail(results):
    seq = list(results)

    def create(*_a, **_k):
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    return create


def test_a_failure_mid_turn_rolls_back_the_whole_turn(monkeypatch, tmp_path):
    """Popping only the last message used to leave a tool call with no result in the history,
    which the API rejects on every later turn."""
    from google.genai import types

    monkeypatch.setattr(llm, "USAGE_LOG", tmp_path / "usage.jsonl")
    ran = []
    tool = llm.Tool("t", "test tool", {"type": "object", "properties": {}}, lambda inp: ran.append(1) or {"ok": True})

    g = llm.GeminiToolLoop("sys", [tool], model="gemini-test")
    call = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
        role="model", parts=[types.Part(function_call=types.FunctionCall(name="t", args={}))]))])
    monkeypatch.setattr(g, "_create", _calls_then_fail([call, LLMUnavailable("busy")]))
    res = g.run("first message")
    assert ran == [1] and res.error == "busy" and not res.unavailable  # the tool ran, so this is not "unreachable"
    assert g.contents == []

    c = llm.ToolLoop("sys", [tool], model="claude-test")
    use = SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                          content=[SimpleNamespace(type="tool_use", name="t", input={}, id="u1")])
    monkeypatch.setattr(c, "_create", _calls_then_fail([use, LLMUnavailable("busy")]))
    c.messages = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
    c.run("second message")
    assert c.messages == [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]


def test_an_unreachable_provider_falls_through_to_the_next(monkeypatch):
    class Fake:
        def __init__(self, res):
            self.res, self.model, self.spend_usd = res, "m", 0.0

        def run(self, _):
            return self.res

    loop = llm.AutoLoop("sys", [])
    monkeypatch.setattr(llm, "providers", lambda: ["anthropic", "gemini"])
    loop._loops = {"anthropic": Fake(llm.LoopResult("", error="Could not reach the API.", unavailable=True)),
                   "gemini": Fake(llm.LoopResult("done"))}
    assert loop.run("hi").text == "done" and loop.provider == "gemini"
    assert "anthropic" not in llm._DOWN  # a network error doesn't rule a provider out


def test_a_failed_model_listing_is_reported_not_raised(monkeypatch):
    from google.genai import errors

    class Broken:
        def __init__(self, exc):
            self.models = SimpleNamespace(list=lambda: (_ for _ in ()).throw(exc))

    monkeypatch.setattr(llm, "_GEMINI_CANDIDATES", [])
    bad_key = errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid.", "status": "S"}})
    with pytest.raises(LLMUnavailable, match="rejected"):
        llm.gemini_candidates(Broken(bad_key))
    with pytest.raises(LLMUnavailable, match="Could not reach") as e:
        llm.gemini_candidates(Broken(ConnectionError("down")))
    assert gemini_failure(e.value) == "fatal" and not is_permanent(str(e.value))


def test_the_last_answering_model_is_remembered_for_a_few_hours(monkeypatch, tmp_path):
    import json
    import time

    monkeypatch.setattr(llm, "GEMINI_STATE", tmp_path / "gemini_last_model.json")
    assert llm._remembered_gemini_model() is None
    llm._remember_gemini_model("gemini-3.5-flash-lite")
    assert llm._remembered_gemini_model() == "gemini-3.5-flash-lite"
    llm.GEMINI_STATE.write_text(json.dumps({"model": "gemini-old", "t": time.time() - llm.GEMINI_STATE_TTL_S - 1}))
    assert llm._remembered_gemini_model() is None
    llm.GEMINI_STATE.write_text("not json")
    assert llm._remembered_gemini_model() is None

    monkeypatch.setattr(llm, "_GEMINI_GOOD", [])
    monkeypatch.setattr(llm, "_GEMINI_COOL", {})
    llm._remember_gemini_model("gemini-3.5-flash-lite")
    g = llm.GeminiToolLoop("sys", [])
    monkeypatch.setattr(llm, "gemini_candidates", lambda _c: ["gemini-3.8-flash", "gemini-3.5-flash-lite"])
    monkeypatch.setattr(g, "client", lambda: None)
    assert g._order() == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
