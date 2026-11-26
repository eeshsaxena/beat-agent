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
