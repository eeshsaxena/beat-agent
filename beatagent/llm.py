"""Minimal Claude tool-use loop shared by the edit agent and the recreation agent.

The LLM only produces tool calls (typed operations) and short replies. Every
tool call is executed by deterministic Python, which validates it first.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)  # the project .env wins over a stale shell variable
except Exception:  # pragma: no cover
    pass

DEFAULT_MODEL = "claude-opus-5-5"
# (input, output, cache read, cache write) USD per million tokens; used for the spend log only.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
USAGE_LOG = Path(os.environ.get("BEAT_AGENT_USAGE_LOG", Path(__file__).resolve().parents[1] / "runtime" / "llm_usage.jsonl"))


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict], dict]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class LoopResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str | None = None
    unavailable: bool = False  # the LLM could not be reached at all (no key, auth, network)


def _strip(schema):
    """Drop pydantic-only keywords the API does not need."""
    if isinstance(schema, dict):
        return {k: _strip(v) for k, v in schema.items() if k not in ("discriminator",)}
    if isinstance(schema, list):
        return [_strip(v) for v in schema]
    return schema


def schema_with_defs(props: dict, required: list[str], defs: dict | None = None) -> dict:
    s = {"type": "object", "properties": props, "required": required}
    if defs:
        s["$defs"] = defs
    return _strip(s)


class ToolLoop:
    def __init__(self, system: str, tools: list[Tool], model: str | None = None, effort: str | None = None,
                 max_steps: int = 6, tag: str = "agent"):
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.model = model or os.environ.get("BEAT_AGENT_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.environ.get("BEAT_AGENT_EFFORT", "medium")
        self.max_steps = max_steps
        self.tag = tag
        self.messages: list[dict] = []
        self.spend_usd = 0.0
        self._client = None

    def client(self):
        if self._client is None:
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                raise LLMUnavailable("No ANTHROPIC_API_KEY is set, so chat commands can't be interpreted. "
                                     "Copy .env.example to .env and add your key.")
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _create(self):
        import anthropic

        kwargs = dict(
            model=self.model, max_tokens=16000, system=self.system, messages=self.messages,
            tools=[t.spec() for t in self.tools.values()], tool_choice={"type": "auto"},
            output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
        )
        try:
            if self.model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-2"):
                # server-side fallback model if the primary model's safety classifier declines
                return self.client().beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                          fallbacks="default", **kwargs)
            return self.client().beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMUnavailable(f"The API key was rejected ({e.status_code}).") from e
        except anthropic.RateLimitError as e:
            raise LLMUnavailable("The language model is rate-limited right now; try again in a moment.") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable("Could not reach the language model API (network error).") from e
        except anthropic.APIStatusError as e:
            raise LLMUnavailable(f"The language model API returned an error ({e.status_code}): {e.message}") from e

    def _account(self, usage) -> dict:
        p = PRICES.get(self.model, PRICES[DEFAULT_MODEL])
        u = {"input": usage.input_tokens or 0, "output": usage.output_tokens or 0,
             "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
             "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0}
        cost = (u["input"] * p[0] + u["output"] * p[1] + u["cache_read"] * p[2] + u["cache_write"] * p[3]) / 1e6
        u["usd"] = round(cost, 5)
        self.spend_usd += cost
        try:
            USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(USAGE_LOG, "a") as f:
                f.write(json.dumps({"t": time.time(), "tag": self.tag, "model": self.model, **u}) + "\n")
        except OSError:
            pass
        return u

    def run(self, user_content: str) -> LoopResult:
        self.messages.append({"role": "user", "content": [{"type": "text", "text": user_content}]})
        calls, total = [], {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "usd": 0.0}
        for _ in range(self.max_steps):
            try:
                resp = self._create()
            except LLMUnavailable as e:
                self._rollback_user()
                return LoopResult("", calls, total, str(e), unavailable=not calls)
            u = self._account(resp.usage)
            for k in total:
                total[k] += u[k]
            if resp.stop_reason == "refusal":
                self._rollback_user()
                return LoopResult("", calls, total, "The model declined to handle that message.")
            self.messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason == "max_tokens" and not uses:
                return LoopResult(_text(resp), calls, total, "The model's reply was cut off.")
            if not uses:
                return LoopResult(_text(resp), calls, total)
            results = []
            for b in uses:
                tool = self.tools.get(b.name)
                inp = b.input if isinstance(b.input, dict) else {}
                if tool is None:
                    out, is_err = {"error": f"Unknown tool {b.name}"}, True
                else:
                    try:
                        out = tool.fn(inp)
                        is_err = bool(out.get("error"))
                    except Exception as e:  # tool code must never crash the chat
                        out, is_err = {"error": f"{type(e).__name__}: {e}"}, True
                calls.append({"tool": b.name, "input": inp, "output": out})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out),
                                **({"is_error": True} if is_err else {})})
            self.messages.append({"role": "user", "content": results})
        return LoopResult("", calls, total, "Stopped after too many tool steps.")

    def _rollback_user(self):
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()


def _text(resp) -> str:
    return "\n".join(b.text for b in resp.content if b.type == "text").strip()
