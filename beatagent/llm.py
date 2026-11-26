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


# =============================================================================
# Gemini backend (the brief allows any LLM with tool calling)
# =============================================================================
# Rough USD per million tokens (input, output) for the spend log only; unknown models use the flash row.
GEMINI_PRICES = {"flash": (0.30, 2.50), "pro": (1.25, 10.0)}
_GEMINI_CANDIDATES: list[str] = []
_GEMINI_COOL: dict[str, float] = {}  # model -> when it last failed (busy, timed out, missing, no quota)
_GEMINI_GOOD: list[str] = []  # the model that last answered, tried first by every new loop
GEMINI_TIMEOUT_MS = 40_000  # per request
GEMINI_COOL_S = 300  # a failed model goes to the back of the list for this long
GEMINI_TURN_BUDGET_S = 120  # stop trying further models after this long
_KEEP = {"type", "properties", "required", "items", "enum", "anyOf", "description", "minimum", "maximum",
         "minItems", "maxItems", "nullable", "format"}


def is_permanent(error: str | None) -> bool:
    """A missing or rejected key: stop asking that provider for the rest of the session."""
    return bool(error) and ("rejected" in error or error.startswith("No "))


def gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def to_gemini_schema(schema: dict) -> dict:
    """Our pydantic JSON Schema -> the conservative subset Gemini function declarations accept:
    $refs inlined, oneOf -> anyOf, const -> enum, and only well-supported keywords kept."""
    defs = schema.get("$defs", {})

    def walk(s):
        if isinstance(s, list):
            return [walk(x) for x in s]
        if not isinstance(s, dict):
            return s
        if "$ref" in s:
            return walk(defs[s["$ref"].split("/")[-1]])
        s = dict(s)
        if "oneOf" in s:
            s["anyOf"] = s.pop("oneOf")
        if "const" in s:
            s["enum"] = [s.pop("const")]
            s.setdefault("type", "string")
        out = {}
        for k, v in s.items():
            if k == "properties":
                out[k] = {name: walk(sub) for name, sub in v.items()}
            elif k in _KEEP:
                out[k] = walk(v)
        return out

    return walk(schema)


def gemini_candidates(client) -> list[str]:
    """Models to try, best first: BEAT_AGENT_GEMINI_MODEL if set, then this key's stable Flash models
    newest first, then Flash-Lite, then Pro. Busy or missing models are skipped for the next one."""
    global _GEMINI_CANDIDATES
    if _GEMINI_CANDIDATES:
        return _GEMINI_CANDIDATES
    import re

    names = []
    for m in client.models.list():
        name = (m.name or "").split("/")[-1]
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions and name.startswith("gemini-"):
            names.append(name)
    stable = [n for n in names if not re.search(r"preview|exp|image|tts|audio|live|embed|thinking|latest", n)]

    def version(n):
        v = re.match(r"gemini-(\d+(?:\.\d+)?)", n)
        return float(v.group(1)) if v else 0.0

    ranked = []
    for pool in ([n for n in stable if n.endswith("-flash")], [n for n in stable if n.endswith("-flash-lite")],
                 [n for n in stable if n.endswith("-pro")]):
        ranked += [n for n in sorted(pool, key=version, reverse=True) if n not in ranked]
    forced = os.environ.get("BEAT_AGENT_GEMINI_MODEL")
    if forced:
        ranked = [forced] + [n for n in ranked if n != forced]
    if not ranked:
        raise LLMUnavailable("This Gemini key has no model that supports generateContent.")
    _GEMINI_CANDIDATES = ranked
    return ranked


def pick_gemini_model(client) -> str:
    return gemini_candidates(client)[0]


def gemini_failure(e: LLMUnavailable) -> str:
    """'fatal' (stop: bad key, bad request, no network), 'retry' (busy: retry once, then next model)
    or 'skip' (go straight to the next model: timeout, 404, 429, 504)."""
    from google.genai import errors

    cause = e.__cause__
    if isinstance(cause, errors.APIError):
        code = getattr(cause, "code", None)
        if code in (500, 502, 503):
            return "retry"
        if code in (404, 429, 504):
            return "skip"
        return "fatal"
    name = type(cause).__name__.lower()
    return "skip" if "timeout" in name else "fatal"


class GeminiToolLoop:
    """Same interface as ToolLoop (run, tools, spend_usd, model), on Google's Gemini API."""

    provider = "gemini"

    def __init__(self, system: str, tools: list[Tool], model: str | None = None, max_steps: int = 6,
                 tag: str = "agent", **_):
        self.system = system
        self.tools = {t.name: t for t in tools}
        self._model = model
        self.max_steps = max_steps
        self.tag = tag
        self.contents: list = []
        self.spend_usd = 0.0
        self._client = None

    @property
    def model(self) -> str:
        return self._model or os.environ.get("BEAT_AGENT_GEMINI_MODEL") or "gemini (auto-selected)"

    def client(self):
        if self._client is None:
            if not gemini_key():
                raise LLMUnavailable("No GEMINI_API_KEY is set.")
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=gemini_key(),
                                        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
        return self._client

    def _config(self):
        from google.genai import types

        decls = [types.FunctionDeclaration(name=t.name, description=t.description,
                                           parameters_json_schema=to_gemini_schema(t.input_schema))
                 for t in self.tools.values()]
        return types.GenerateContentConfig(
            system_instruction=self.system, tools=[types.Tool(function_declarations=decls)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO")))

    def _strip_signatures(self):
        """Thought signatures are tied to the model that wrote them; drop them before switching models."""
        for c in self.contents:
            for part in c.parts or []:
                if getattr(part, "thought_signature", None):
                    part.thought_signature = None

    def _order(self) -> list[str]:
        order = list(dict.fromkeys([m for m in (self._model, *_GEMINI_GOOD) if m] + gemini_candidates(self.client())))
        now = time.time()
        fresh = [m for m in order if now - _GEMINI_COOL.get(m, 0.0) > GEMINI_COOL_S]
        return fresh + [m for m in order if m not in fresh]

    def _create(self):
        """generate_content, falling back across models: a busy model (500/502/503) gets one quick retry,
        a timed-out, missing (404) or out-of-quota (429) one is skipped at once; either way the next
        candidate is tried. A rejected key, a bad request or no network stops the search."""
        t0 = time.time()
        last: LLMUnavailable | None = None
        for model in self._order():
            if time.time() - t0 > GEMINI_TURN_BUDGET_S:
                break
            if self._model and model != self._model:
                self._strip_signatures()
            for attempt in range(2):
                try:
                    resp = self._create_once(model)
                except LLMUnavailable as e:
                    last = e
                    kind = gemini_failure(e)
                    if kind == "fatal":
                        raise
                    if kind == "retry" and attempt == 0:
                        time.sleep(1.5)
                        continue
                    _GEMINI_COOL[model] = time.time()
                    break
                self._model = model
                _GEMINI_GOOD[:] = [model]
                _GEMINI_COOL.pop(model, None)
                return resp
        raise LLMUnavailable(f"Every Gemini model was busy or unavailable (last: {last}).")

    def _create_once(self, model: str):
        from google.genai import errors

        try:
            return self.client().models.generate_content(model=model, contents=self.contents, config=self._config())
        except errors.ClientError as e:
            code = getattr(e, "code", None)
            msg = str(e)
            if code in (401, 403) or "API key not valid" in msg or "API_KEY_INVALID" in msg:
                raise LLMUnavailable(f"The Gemini API key was rejected ({code}).") from e
            if code == 429:
                raise LLMUnavailable("Gemini is rate-limited right now; try again in a moment.") from e
            raise LLMUnavailable(f"The Gemini API returned an error ({code}): {msg[:200]}") from e
        except errors.ServerError as e:
            raise LLMUnavailable(f"The Gemini API had a server error ({getattr(e, 'code', '')}).") from e
        except Exception as e:  # network errors and timeouts from the HTTP client
            raise LLMUnavailable(f"Could not reach the Gemini API ({type(e).__name__}).") from e

    def _account(self, resp) -> dict:
        um = getattr(resp, "usage_metadata", None)
        u = {"input": (getattr(um, "prompt_token_count", 0) or 0) if um else 0,
             "output": ((getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0))
             if um else 0, "cache_read": 0, "cache_write": 0}
        p = GEMINI_PRICES["pro" if "pro" in self.model else "flash"]
        cost = (u["input"] * p[0] + u["output"] * p[1]) / 1e6
        u["usd"] = round(cost, 5)
        self.spend_usd += cost
        try:
            USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(USAGE_LOG, "a") as f:
                f.write(json.dumps({"t": time.time(), "tag": self.tag, "model": self.model, "estimate": True, **u}) + "\n")
        except OSError:
            pass
        return u

    def run(self, user_content: str) -> LoopResult:
        from google.genai import types

        self.contents.append(types.Content(role="user", parts=[types.Part(text=user_content)]))
        calls, total = [], {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "usd": 0.0}
        for _ in range(self.max_steps):
            try:
                resp = self._create()
            except LLMUnavailable as e:
                self._rollback_user()
                return LoopResult("", calls, total, str(e), unavailable=not calls)
            u = self._account(resp)
            for k in total:
                total[k] += u[k]
            cands = resp.candidates or []
            if not cands or cands[0].content is None:
                reason = getattr(cands[0], "finish_reason", None) if cands else None
                self._rollback_user()
                return LoopResult("", calls, total, f"Gemini returned no answer ({reason or 'blocked'}).")
            content = cands[0].content
            self.contents.append(content)  # as returned, so thought signatures are preserved
            fcs = [p.function_call for p in (content.parts or []) if p.function_call]
            text = "".join(p.text for p in (content.parts or []) if p.text and not p.thought).strip()
            if not fcs:
                return LoopResult(text, calls, total)
            parts = []
            for fc in fcs:
                tool = self.tools.get(fc.name)
                inp = dict(fc.args or {})
                if tool is None:
                    out = {"error": f"Unknown tool {fc.name}"}
                else:
                    try:
                        out = tool.fn(inp)
                    except Exception as e:  # tool code must never crash the chat
                        out = {"error": f"{type(e).__name__}: {e}"}
                calls.append({"tool": fc.name, "input": inp, "output": out})
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=fc.id, name=fc.name, response=json.loads(json.dumps(out, default=str)))))
            self.contents.append(types.Content(role="user", parts=parts))
        return LoopResult("", calls, total, "Stopped after too many tool steps.")

    def _rollback_user(self):
        if self.contents and self.contents[-1].role == "user":
            self.contents.pop()


# =============================================================================
# Provider selection
# =============================================================================
_DOWN: set[str] = set()  # providers whose key was rejected in this process


def anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")


def providers() -> list[str]:
    """Providers to try, in order. BEAT_AGENT_LLM=anthropic|gemini forces one; 'auto' (default) tries
    Claude then Gemini, among those with a key that hasn't been rejected in this process."""
    choice = os.environ.get("BEAT_AGENT_LLM", "auto").lower()
    have = {"anthropic": bool(anthropic_key()), "gemini": bool(gemini_key())}
    order = [choice] if choice in have else ["anthropic", "gemini"]
    return [p for p in order if have[p] and p not in _DOWN]


def mark_down(provider: str) -> None:
    _DOWN.add(provider)


class AutoLoop:
    """Tries each available provider in turn; one whose key is rejected is skipped from then on.
    Conversation history lives in the active provider's loop."""

    def __init__(self, system: str, tools: list[Tool], tag: str = "agent", **kw):
        self._args = (system, tools)
        self._kw = dict(kw, tag=tag)
        self.tools = {t.name: t for t in tools}
        self._loops: dict[str, object] = {}
        self.provider: str | None = None

    def _loop(self, p: str):
        if p not in self._loops:
            cls = GeminiToolLoop if p == "gemini" else ToolLoop
            kw = dict(self._kw)
            if kw.get("model") and kw["model"].startswith("gemini") != (p == "gemini"):
                kw["model"] = None  # a model name for the other provider: use this one's default
            self._loops[p] = cls(*self._args, **kw)
        return self._loops[p]

    @property
    def model(self) -> str:
        return self._loop(self.provider).model if self.provider else "none"

    @property
    def spend_usd(self) -> float:
        return sum(lp.spend_usd for lp in self._loops.values())

    def run(self, user_content: str) -> LoopResult:
        last = LoopResult("", [], {}, "No LLM key is configured (ANTHROPIC_API_KEY or GEMINI_API_KEY).", unavailable=True)
        for p in providers():
            res = self._loop(p).run(user_content)
            if res.unavailable and is_permanent(res.error):
                mark_down(p)
                last = res
                continue
            self.provider = p
            return res
        return last


def make_loop(system: str, tools: list[Tool], tag: str = "agent", model: str | None = None,
              effort: str | None = None) -> AutoLoop:
    return AutoLoop(system, tools, tag=tag, model=model, effort=effort)


def any_llm_key() -> bool:
    return bool(providers())
