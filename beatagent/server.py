"""Beat Agent web app: analyse, chat-edit, sync and recreate, with visible progress and errors.

    python -m beatagent.server          # http://127.0.0.1:8000
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from . import audio_io, click, recreate, tracking
from . import score as score_mod
from . import sync as syncmod
from .agent import EditAgent
from .edl import EditError, splice_points
from .grid import BeatGrid
from .recreate_agent import RecreateAgent
from .session import EditSession

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("BEAT_AGENT_RUNTIME", ROOT / "runtime"))
UPLOADS = RUNTIME / "uploads"
PACK = Path(os.environ.get("BEAT_AGENT_PACK", ROOT / "testpack" / "stand-in"))
GRID_CACHE = ROOT / "evals" / "out" / "part1"
STATIC = Path(__file__).parent / "static"
AUDIO_EXT = (".wav", ".flac", ".mp3", ".aiff", ".ogg")

app = FastAPI(title="Beat Agent")
POOL = ThreadPoolExecutor(max_workers=2)
LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
SESSIONS: dict[str, dict] = {}
RECREATIONS: dict[str, dict] = {}
GRIDS: dict[str, BeatGrid] = {}


# ---------------------------------------------------------------- helpers
def _library() -> dict[str, Path]:
    lib = {}
    for d in (PACK, UPLOADS):
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in AUDIO_EXT:
                    lib[p.stem] = p
    return lib


def _path(track_id: str) -> Path:
    lib = _library()
    if track_id not in lib:
        raise HTTPException(404, f"Unknown track {track_id}")
    return lib[track_id]


def _grid_for(track_id: str, audio: audio_io.Audio, say) -> BeatGrid:
    if track_id in GRIDS:
        return GRIDS[track_id]
    cached = GRID_CACHE / track_id / "grid.json"
    if cached.exists() and audio.path and audio.path.parent == PACK:
        say("Loading the cached beat grid from the evals")
        g = BeatGrid.from_dict(json.loads(cached.read_text()))
    else:
        g = tracking.analyze(audio, progress=say)
    GRIDS[track_id] = g
    return g


def _job(fn, *args) -> str:
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"status": "running", "stage": "Queued", "started": time.time(), "result": None, "error": None}

    def say(msg):
        JOBS[jid]["stage"] = msg

    def run():
        try:
            JOBS[jid]["result"] = fn(say, *args)
            JOBS[jid]["status"] = "done"
        except (EditError, syncmod.SyncError, score_mod.ScoreError, ValueError) as e:
            JOBS[jid].update(status="error", error=str(e))
        except Exception as e:  # never crash silently: report it in the UI
            traceback.print_exc()
            JOBS[jid].update(status="error", error=f"{type(e).__name__}: {e}")
        JOBS[jid]["elapsed_s"] = round(time.time() - JOBS[jid]["started"], 1)

    POOL.submit(run)
    return jid


def _get(table: dict, key: str, what: str):
    if key not in table:
        raise HTTPException(404, f"Unknown {what}")
    return table[key]


def _peaks(x: np.ndarray, cols: int = 1800) -> list[list[float]]:
    m = x.mean(axis=1) if x.ndim > 1 else x
    n = max(1, len(m) // cols)
    m = m[: n * cols].reshape(-1, n) if len(m) >= cols else m.reshape(-1, 1)
    return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(m.min(axis=1), m.max(axis=1))]


def _wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _session_view(sid: str) -> dict:
    s = SESSIONS[sid]
    sess: EditSession = s["session"]
    g = sess.grid
    y = sess.render()
    return {
        "id": sid, "name": sess.name, "duration_s": round(g.duration_s, 3), "sample_rate": sess.sr,
        "format": sess.audio.output_spec()[:2], "summary": g.summary(), "tempo_bpm": round(g.tempo_bpm, 3),
        "time_signature": g.time_signature, "grid_type": g.grid_type, "n_beats": g.n, "n_bars": g.last_bar,
        "confidence": round(g.confidence, 3), "beats": [round(float(t), 4) for t in g.times],
        "beat_in_bar": [int(b) for b in g.beat_in_bar], "bars": [int(b) for b in g.bars],
        "splices": [round(p / sess.sr, 4) for p in splice_points(sess.state.timeline)],
        "peaks": _peaks(y), "history": [h.label for h in sess.history[1:sess.cursor + 1]],
        "redo": [h.label for h in sess.history[sess.cursor + 1:]], "can_undo": sess.can_undo(),
        "can_redo": sess.can_redo(), "version": s["version"], "chat": s["chat"],
    }


# ---------------------------------------------------------------- API
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/library")
def library():
    out = []
    for tid, p in _library().items():
        try:
            info = sf.info(str(p))
            out.append({"id": tid, "file": p.name, "duration_s": round(info.duration, 2), "sample_rate": info.samplerate,
                        "kind": "clip" if tid.startswith("P") and info.duration < 40 else "track",
                        "uploaded": p.parent == UPLOADS})
        except Exception:
            continue
    mode = os.environ.get("BEAT_AGENT_INTERPRETER", "auto")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    interpreter = "offline" if mode == "offline" or not has_key else "llm"
    return {"tracks": out, "llm": has_key, "interpreter": interpreter, "llm_status": LLM_STATUS, "pack": str(PACK)}


LLM_STATUS: dict = {"checked": False}


def check_llm() -> None:
    """Probe the API key once (models.list is free). A rejected key switches this process to the
    offline grammar so the chat doesn't wait on a failing call for every message."""
    LLM_STATUS["checked"] = True
    if os.environ.get("BEAT_AGENT_INTERPRETER", "auto") != "auto":
        LLM_STATUS["note"] = f"interpreter forced to {os.environ['BEAT_AGENT_INTERPRETER']}"
        return
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        LLM_STATUS["note"] = "no API key: offline grammar"
        return
    try:
        import anthropic

        anthropic.Anthropic().models.list(limit=1)
        LLM_STATUS["note"] = "API key works"
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "status_code", None)
        LLM_STATUS["note"] = f"API key not usable ({code or type(e).__name__}): offline grammar"
        if code in (401, 403):
            os.environ["BEAT_AGENT_INTERPRETER"] = "offline"


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_EXT:
        raise HTTPException(400, f"Unsupported file type {ext!r}; use WAV, FLAC or MP3.")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    stem = "".join(c for c in Path(file.filename).stem if c.isalnum() or c in "-_")[:40] or "upload"
    dest = UPLOADS / f"{stem}{ext}"
    dest.write_bytes(await file.read())
    try:
        sf.info(str(dest))
    except Exception:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "That file could not be decoded as audio.")
    return {"id": dest.stem}


class OpenReq(BaseModel):
    track: str


@app.post("/api/sessions")
def open_session(req: OpenReq):
    path = _path(req.track)

    def work(say):
        say("Loading audio")
        a = audio_io.load(path)
        g = _grid_for(req.track, a, say)
        sess = EditSession(a, g, req.track)
        sid = uuid.uuid4().hex[:8]
        partners = [t for t, p in _library().items() if t != req.track and sf.info(str(p)).duration >= 40]
        SESSIONS[sid] = {"session": sess, "agent": None, "version": 0, "chat": [], "library_ids": partners,
                         "lock": threading.Lock()}  # one edit at a time per session (chat, undo/redo, direct ops)
        return {"session": sid}

    return {"job": _job(work)}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "Unknown job")
    j = dict(JOBS[jid])
    j["elapsed_s"] = j.get("elapsed_s") or round(time.time() - j["started"], 1)
    return j


@app.get("/api/sessions/{sid}")
def session_view(sid: str):
    if sid not in SESSIONS:
        raise HTTPException(404, "Unknown session")
    return _session_view(sid)


@app.get("/api/sessions/{sid}/audio")
def session_audio(sid: str, clicks: int = 0, v: int = 0):
    s = _get(SESSIONS, sid, "session")
    sess: EditSession = s["session"]
    y = sess.render()
    if clicks:
        y = click.audition(y, sess.sr, sess.grid)
    return Response(_wav_bytes(y, sess.sr), media_type="audio/wav")


@app.get("/api/sessions/{sid}/download/{kind}")
def session_download(sid: str, kind: str):
    s = _get(SESSIONS, sid, "session")
    sess: EditSession = s["session"]
    d = RUNTIME / "exports" / sid
    paths = sess.export(d, f"{sess.name}_edit{s['version']}", listen_copy=False)
    p = {"audio": paths["audio"], "grid": paths["grid"], "click": paths["click_audition"], "edl": paths["edl"]}.get(kind)
    if not p:
        raise HTTPException(404, "Unknown download")
    return FileResponse(p, filename=Path(p).name)


class ChatReq(BaseModel):
    message: str
    confirm: bool = False


def _agent(sid: str) -> EditAgent:
    s = SESSIONS[sid]
    if s["agent"] is None:
        def loader(other):
            def load():
                a = audio_io.load(_path(other))
                return a, _grid_for(other, a, lambda *_: None)
            return load

        lib = {other: loader(other) for other in s["library_ids"]}  # loaded on first sync only
        s["agent"] = EditAgent(s["session"], library=lib, out_dir=RUNTIME / "sync" / sid)
    return s["agent"]


@app.post("/api/sessions/{sid}/chat")
def session_chat(sid: str, req: ChatReq):
    if sid not in SESSIONS:
        raise HTTPException(404, "Unknown session")
    msg = req.message.strip()[:2000]
    if not msg:
        raise HTTPException(400, "Empty message")

    def work(say):
        say("Interpreting the message")
        with SESSIONS[sid]["lock"]:
            ag = _agent(sid)
            ag.confirm = req.confirm
            r = ag.ask(msg).to_dict()
        for ev in r["events"]:
            for k, p in (ev.get("paths") or {}).items():
                ev.setdefault("urls", {})[k] = "/api/files/" + Path(p).relative_to(RUNTIME).as_posix()
        s = SESSIONS[sid]
        s["chat"].append(r)
        if any(ev["type"] in ("applied", "undo", "redo") for ev in r["events"]):
            s["version"] += 1
        return r

    return {"job": _job(work)}


class OpsReq(BaseModel):
    ops: list[dict]


@app.post("/api/sessions/{sid}/ops")
def session_ops(sid: str, req: OpsReq):
    """Direct typed edits without the LLM: the same validation, echo and undo as chat edits."""
    from pydantic import TypeAdapter, ValidationError

    from .edl import EditOp

    s = _get(SESSIONS, sid, "session")
    try:
        ops = TypeAdapter(list[EditOp]).validate_python(req.ops)
        with s["lock"]:
            res = s["session"].apply(ops, turn_id=None)
    except ValidationError as e:
        return JSONResponse({"error": "Malformed operation: " + "; ".join(err["msg"] for err in e.errors()[:3])},
                            status_code=422)
    except EditError as e:
        entry = {"user": "[direct edit]", "reply": "", "events": [{"type": "error", "message": str(e)}], "error": None}
        s["chat"].append(entry)
        return JSONResponse({"error": str(e), "entry": entry}, status_code=409)
    s["version"] += 1
    entry = {"user": "[direct edit] " + ", ".join(o.op for o in ops), "reply": "",
             "events": [{"type": "applied", "echo": res.echo, "warnings": res.warnings}], "error": None}
    s["chat"].append(entry)
    return {"ok": True, "entry": entry}


@app.post("/api/sessions/{sid}/{action}")
def session_undo_redo(sid: str, action: str):
    if sid not in SESSIONS or action not in ("undo", "redo"):
        raise HTTPException(404, "Unknown session or action")
    s = SESSIONS[sid]
    try:
        with s["lock"]:
            msg = s["session"].undo() if action == "undo" else s["session"].redo()
    except EditError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    s["version"] += 1
    s["chat"].append({"user": f"[{action} button]", "reply": "", "events": [{"type": action, "echo": [msg]}], "error": None})
    return {"ok": True, "message": msg}


class SyncReq(BaseModel):
    a: str
    b: str
    start_bar_a: int = 33
    start_bar_b: int = 1
    seconds: float = 60.0


@app.post("/api/sync")
def sync_tracks(req: SyncReq):
    def work(say):
        say(f"Loading {req.a} and {req.b}")
        a, b = audio_io.load(_path(req.a)), audio_io.load(_path(req.b))
        ga, gb = _grid_for(req.a, a, say), _grid_for(req.b, b, say)
        r = syncmod.sync(a, ga, b, gb, req.start_bar_a, req.start_bar_b, req.seconds, progress=say)
        d = RUNTIME / "sync" / f"{req.b}_to_{req.a}_{uuid.uuid4().hex[:6]}"
        out = {}
        for name, x in (("mix", r.mix), ("stem_a", r.a_stem), ("stem_b", r.b_stem)):
            out[name] = "/api/files/" + audio_io.save(d / f"{name}.wav", x, r.sr).relative_to(RUNTIME).as_posix()
        (d / "report.json").write_text(json.dumps(r.report, indent=1))
        return {"report": r.report, "urls": out}

    return {"job": _job(work)}


class RecreateReq(BaseModel):
    clip: str


@app.post("/api/recreations")
def make_recreation(req: RecreateReq):
    path = _path(req.clip)

    def work(say):
        a = audio_io.load(path)
        if a.duration > 60:
            raise ValueError("Recreation is for short clips (2-16 bars); this file is over 60 s.")
        sc = recreate.build_score(a, req.clip, progress=say)
        rid = uuid.uuid4().hex[:8]
        d = RUNTIME / "recreations" / rid
        page = score_mod.build_page(sc, d / "page.html")
        (d / "score.json").write_text(json.dumps(sc, indent=1))
        RECREATIONS[rid] = {"score": sc, "agent": RecreateAgent(sc, page), "dir": d, "clip": req.clip, "version": 0,
                            "chat": [], "lock": threading.Lock()}
        return {"recreation": rid}

    return {"job": _job(work)}


@app.get("/api/recreations/{rid}")
def recreation_view(rid: str):
    r = _get(RECREATIONS, rid, "recreation")
    sc = r["agent"].score
    return {"id": rid, "clip": r["clip"], "tempo_bpm": sc["tempo_bpm"], "bars": sc["bars"],
            "tracks": [{"id": t["id"], "name": t["name"], "notes": len(t["notes"]), "mute": t.get("mute", False),
                        "gain_db": t.get("gain_db", 0)} for t in sc["tracks"]],
            "provenance": sc["provenance"], "version": r["version"], "chat": r["chat"],
            "page_url": f"/api/recreations/{rid}/page?v={r['version']}", "original_url": f"/api/original/{r['clip']}"}


@app.get("/api/recreations/{rid}/page", response_class=HTMLResponse)
def recreation_page(rid: str, v: int = 0):
    r = _get(RECREATIONS, rid, "recreation")
    return HTMLResponse((r["dir"] / "page.html").read_text(encoding="utf-8"))


@app.get("/api/recreations/{rid}/score")
def recreation_score(rid: str):
    r = _get(RECREATIONS, rid, "recreation")
    return FileResponse(r["dir"] / "page.html", filename=f"{r['clip']}_recreation.html")


@app.post("/api/recreations/{rid}/chat")
def recreation_chat(rid: str, req: ChatReq):
    if rid not in RECREATIONS:
        raise HTTPException(404, "Unknown recreation")

    def work(say):
        say("Interpreting the message")
        r = RECREATIONS[rid]
        with r["lock"]:
            res = r["agent"].ask(req.message.strip()[:2000])
        r["chat"].append(res)
        if any(ev["type"] in ("applied", "undo") for ev in res["events"]):
            r["version"] += 1
        return res

    return {"job": _job(work)}


@app.get("/api/original/{track_id}")
def original(track_id: str):
    return FileResponse(_path(track_id))


@app.get("/api/files/{path:path}")
def files(path: str):
    p = (RUNTIME / path).resolve()
    if not str(p).startswith(str(RUNTIME.resolve())) or not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p)


def main():
    import uvicorn

    host = os.environ.get("BEAT_AGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("BEAT_AGENT_PORT", "8000"))
    check_llm()
    print(f"Beat Agent on http://{host}:{port} ({LLM_STATUS.get('note')})")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
