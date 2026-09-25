"""2단: 로컬 웹 — FastAPI + 정적 HTML 1장 (drag/drop + SSE stepper).

    python -m capcut_agent.server        # http://127.0.0.1:8765

런타임 단계: silence → asr → filler → draft
- 단계당 최소 0.5s (캐시 hit 이어도 애니메이션이 보이도록)
- 업로드는 content hash 로 저장 → 같은 영상 재업로드 시 캐시 hit
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .asr import pick_backend, transcribe
from .draft import build_draft, probe
from .env_check import capcut_draft_root, detect_track
from .silence import SilenceParams, detect_silences, keep_ranges

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("CAPCUT_AGENT_CACHE", ROOT.parent / ".cache"))
UPLOADS = CACHE / "uploads"
FALLBACK_DRAFTS = ROOT.parent / "out" / "drafts"
MIN_STEP_SEC = 0.5
ALLOWED_EXT = {".mp4", ".mov", ".m4v"}
STEPS = ["silence", "asr", "filler", "draft"]

app = FastAPI(title="캡컷 에이전트")


def draft_root() -> tuple[Path, bool]:
    root = capcut_draft_root()
    if root and root.is_dir():
        return root, True
    return FALLBACK_DRAFTS, False


# ── jobs ────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    src: Path
    params: SilenceParams
    cache_hit: bool
    content_hash: str
    events: list[dict] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    done: bool = False

    async def emit(self, ev: dict) -> None:
        async with self.cond:
            self.events.append(ev)
            self.cond.notify_all()


JOBS: dict[str, Job] = {}
# whisper 는 동시 호출 시 segfault (numba/mlx 비안전) → 프로세스 전체에서 한 번에 하나
ASR_LOCK = asyncio.Lock()


async def _step(job: Job, name: str, fn: Callable[[], Awaitable[Any]]) -> Any:
    await job.emit({"type": "step", "step": name, "status": "running"})
    t0 = time.perf_counter()
    result, detail = await fn()
    elapsed = time.perf_counter() - t0
    if elapsed < MIN_STEP_SEC:
        await asyncio.sleep(MIN_STEP_SEC - elapsed)
    await job.emit({"type": "step", "step": name, "status": "done",
                    "detail": detail, "sec": round(elapsed, 2)})
    return result


async def _skip(job: Job, name: str, detail: str) -> None:
    await job.emit({"type": "step", "step": name, "status": "running"})
    await asyncio.sleep(MIN_STEP_SEC)
    await job.emit({"type": "step", "step": name, "status": "skipped", "detail": detail})


async def run_job(job: Job) -> None:
    current = "silence"
    try:
        info = await asyncio.to_thread(probe, str(job.src))

        async def silence():
            sil = await asyncio.to_thread(detect_silences, str(job.src),
                                          job.params.noise_db, job.params.min_silence)
            keeps = keep_ranges(sil, info.duration, job.params)
            return keeps, f"무음 {len(sil)} · 보존 {len(keeps)}구간"

        keeps = await _step(job, "silence", silence)
        current = "asr"
        loop = asyncio.get_running_loop()
        last_pct = [-1]

        def on_progress(frac: float) -> None:
            pct = int(frac * 100)
            if pct != last_pct[0]:
                last_pct[0] = pct
                asyncio.run_coroutine_threadsafe(
                    job.emit({"type": "step", "step": "asr", "status": "running", "detail": f"{pct}%"}), loop)

        async def asr():
            await job.emit({"type": "step", "step": "asr", "status": "running",
                            "detail": pick_backend()[1].split("/")[-1]})
            if ASR_LOCK.locked():
                await job.emit({"type": "step", "step": "asr", "status": "running", "detail": "대기 중"})
            async with ASR_LOCK:
                tr, hit = await asyncio.to_thread(transcribe, job.src, job.content_hash, on_progress)
            d = f"세그먼트 {len(tr.segments)} · 단어 {tr.word_count}" + (" · 캐시 hit" if hit else "")
            return tr, d

        transcript = await _step(job, "asr", asr)
        await job.emit({"type": "transcript", "text": transcript.text,
                        "segments": [{"start": round(x.start, 2), "end": round(x.end, 2), "text": x.text.strip()}
                                     for x in transcript.segments]})
        current = "filler"
        await _skip(job, "filler", "4단 예정")
        current = "draft"
        root, is_capcut = draft_root()

        async def draft():
            r = await asyncio.to_thread(build_draft, str(job.src), keeps, str(root), None, transcript)
            return r, f"{r['segments']}컷 · 자막 {r['subtitles']}"

        r = await _step(job, "draft", draft)
        await job.emit({"type": "result", **r, "capcut_root": is_capcut,
                        "cache_hit": job.cache_hit})
    except Exception as e:  # noqa: BLE001 — 사용자에게 그대로 보여준다
        await job.emit({"type": "step", "step": current, "status": "error", "detail": str(e)})
        await job.emit({"type": "error", "message": str(e)})
    finally:
        job.done = True
        async with job.cond:
            job.cond.notify_all()


# ── upload (content hash cache) ─────────────────────────────────────────────

async def save_upload(f: UploadFile) -> tuple[Path, bool, str]:
    name = Path(f.filename or "video.mp4").name
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, f"mp4 / mov 만 지원: {name}")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS / f".tmp-{uuid.uuid4().hex}"
    h = hashlib.sha256()
    with tmp.open("wb") as out:
        while chunk := await f.read(1 << 20):
            h.update(chunk)
            out.write(chunk)
    digest = h.hexdigest()[:16]
    dst_dir = UPLOADS / digest
    dst = dst_dir / name
    if dst_dir.exists() and any(dst_dir.iterdir()):
        existing = next(p for p in dst_dir.iterdir())
        tmp.unlink()
        return existing, True, digest
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(tmp, dst)
    return dst, False, digest


# ── routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/env")
def env() -> dict:
    t = detect_track()
    root, is_capcut = draft_root()
    d = SilenceParams()
    backend, model = pick_backend()
    return {"track": t.code, "asr": t.asr, "asr_model": model, "label": t.label,
            "draft_root": str(root), "capcut_root": is_capcut, "steps": STEPS,
            "defaults": {"noise": d.noise_db, "min_silence": d.min_silence, "pad": d.pad}}


@app.post("/api/jobs")
async def create_job(file: UploadFile, noise: float = Form(-35.0),
                     min_silence: float = Form(0.45), pad: float = Form(0.12)) -> dict:
    src, hit, digest = await save_upload(file)
    job = Job(uuid.uuid4().hex[:12], src, SilenceParams(noise, min_silence, pad), hit, digest)
    JOBS[job.id] = job
    asyncio.create_task(run_job(job))
    return {"job_id": job.id, "cache_hit": hit}


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job 없음")

    async def stream():
        i = 0
        while True:
            async with job.cond:
                await job.cond.wait_for(lambda: i < len(job.events) or job.done)
                batch, finished = job.events[i:], job.done
            for ev in batch:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            i += len(batch)
            if finished and i >= len(job.events):
                yield "event: end\ndata: {}\n\n"
                return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/reveal")
def reveal(body: dict) -> dict:
    """드래프트 폴더를 Finder/탐색기에서 연다 (로컬 전용)."""
    p = Path(body.get("path", "")).resolve()
    root = draft_root()[0].resolve()
    if not p.is_relative_to(root) or not p.exists():
        raise HTTPException(404, "드래프트 폴더 밖이거나 없음")
    import platform
    import subprocess
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", "-R", str(p)])
    elif system == "Windows":
        subprocess.Popen(["explorer", "/select,", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p.parent)])
    return {"ok": True}


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    url = f"http://127.0.0.1:{port}"
    print(f"▶ {url}")
    if not os.environ.get("NO_BROWSER"):
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
