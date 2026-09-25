"""ffmpeg silencedetect → 보존(keep) 구간 계산. 단위는 전부 초(float)."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from .env_check import ffmpeg_exe

_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END = re.compile(r"silence_end:\s*([\d.]+)")


@dataclass
class SilenceParams:
    noise_db: float = -35.0     # 이보다 작으면 무음
    min_silence: float = 0.45   # 이보다 긴 무음만 자름
    pad: float = 0.12           # 말 앞뒤로 남길 여유 (말끝 잘림 방지)
    min_keep: float = 0.25      # 이보다 짧은 보존 구간은 버림 (숨소리·클릭음)


def detect_silences(path: str, noise_db: float, min_silence: float) -> list[tuple[float, float | None]]:
    ff = ffmpeg_exe()
    if not ff:
        raise RuntimeError("ffmpeg 없음 — python -m capcut_agent.env_check 참고")
    cmd = [ff, "-hide_banner", "-nostats", "-i", path, "-vn",
           "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg silencedetect 실패:\n{proc.stderr[-2000:]}")

    silences: list[tuple[float, float | None]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        if m := _START.search(line):
            start = max(0.0, float(m.group(1)))
        elif (m := _END.search(line)) and start is not None:
            silences.append((start, float(m.group(1))))
            start = None
    if start is not None:           # 파일 끝까지 무음
        silences.append((start, None))
    return silences


def keep_ranges(silences: list[tuple[float, float | None]], duration: float,
                p: SilenceParams) -> list[tuple[float, float]]:
    """무음의 여집합 + 패딩 → 겹치면 병합 → 너무 짧은 건 버림."""
    keeps: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in silences:
        e = duration if e is None else e
        if s > cursor:
            keeps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        keeps.append((cursor, duration))

    padded = [(max(0.0, a - p.pad), min(duration, b + p.pad)) for a, b in keeps]
    merged: list[tuple[float, float]] = []
    for a, b in padded:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= p.min_keep]
