"""보존 구간 → 캡컷 드래프트 폴더.

캡컷 함정 대응:
- Mac 신버전은 draft_info.json 을 읽는다 → draft_content.json 과 둘 다 쓴다.
- draft_meta_info.json 의 tm_duration(µs)이 0이면 목록 썸네일/길이가 깨진다 → 채운다.
- Mac 샌드박스: 캡컷이 외부 경로 미디어를 못 여는 경우가 있다 → 원본을 드래프트 폴더
  안 materials/ 로 복사(가능하면 하드링크)하고 그 경로를 참조한다.
- 컷 경계는 프레임 단위로 스냅 → 타임라인에 1프레임 틈/겹침이 생기지 않게.
- 자막 위치 transform_y 는 '캔버스 높이의 절반' 단위, 음수 = 아래.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pycapcut as cc
import pymediainfo

from .asr import Transcript
from .subtitles import Cue, Piece, build_cues


@dataclass
class MediaInfo:
    duration: float   # 초
    width: int        # 회전 반영 후 화면 기준
    height: int
    fps: float


def probe(path: str) -> MediaInfo:
    info = pymediainfo.MediaInfo.parse(path)
    if not info.video_tracks:
        raise ValueError(f"비디오 트랙 없음: {path}")
    v = info.video_tracks[0]
    w, h = int(v.width), int(v.height)
    rot = int(float(v.rotation or 0)) % 180
    if rot == 90:
        w, h = h, w
    fps = float(v.frame_rate or 30)
    dur = float(v.duration or info.general_tracks[0].duration) / 1000
    return MediaInfo(dur, w, h, fps)


def _place_media(src: Path, draft_dir: Path) -> Path:
    dst_dir = draft_dir / "materials"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _unique_name(root: Path, base: str) -> str:
    name, i = base, 2
    while (root / name).exists():
        name, i = f"{base} ({i})", i + 1
    return name


def plan_timeline(keeps: list[tuple[float, float]], fps: float, material_dur_us: int) -> list[Piece]:
    """보존 구간(초) → 프레임 스냅된 타임라인 조각(µs). 조각끼리 틈 없이 이어진다."""
    us_per_frame = 1_000_000 / fps
    fr = lambda f: round(f * us_per_frame)  # noqa: E731  frame → µs
    pieces: list[Piece] = []
    cursor = 0  # 타임라인 누적 프레임
    for a, b in keeps:
        f0, f1 = round(a * fps), round(b * fps)
        n = f1 - f0
        if n <= 0:
            continue
        src = fr(f0)
        dur = min(fr(cursor + n) - fr(cursor), material_dur_us - src)
        if dur <= 0:
            continue
        pieces.append(Piece(src, fr(cursor), dur))
        cursor += n
    return pieces


def _subtitle_style(m: MediaInfo) -> dict:
    vertical = m.height > m.width
    return dict(
        style=cc.TextStyle(size=9.0 if vertical else 7.0, bold=True, align=1,
                           auto_wrapping=True, max_line_width=0.82),
        border=cc.TextBorder(width=40.0),
        # transform_y 단위 = 캔버스 높이의 절반, 음수가 아래. 세로 영상은 쇼츠 UI 피해서 조금 위로.
        clip_settings=cc.ClipSettings(transform_y=-0.55 if vertical else -0.8),
    )


def build_draft(src_path: str, keeps: list[tuple[float, float]], draft_root: str,
                draft_name: str | None = None, transcript: Transcript | None = None) -> dict:
    src = Path(src_path).resolve()
    root = Path(draft_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    m = probe(str(src))
    fps_i = max(1, round(m.fps))

    name = _unique_name(root, draft_name or f"{src.stem}_컷편집")
    folder = cc.DraftFolder(str(root))
    script = folder.create_draft(name, m.width, m.height, fps=fps_i)
    draft_dir = root / name

    media = _place_media(src, draft_dir)
    material = cc.VideoMaterial(str(media))
    script.add_track(cc.TrackType.video)

    pieces = plan_timeline(keeps, m.fps, material.duration)
    for p in pieces:
        script.add_segment(cc.VideoSegment(material, cc.Timerange(p.tl, p.dur),
                                           source_timerange=cc.Timerange(p.src, p.dur)))

    cues: list[Cue] = []
    if transcript is not None:
        cues = build_cues(transcript, pieces, max_chars=16 if m.height > m.width else 24)
        if cues:
            script.add_track(cc.TrackType.text, "자막")
            st = _subtitle_style(m)
            for c in cues:
                script.add_segment(cc.TextSegment(c.text, cc.Timerange(c.start, c.end - c.start), **st), "자막")

    script.save()
    content_path = draft_dir / "draft_content.json"
    shutil.copy2(content_path, draft_dir / "draft_info.json")

    total_us = json.loads(content_path.read_text(encoding="utf-8"))["duration"]
    now_us = int(time.time() * 1_000_000)
    meta_path = draft_dir / "draft_meta_info.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({
        "draft_id": str(uuid.uuid4()).upper(),
        "draft_name": name,
        "draft_fold_path": str(draft_dir),
        "draft_root_path": str(root),
        "tm_duration": total_us,
        "tm_draft_create": now_us,
        "tm_draft_modified": now_us,
        "draft_timeline_materials_size_": media.stat().st_size,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=4), encoding="utf-8")

    return {
        "draft_name": name,
        "draft_dir": str(draft_dir),
        "segments": len(pieces),
        "subtitles": len(cues),
        "source_sec": round(m.duration, 2),
        "output_sec": round(total_us / 1e6, 2),
        "canvas": f"{m.width}x{m.height}@{m.fps:g}",
    }
