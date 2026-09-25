"""Transcript → 타임라인 자막 (3단: 세그먼트 단위, 화면용 청크).

1. whisper 토큰 → 어절: 앞 공백 있는 토큰이 새 어절 시작 ("안녕"+"하세요" → "안녕하세요")
2. 세그먼트마다 어절을 max_chars 이내 청크로 묶는다.
   의존명사 함정: "할 | 수 있어요", "하는 | 거예요" 처럼 끊기면 읽기 불편 →
   의존명사로 시작하는 어절 앞, 그리고 "수/줄/적" 바로 뒤에서는 끊지 않는다.
3. 원본 시간 → 컷 타임라인 시간 매핑. 잘린 구간에 떨어진 어절은 버린다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .asr import Segment, Transcript


@dataclass
class Eojeol:
    start: float
    end: float
    text: str


@dataclass
class Piece:
    """타임라인 조각: 원본 [src, src+dur) → 타임라인 [tl, tl+dur). 단위 µs."""
    src: int
    tl: int
    dur: int


@dataclass
class Cue:
    start: int   # 타임라인 µs
    end: int
    text: str


_DEP = r"(?:것|거|게|수|줄|데|때문|뿐|만큼|듯|채|척|체|대로|따름|김|터|적|바|지)"
_SUFFIX = r"(?:은|는|이|가|을|를|도|만|에|에서|으로|로|의|야|요|죠|이죠|예요|이에요|입니다|인데|이라서|라서|처럼|밖에)"
_DEP_RE = re.compile(rf"^{_DEP}{_SUFFIX}{{0,2}}$")
# "할 수(도) 있다", "할 줄(을) 알다", "한 적(이) 있다" → 뒤 어절과 붙인다
_BIND_NEXT_RE = re.compile(r"^(?:수|줄|적)(?:이|가|도|은|는|을|를)?$")
_PUNCT = ".,?!…~"


def _bare(t: str) -> str:
    return t.strip().strip(_PUNCT + "\"'“”‘’")


def is_dependent_noun(text: str) -> bool:
    return bool(_DEP_RE.match(_bare(text)))


def to_eojeols(seg: Segment) -> list[Eojeol]:
    out: list[Eojeol] = []
    for w in seg.words:
        if not w.text.strip():
            continue
        if out and not w.text.startswith(" "):
            e = out[-1]
            e.text += w.text
            e.end = max(e.end, w.end)
        else:
            out.append(Eojeol(w.start, max(w.start, w.end), w.text.strip()))
    if not out and seg.text.strip():   # 단어 타임스탬프가 없으면 세그먼트 통째로
        out.append(Eojeol(seg.start, seg.end, seg.text.strip()))
    return out


def _can_break(prev: Eojeol, nxt: Eojeol) -> bool:
    return not is_dependent_noun(nxt.text) and not _BIND_NEXT_RE.match(_bare(prev.text))


def chunk(eojeols: list[Eojeol], max_chars: int) -> list[list[Eojeol]]:
    chunks: list[list[Eojeol]] = []
    cur: list[Eojeol] = []
    last_ok = -1   # cur 안에서 마지막으로 끊어도 되는 위치 (그 인덱스 뒤에서 끊음)
    for e in eojeols:
        if cur:
            if _can_break(cur[-1], e):
                last_ok = len(cur) - 1
            length = len(" ".join(x.text for x in cur + [e]))
            ends_sentence = cur[-1].text.rstrip()[-1:] in ".?!"
            if ends_sentence and _can_break(cur[-1], e):
                chunks.append(cur)
                cur, last_ok = [], -1
            elif length > max_chars and last_ok >= 0:
                chunks.append(cur[: last_ok + 1])
                cur, last_ok = cur[last_ok + 1:], -1
                for i in range(1, len(cur)):
                    if _can_break(cur[i - 1], cur[i]):
                        last_ok = i - 1
        cur.append(e)
    if cur:
        chunks.append(cur)
    return chunks


def _map(t_us: int, pieces: list[Piece]) -> tuple[int, int] | None:
    """원본 µs → (타임라인 µs, 조각 index). 잘린 구간이면 None."""
    for i, p in enumerate(pieces):
        if p.src <= t_us < p.src + p.dur:
            return p.tl + (t_us - p.src), i
    return None


def _clamp_to_piece(t_us: int, p: Piece) -> int:
    return p.tl + min(max(t_us - p.src, 0), p.dur)


def build_cues(tr: Transcript, pieces: list[Piece], max_chars: int = 16,
               min_dur: float = 0.3, gap_fill: float = 0.3) -> list[Cue]:
    cues: list[Cue] = []
    for seg in tr.segments:
        for ch in chunk(to_eojeols(seg), max_chars):
            kept: list[tuple[Eojeol, int]] = []
            for e in ch:
                mid = int((e.start + e.end) / 2 * 1e6)
                m = _map(mid, pieces)
                if m:
                    kept.append((e, m[1]))
            if not kept:
                continue
            (first, pi0), (last, pi1) = kept[0], kept[-1]
            start = _clamp_to_piece(int(first.start * 1e6), pieces[pi0])
            end = _clamp_to_piece(int(last.end * 1e6), pieces[pi1])
            cues.append(Cue(start, end, " ".join(e.text for e, _ in kept)))

    total = pieces[-1].tl + pieces[-1].dur if pieces else 0
    cues.sort(key=lambda c: c.start)
    out: list[Cue] = []
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else total
        if out and c.start < out[-1].end:          # 겹침 제거
            c.start = out[-1].end
        end = max(c.end, c.start + int(min_dur * 1e6))
        if 0 <= nxt - end < gap_fill * 1e6:         # 짧은 틈은 메워서 깜빡임 방지
            end = nxt
        c.end = min(end, nxt, total)
        if c.end > c.start:
            out.append(c)
    return out
