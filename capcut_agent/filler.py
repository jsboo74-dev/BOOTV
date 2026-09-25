"""4단: 대본(Transcript) 기반 잔말 / NG 감지 — 단어 하나가 아니라 문장 흐름을 보고 결정한다.

- filler  : "음", "어", "으음" 같은 군말. "그/저/뭐/이제"는 뒤에 쉼(≥0.3s)이 있거나 문장 끝일 때만.
- stutter : 같은 어절 연속 반복 ("그래서 그래서") → 앞의 것을 컷.
- ng      : 같은 문장을 다시 시작한 경우 (앞 시도가 뒤 문장의 앞부분과 거의 같음) → 앞 시도를 컷.
            "다시 할게요", "NG", "처음부터" 같은 명시적 마커 문장도 컷 (+ 그 앞 문장).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from .asr import Transcript
from .subtitles import to_eojeols

_ALWAYS = re.compile(r"^(?:음+|으+음*|어+|엄+|흠+|에+|아+|으+|어음|음음|허)$")
_PAUSE_ONLY = re.compile(r"^(?:그+|저+|뭐+|이제|막|약간|그니까)$")
_MARKER = re.compile(r"^(?:아+)?(?:다시(?:할게요|갈게요|하겠습니다|요|한번)?|잠깐(?:만요)?|엔지|ng|처음부터(?:다시)?(?:할게요|갈게요)?|컷|아니다|잘못했다)$")
_MARKER_RETAKE = re.compile(r"다시|엔지|ng|처음부터")
_PUNCT = ".,?!…~\"'“”‘’"
PAUSE_SEC = 0.3


def _bare(t: str) -> str:
    return t.strip().strip(_PUNCT)


def _norm(t: str) -> str:
    return re.sub(rf"[\s{re.escape(_PUNCT)}]", "", t).lower()


@dataclass
class Word:
    t: str
    s: float
    e: float
    cut: str | None = None   # None | "filler" | "stutter" | "ng"


@dataclass
class Sentence:
    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].s

    @property
    def end(self) -> float:
        return self.words[-1].e

    def text(self, include_cut: bool = True) -> str:
        return " ".join(w.t for w in self.words if include_cut or not w.cut)


@dataclass
class Cut:
    kind: str
    start: float
    end: float
    text: str


@dataclass
class Analysis:
    sentences: list[Sentence]
    cuts: list[Cut]

    def kept_words(self) -> list[tuple[float, float]]:
        return [(w.s, w.e) for s in self.sentences for w in s.words if not w.cut]

    def counts(self) -> dict[str, int]:
        c = {"filler": 0, "stutter": 0, "ng": 0}
        for x in self.cuts:
            c[x.kind] += 1
        return c

    def to_dict(self) -> dict:
        return {"sentences": [{"start": round(s.start, 2), "end": round(s.end, 2),
                               "words": [asdict(w) for w in s.words]} for s in self.sentences],
                "cuts": [asdict(c) for c in self.cuts], "counts": self.counts()}


def split_sentences(tr: Transcript) -> list[Sentence]:
    out: list[Sentence] = []
    for seg in tr.segments:
        cur = Sentence()
        for e in to_eojeols(seg):
            cur.words.append(Word(e.text, e.start, e.end))
            if e.text.rstrip()[-1:] in ".?!":
                out.append(cur)
                cur = Sentence()
        if cur.words:
            out.append(cur)
    return out


def _mark_fillers(sentences: list[Sentence]) -> None:
    for s in sentences:
        for i, w in enumerate(s.words):
            b = _bare(w.t)
            if _ALWAYS.match(b):
                w.cut = "filler"
            elif _PAUSE_ONLY.match(b):
                last = i == len(s.words) - 1
                if last or s.words[i + 1].s - w.e >= PAUSE_SEC:
                    w.cut = "filler"


def _mark_stutters(sentences: list[Sentence]) -> None:
    for s in sentences:
        live = [w for w in s.words if not w.cut]
        for a, b in zip(live, live[1:]):
            if _norm(a.t) and _norm(a.t) == _norm(b.t):
                a.cut = "stutter"


def _live_text(s: Sentence) -> str:
    return _norm(s.text(include_cut=False))


def is_retake(a: str, b: str, a_complete: bool) -> bool:
    """a 가 b 를 다시 시작하기 전의 실패한 시도인가."""
    if len(a) < 4 or len(b) < len(a) * 0.8:
        return False
    k = min(len(a), 8)
    if SequenceMatcher(None, a[:k], b[:k]).ratio() < 0.7:
        return False
    head = b[: len(a) + 2]
    return SequenceMatcher(None, a, head).ratio() >= (0.8 if a_complete else 0.6)


def _mark_ng(sentences: list[Sentence]) -> None:
    def cut_all(s: Sentence) -> None:
        for w in s.words:
            w.cut = w.cut or "ng"

    live = [s for s in sentences if _live_text(s)]
    for i, s in enumerate(live):
        t = _live_text(s)
        if _MARKER.match(t):
            cut_all(s)
            if _MARKER_RETAKE.search(t) and i > 0:
                cut_all(live[i - 1])
            continue
        # 다음 (마커가 아닌) 문장과 비교
        j = i + 1
        while j < len(live) and _MARKER.match(_live_text(live[j])):
            j += 1
        if j < len(live):
            complete = s.words[-1].t.rstrip()[-1:] in ".?!"
            if is_retake(t, _live_text(live[j]), complete):
                cut_all(s)


def _collect_cuts(sentences: list[Sentence]) -> list[Cut]:
    """연속된 같은 종류의 컷 단어는 하나의 구간으로."""
    cuts: list[Cut] = []
    for s in sentences:
        for w in s.words:
            if not w.cut:
                continue
            prev = cuts[-1] if cuts else None
            if prev and prev.kind == w.cut and w.s - prev.end < 0.35:
                prev.end = max(prev.end, w.e)
                prev.text += " " + w.t
            else:
                cuts.append(Cut(w.cut, w.s, w.e, w.t))
    return cuts


def analyze(tr: Transcript, filler: bool = True, ng: bool = True) -> Analysis:
    sentences = split_sentences(tr)
    if filler:
        _mark_fillers(sentences)
        _mark_stutters(sentences)
    if ng:
        _mark_ng(sentences)
    return Analysis(sentences, _collect_cuts(sentences))


def apply_cuts(keeps: list[tuple[float, float]], cuts: list[Cut],
               kept_words: list[tuple[float, float]] | None = None,
               min_keep: float = 0.12, orphan_max: float = 0.6) -> list[tuple[float, float]]:
    """보존 구간(무음 기준) − 잔말/NG 구간 → 최종 보존 구간.

    컷 때문에 생긴 자투리(말 없는 짧은 조각, 대개 패딩 잔여)는 같이 버린다 → 미세 점프컷 방지.
    """
    out: list[tuple[float, float]] = []
    spans = sorted((c.start, c.end) for c in cuts)
    for a, b in keeps:
        pieces = [(a, b)]
        for cs, ce in spans:
            nxt = []
            for x, y in pieces:
                if ce <= x or cs >= y:
                    nxt.append((x, y))
                    continue
                if cs > x:
                    nxt.append((x, cs))
                if ce < y:
                    nxt.append((ce, y))
            pieces = nxt
        for x, y in pieces:
            if y - x < min_keep:
                continue
            touched = (x, y) != (a, b)
            if touched and kept_words is not None and y - x < orphan_max and \
                    not any(x <= (ws + we) / 2 < y for ws, we in kept_words):
                continue
            out.append((x, y))
    return out
