"""3단: whisper → Transcript(세그먼트 + 단어). 전체 대본을 먼저 뽑고 이후 단계는 전부 이걸 본다.

백엔드
- 트랙 A (Mac M칩): mlx-whisper   · 기본 모델 mlx-community/whisper-large-v3-turbo
- 그 외           : faster-whisper · 기본 모델 large-v3-turbo (CPU int8 / CUDA fp16)
환경변수 CAPCUT_ASR_BACKEND=mlx|faster, CAPCUT_ASR_MODEL=... 로 덮어쓰기.

함정
- whisper 추론은 스레드 안전하지 않다 (numba/mlx, 동시 호출 시 segfault) → 모듈 락으로 직렬화.
  서버는 여기에 asyncio.Lock 을 한 겹 더 건다.
- 캐시 키는 content hash (mtime/경로 기반은 업로드마다 miss).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .env_check import detect_track

CACHE_DIR = Path(os.environ.get("CAPCUT_AGENT_CACHE", Path(__file__).resolve().parent.parent / ".cache")) / "asr"
DEFAULT_MODELS = {"mlx": "mlx-community/whisper-large-v3-turbo", "faster": "large-v3-turbo"}
# 한국어 구두점/띄어쓰기 + 군말 전사 유도 (whisper 는 기본적으로 "음/어"를 지워버린다 → 잔말 감지 불가)
INITIAL_PROMPT = "음, 안녕하세요. 어, 오늘은 음 영상 편집에 대해서 이야기해 볼게요."
# 프롬프트/후처리가 바뀌면 올린다 → 캐시 무효화
ASR_VERSION = "v2"

_LOCK = threading.Lock()
_MODELS: dict[str, object] = {}

Progress = Callable[[float], None]


@dataclass
class Word:
    start: float
    end: float
    text: str        # whisper 원문 토큰 (앞 공백 = 새 어절 시작)
    prob: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    backend: str
    model: str
    duration: float
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.segments)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        segs = [Segment(s["start"], s["end"], s["text"], [Word(**w) for w in s["words"]])
                for s in d["segments"]]
        return cls(d["language"], d["backend"], d["model"], d["duration"], segs)


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def pick_backend() -> tuple[str, str]:
    backend = os.environ.get("CAPCUT_ASR_BACKEND") or ("mlx" if detect_track().code == "A" else "faster")
    if backend not in DEFAULT_MODELS:
        raise ValueError(f"CAPCUT_ASR_BACKEND 는 mlx | faster: {backend}")
    return backend, os.environ.get("CAPCUT_ASR_MODEL") or DEFAULT_MODELS[backend]


def cache_path(content_hash: str, backend: str, model: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", model)
    return CACHE_DIR / f"{content_hash}.{backend}.{safe}.{ASR_VERSION}.json"


# ── backends ────────────────────────────────────────────────────────────────

def _faster(path: str, model_name: str, progress: Progress | None) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("faster-whisper 없음 → pip install faster-whisper") from e
    key = f"faster:{model_name}"
    if key not in _MODELS:
        try:
            import ctranslate2
            cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            cuda = False
        _MODELS[key] = WhisperModel(model_name, device="cuda" if cuda else "cpu",
                                    compute_type="float16" if cuda else "int8")
    model = _MODELS[key]
    segs_iter, info = model.transcribe(
        path, language="ko", word_timestamps=True, vad_filter=False,
        condition_on_previous_text=False, initial_prompt=INITIAL_PROMPT, beam_size=5)
    segments: list[Segment] = []
    for s in segs_iter:
        if progress and info.duration:
            progress(min(1.0, s.end / info.duration))
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:   # 무음 환각
            continue
        words = [Word(w.start, w.end, w.word, w.probability) for w in (s.words or [])]
        segments.append(Segment(s.start, s.end, s.text, words))
    return Transcript("ko", "faster", model_name, float(info.duration), segments)


def _mlx(path: str, model_name: str, progress: Progress | None) -> Transcript:
    try:
        import mlx_whisper
    except ImportError as e:
        raise RuntimeError("mlx-whisper 없음 → pip install mlx-whisper") from e
    r = mlx_whisper.transcribe(
        path, path_or_hf_repo=model_name, language="ko", word_timestamps=True,
        condition_on_previous_text=False, initial_prompt=INITIAL_PROMPT, verbose=None)
    segments: list[Segment] = []
    for s in r["segments"]:
        if s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0:
            continue
        words = [Word(w["start"], w["end"], w["word"], w.get("probability", 1.0)) for w in s.get("words", [])]
        segments.append(Segment(s["start"], s["end"], s["text"], words))
    duration = segments[-1].end if segments else 0.0
    if progress:
        progress(1.0)
    return Transcript("ko", "mlx", model_name, duration, segments)


def transcribe(path: str | Path, content_hash: str | None = None,
               progress: Progress | None = None) -> tuple[Transcript, bool]:
    """(Transcript, cache_hit). 블로킹 — 서버에서는 to_thread 로 호출."""
    backend, model = pick_backend()
    ch = content_hash or file_hash(path)
    cp = cache_path(ch, backend, model)
    if cp.exists():
        return Transcript.from_dict(json.loads(cp.read_text(encoding="utf-8"))), True

    with _LOCK:
        fn = _mlx if backend == "mlx" else _faster
        t = fn(str(path), model, progress)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".tmp")
    tmp.write_text(json.dumps(t.to_dict(), ensure_ascii=False), encoding="utf-8")
    tmp.replace(cp)
    return t, False
