"""무음 컷 + 대본 추출 + 세그먼트 자막 → 캡컷 드래프트.

    python -m capcut_agent input.mp4 [--noise -35] [--min-silence 0.45] [--pad 0.12] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
import time

from .asr import transcribe
from .draft import build_draft, probe
from .env_check import capcut_draft_root
from .filler import analyze, apply_cuts
from .silence import SilenceParams, detect_silences, keep_ranges


def main(argv: list[str] | None = None) -> int:
    d = SilenceParams()
    ap = argparse.ArgumentParser(prog="capcut_agent", description="토킹 영상 → 캡컷 점프컷 드래프트")
    ap.add_argument("input")
    ap.add_argument("--noise", type=float, default=d.noise_db, help="무음 기준 dB (기본 %(default)s)")
    ap.add_argument("--min-silence", type=float, default=d.min_silence, help="자를 최소 무음 길이 s")
    ap.add_argument("--pad", type=float, default=d.pad, help="말 앞뒤 여유 s")
    ap.add_argument("--min-keep", type=float, default=d.min_keep, help="버릴 짧은 구간 s")
    ap.add_argument("--out", help="드래프트 루트 (기본: 캡컷 드래프트 폴더)")
    ap.add_argument("--name", help="드래프트 이름")
    ap.add_argument("--no-asr", action="store_true", help="대본/자막/잔말 생략 (1단 동작)")
    ap.add_argument("--no-filler", action="store_true", help="잔말(음/어/반복) 컷 끄기")
    ap.add_argument("--no-ng", action="store_true", help="NG(다시 시작한 문장) 컷 끄기")
    a = ap.parse_args(argv)

    out = a.out or capcut_draft_root()
    if not out:
        print("캡컷 드래프트 폴더를 찾을 수 없음 → --out 으로 지정", file=sys.stderr)
        return 2

    p = SilenceParams(a.noise, a.min_silence, a.pad, a.min_keep)
    t0 = time.time()
    dur = probe(a.input).duration
    sil = detect_silences(a.input, p.noise_db, p.min_silence)
    keeps = keep_ranges(sil, dur, p)
    print(f"silence  {len(sil)}개 감지 → 보존 {len(keeps)}구간  ({time.time() - t0:.1f}s)")

    tr = None
    if not a.no_asr:
        t0 = time.time()
        tr, hit = transcribe(a.input)
        print(f"asr      세그먼트 {len(tr.segments)} · 단어 {tr.word_count}{' · 캐시 hit' if hit else ''}"
              f"  ({time.time() - t0:.1f}s, {tr.backend}:{tr.model})")
        print(f"         {tr.text[:120]}{'…' if len(tr.text) > 120 else ''}")

    final = keeps
    if tr is not None:
        an = analyze(tr, not a.no_filler, not a.no_ng)
        final = apply_cuts(keeps, an.cuts, an.kept_words())
        c = an.counts()
        removed = sum(y - x for x, y in keeps) - sum(y - x for x, y in final)
        print(f"filler   잔말 {c['filler'] + c['stutter']} · NG {c['ng']} → −{removed:.1f}s")
        for cut in an.cuts:
            print(f"         {cut.kind:<8}{cut.start:7.2f}–{cut.end:<7.2f} {cut.text}")

    r = build_draft(a.input, final, str(out), a.name, tr)
    cut = r["source_sec"] - r["output_sec"]
    print(f"draft    {r['draft_dir']}")
    print(f"         {r['source_sec']}s → {r['output_sec']}s  (−{cut:.1f}s, {cut / max(r['source_sec'], 1e-9):.0%})"
          f"  · {r['segments']}컷 · 자막 {r['subtitles']} · {r['canvas']}")
    print("\n▶ 캡컷을 (재)실행해서 드래프트 목록에서 열고 직접 재생해 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
