"""Step 0: OS 감지 + 환경 점검.

    python -m capcut_agent.env_check

빠진 도구는 설치 명령만 출력한다 (자동 설치 X).
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

MIN_DISK_GB = 5


@dataclass
class Track:
    code: str      # "A" | "C" | "fallback"
    asr: str       # "mlx-whisper" | "faster-whisper"
    label: str


def detect_track() -> Track:
    system, machine = platform.system(), platform.machine()
    if system == "Darwin" and machine == "arm64":
        return Track("A", "mlx-whisper", "Mac M칩")
    if system == "Windows" and machine in ("AMD64", "x86_64"):
        return Track("C", "faster-whisper", "Windows (MP4 export 보너스)")
    return Track("fallback", "faster-whisper", f"{system} {machine}")


def capcut_draft_root() -> Path | None:
    """캡컷 드래프트 폴더. 환경변수 CAPCUT_DRAFT_ROOT 로 덮어쓰기 가능."""
    override = os.environ.get("CAPCUT_DRAFT_ROOT")
    if override:
        return Path(override).expanduser()
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Movies/CapCut/User Data/Projects/com.lveditor.draft"
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))
        return Path(local) / "CapCut/User Data/Projects/com.lveditor.draft"
    return None


def ffmpeg_exe() -> str | None:
    """시스템 ffmpeg 우선, 없으면 imageio-ffmpeg 번들 바이너리."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def main() -> int:
    track = detect_track()
    system = platform.system()
    print(f"{platform.system()} {platform.machine()}")
    print(f"▶ 트랙 {track.code} — {track.label} · ASR={track.asr}")
    print()

    missing: list[str] = []

    # Python
    ok_py = sys.version_info >= (3, 10)
    print(f"{'●' if ok_py else '○'} python {platform.python_version()}")
    if not ok_py:
        missing.append("brew install python@3.11" if system == "Darwin"
                       else "https://www.python.org/downloads/ 에서 3.11 설치")

    # brew (Mac)
    if system == "Darwin":
        has_brew = shutil.which("brew") is not None
        print(f"{'●' if has_brew else '○'} brew")
        if not has_brew:
            missing.append('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')

    # ffmpeg
    sys_ff = shutil.which("ffmpeg")
    ff = ffmpeg_exe()
    print(f"{'●' if sys_ff else ('◐' if ff else '○')} ffmpeg {sys_ff or (ff + ' (imageio 번들)' if ff else '없음')}")
    if not sys_ff:
        missing.append("brew install ffmpeg" if system == "Darwin"
                       else "winget install ffmpeg" if system == "Windows"
                       else "sudo apt install ffmpeg")

    # CapCut 드래프트 폴더
    root = capcut_draft_root()
    has_root = bool(root and root.is_dir())
    print(f"{'●' if has_root else '○'} CapCut 드래프트 폴더 {root or '(이 OS는 캡컷 미지원)'}")
    if not has_root:
        missing.append("CapCut 설치 후 1회 실행 (드래프트 폴더 생성용) — https://www.capcut.com/")

    # 디스크
    target = root if has_root else Path.home()
    free_gb = shutil.disk_usage(target).free / 1e9
    ok_disk = free_gb >= MIN_DISK_GB
    print(f"{'●' if ok_disk else '○'} 디스크 여유 {free_gb:.1f} GB (필요 {MIN_DISK_GB}GB+)")
    if not ok_disk:
        missing.append(f"디스크 {MIN_DISK_GB}GB 이상 확보")

    print()
    if missing:
        print("빠진 것 — 아래 설치 후 다시 실행:")
        for m in missing:
            print(f"  $ {m}")
        return 1
    print("환경 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
