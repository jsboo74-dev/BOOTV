#!/usr/bin/env python3
"""동영상을 원하는 구간만 잘라내고, 등장하는 얼굴을 자동으로 모자이크합니다.

사용 예:
    python blur.py input.mp4
    python blur.py input.mp4 --start 1:10 --end 1:40
    python blur.py input.mp4 -o out.mp4 --mode blur --strength 25
    python blur.py input.mp4 --no-blur --start 0:05 --end 0:20   # 자르기만
    python blur.py input.mp4 --preview                             # 인식 결과 확인용
"""

import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "blaze_face_full_range.tflite"


# ---------------------------------------------------------------- ffmpeg 도우미

def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg를 찾을 수 없습니다. `pip install imageio-ffmpeg` 또는 ffmpeg를 설치하세요.")


def parse_time(text):
    """'90', '1:30', '0:01:30.5' 형식을 초 단위로 변환합니다."""
    if text is None:
        return None
    parts = text.split(":")
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError(f"시간 형식이 올바르지 않습니다: {text}")
    try:
        seconds = 0.0
        for p in parts:
            seconds = seconds * 60 + float(p)
    except ValueError:
        raise argparse.ArgumentTypeError(f"시간 형식이 올바르지 않습니다: {text}")
    return seconds


def trim_args(start, end):
    args = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    if end is not None:
        args += ["-to", f"{end:.3f}"]
    return args


def probe(ffmpeg, src, start, end):
    """(가로, 세로, fps, 오디오 유무)를 돌려줍니다. 휴대폰 영상의 회전 정보도 반영됩니다."""
    # 첫 프레임을 BMP로 받아서 회전이 적용된 실제 크기를 얻는다.
    cmd = [ffmpeg, "-v", "error", *trim_args(start, None), "-i", str(src),
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "bmp", "-"]
    out = subprocess.run(cmd, capture_output=True)
    frame = cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        sys.exit(f"영상을 읽을 수 없습니다: {src}\n{out.stderr.decode(errors='ignore')}")
    h, w = frame.shape[:2]

    info = subprocess.run([ffmpeg, "-hide_banner", "-i", str(src)],
                          capture_output=True).stderr.decode(errors="ignore")
    m = re.search(r"Video:.*?(\d+(?:\.\d+)?) fps", info)
    fps = float(m.group(1)) if m else 30.0
    has_audio = re.search(r"Stream #.*Audio:", info) is not None
    return w, h, fps, has_audio


# ---------------------------------------------------------------- 얼굴 인식

def ensure_model():
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("얼굴 인식 모델을 내려받는 중...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class FaceFinder:
    """MediaPipe 얼굴 인식기. 작은 얼굴도 잡도록 화면을 여러 조각으로 나눠서도 검사한다."""

    def __init__(self, confidence, tiles):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            min_detection_confidence=confidence,
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self.tiles = tiles
        self.confidence = confidence

    def close(self):
        self.detector.close()

    def _detect(self, rgb, ox, oy):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        boxes = []
        for d in self.detector.detect(image).detections:
            b = d.bounding_box
            boxes.append([b.origin_x + ox, b.origin_y + oy, b.width, b.height, d.categories[0].score])
        return boxes

    def find(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        boxes = self._detect(rgb, 0, 0)
        if self.tiles > 1:
            # 작은 얼굴은 화면 전체로는 안 잡히므로 50%씩 겹치는 조각으로 나눠 확대 검사한다.
            # 조각 경계에 잘린 얼굴(오인식이 잦음)은 버린다. 겹침 덕분에 다른 조각에서 온전히 잡힌다.
            tw, th = 2 * w // (self.tiles + 1), 2 * h // (self.tiles + 1)
            margin = 2
            for i in range(self.tiles):
                for j in range(self.tiles):
                    x, y = i * tw // 2, j * th // 2
                    for bx, by, bw, bh, score in self._detect(rgb[y:y + th, x:x + tw], x, y):
                        cut = ((bx - x < margin and x > 0) or (by - y < margin and y > 0)
                               or (x + tw - (bx + bw) < margin and x + tw < w)
                               or (y + th - (by + bh) < margin and y + th < h))
                        if not cut and score >= self.confidence:
                            boxes.append([bx, by, bw, bh, score])
        return merge_boxes(boxes)


def merge_boxes(boxes, iou_limit=0.3):
    if not boxes:
        return []
    rects = [[int(x), int(y), int(bw), int(bh)] for x, y, bw, bh, _ in boxes]
    scores = [float(s) for *_, s in boxes]
    keep = cv2.dnn.NMSBoxes(rects, scores, 0.0, iou_limit)
    return [rects[i] for i in np.array(keep).flatten()]


class Tracker:
    """얼굴을 잠깐 놓친 프레임에서도 직전 위치를 몇 프레임 더 가려서 깜빡임을 막는다."""

    def __init__(self, hold):
        self.hold = hold
        self.faces = []  # [rect, 남은 프레임 수]

    def update(self, rects):
        next_faces = [[r, self.hold] for r in rects]
        for rect, left in self.faces:
            if left > 0 and not any(iou(rect, r) > 0.1 for r in rects):
                next_faces.append([rect, left - 1])
        self.faces = next_faces
        return [r for r, _ in self.faces]


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------- 가리기

def cover(frame, rect, mode, strength, padding):
    h, w = frame.shape[:2]
    x, y, bw, bh = rect
    # 머리카락·턱선까지 가리도록 상자를 넓힌다 (위쪽은 조금 더)
    px, py = int(bw * padding), int(bh * padding)
    x0, y0 = max(x - px, 0), max(y - int(py * 1.5), 0)
    x1, y1 = min(x + bw + px, w), min(y + bh + py, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    rw, rh = x1 - x0, y1 - y0

    if mode == "mosaic":
        small = cv2.resize(roi, (max(rw // strength, 1), max(rh // strength, 1)),
                           interpolation=cv2.INTER_LINEAR)
        frame[y0:y1, x0:x1] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    elif mode == "blur":
        k = max(rw, rh) // 2 * 2 + 1  # 가리는 영역 크기에 맞춘 홀수 커널
        blurred = roi
        for _ in range(max(strength // 10, 1)):
            blurred = cv2.GaussianBlur(blurred, (k, k), 0)
        frame[y0:y1, x0:x1] = blurred
    elif mode == "black":
        frame[y0:y1, x0:x1] = 0
    elif mode == "box":  # --preview 용
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)


# ---------------------------------------------------------------- 메인

def build_parser():
    p = argparse.ArgumentParser(
        description="동영상 자르기 + 얼굴 자동 모자이크",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="원본 동영상 파일")
    p.add_argument("-o", "--output", type=Path, help="저장할 파일 (기본: 원본이름_edited.mp4)")
    p.add_argument("--start", type=parse_time, help="시작 시각 (예: 1:10)")
    p.add_argument("--end", type=parse_time, help="끝 시각 (예: 1:40)")
    p.add_argument("--mode", choices=["mosaic", "blur", "black"], default="mosaic", help="가리는 방식")
    p.add_argument("--strength", type=int, default=15, help="모자이크 블록 수 분모 / 블러 세기 (클수록 강함)")
    p.add_argument("--padding", type=float, default=0.25, help="얼굴 상자를 넓히는 비율")
    p.add_argument("--confidence", type=float, default=0.5, help="얼굴 인식 기준 (낮추면 더 많이 잡지만 오인식 증가)")
    p.add_argument("--tiles", type=int, default=4, help="작은 얼굴 검출용 화면 분할 수 (1이면 분할 안 함)")
    p.add_argument("--hold", type=int, default=15, help="얼굴을 놓쳐도 계속 가릴 프레임 수")
    p.add_argument("--crf", type=int, default=20, help="화질 (낮을수록 고화질, 18~28 권장)")
    p.add_argument("--no-blur", action="store_true", help="모자이크 없이 자르기만")
    p.add_argument("--preview", action="store_true", help="가리는 대신 빨간 상자로 인식 위치만 표시")
    return p


def main():
    args = build_parser().parse_args()
    src = args.input
    if not src.exists():
        sys.exit(f"파일이 없습니다: {src}")
    if args.start is not None and args.end is not None and args.end <= args.start:
        sys.exit("--end 는 --start 보다 뒤여야 합니다.")
    if args.strength < 1:
        sys.exit("--strength 는 1 이상이어야 합니다.")
    out = args.output or src.with_name(f"{src.stem}_edited.mp4")
    ffmpeg = find_ffmpeg()
    trim = trim_args(args.start, args.end)

    if args.no_blur:
        subprocess.run([ffmpeg, "-y", "-v", "error", "-stats", *trim, "-i", str(src),
                        "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
        print(f"완료: {out}")
        return

    w, h, fps, has_audio = probe(ffmpeg, src, args.start, args.end)
    duration = None
    if args.end is not None:
        duration = args.end - (args.start or 0)
    total = int(duration * fps) if duration else None

    # 원본 → (잘라서 디코딩) → 파이썬에서 얼굴 처리 → (소리와 합쳐 인코딩) → 결과
    decoder = subprocess.Popen(
        [ffmpeg, "-v", "error", *trim, "-i", str(src), "-an",
         "-r", f"{fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    encode_cmd = [ffmpeg, "-y", "-v", "error",
                  "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
    if has_audio:
        encode_cmd += [*trim, "-i", str(src), "-map", "0:v", "-map", "1:a:0",
                       "-c:a", "aac", "-b:a", "192k", "-shortest"]
    encode_cmd += ["-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    finder = FaceFinder(args.confidence, args.tiles)
    tracker = Tracker(args.hold)
    mode = "box" if args.preview else args.mode
    frame_size = w * h * 3
    count = 0
    faces_seen = 0
    try:
        while True:
            buf = decoder.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            rects = tracker.update(finder.find(frame))
            faces_seen += bool(rects)
            for rect in rects:
                cover(frame, rect, mode, args.strength, args.padding)
            encoder.stdin.write(frame.tobytes())
            count += 1
            if count % 30 == 0:
                progress = f"{count}/{total}" if total else f"{count}"
                print(f"\r처리 중: {progress} 프레임", end="", flush=True)
    except BrokenPipeError:
        pass
    finally:
        finder.close()
        decoder.stdout.close()
        decoder.wait()
        try:
            encoder.stdin.close()
        except BrokenPipeError:
            pass
        encoder.wait()

    print()
    if encoder.returncode != 0 or count == 0:
        sys.exit("인코딩에 실패했습니다. 위의 ffmpeg 오류를 확인하세요.")
    print(f"완료: {out}  ({count} 프레임, 얼굴이 가려진 프레임 {faces_seen}개)")
    print("※ 자동 인식은 옆얼굴·작은 얼굴을 놓칠 수 있으니 공개 전에 꼭 직접 확인하세요.")


if __name__ == "__main__":
    main()
