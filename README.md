# 캡컷 에이전트

한국어 토킹 영상(mp4/mov) → 캡컷 드래프트 자동 편집기.
무음·잔말·NG 컷 + 단어별 자막. 검증 기준은 **캡컷에서 직접 재생**.

## Step 0 — 환경 점검

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m capcut_agent.env_check
```

| OS | 트랙 | ASR |
|---|---|---|
| Darwin arm64 | A | mlx-whisper |
| Windows AMD64 | C | faster-whisper (+ MP4 export) |
| 그 외 | fallback | faster-whisper |

## 1단 — 무음 점프컷 드래프트

```bash
python -m capcut_agent 내영상.mp4
# 옵션: --noise -35  --min-silence 0.45  --pad 0.12  --min-keep 0.25  --out DIR  --name 이름
```

드래프트는 캡컷 드래프트 폴더에 생성된다 (`CAPCUT_DRAFT_ROOT` 환경변수로 변경 가능).
캡컷을 **완전히 종료 후 재실행** → 목록에서 열기 → 재생해서 확인.

### 확인 체크리스트
- [ ] 드래프트가 목록에 뜨고 썸네일·길이 표시됨
- [ ] 열었을 때 미디어 오프라인(빨간 화면) 없음
- [ ] 컷 경계에서 말끝/말머리 잘림 없음 (잘리면 `--pad` ↑)
- [ ] 무음이 덜 잘리면 `--noise` ↑(예: -30), 너무 잘리면 ↓(예: -40)

## 로드맵
1. ✅ silence_detect + build_draft
2. FastAPI + 정적 HTML (drag/drop + SSE stepper)
3. whisper Transcript + 세그먼트 자막
4. filler/NG 통합 컷 + transcript 카드
5. 영상 프리뷰 + 보존 구간 마킹 (`[` / `]`)
