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

## 2단 — 로컬 웹

```bash
python -m capcut_agent.server      # → http://127.0.0.1:8765 자동으로 열림
```

영상을 끌어다 놓으면 `upload → silence → asr → filler → draft` 순서로 진행 상황이 표시됨 (SSE).
- asr / filler는 3·4단 전까지 `예정`으로 건너뜀
- 업로드는 content hash(sha256)로 `.cache/uploads/`에 저장 → 같은 영상을 다시 올리면 `캐시 hit`
- 단계당 최소 0.5s → 캐시 hit여도 단계 진행이 눈에 보임
- 결과 카드: 원본/결과/컷 3컬럼 + 드래프트 경로 복사 · 폴더 열기

## 3단 — 대본 추출 + 세그먼트 자막

영상 전체 대본(세그먼트 + 단어 타임스탬프)을 먼저 뽑고, 이 대본으로 자막을 만든다.

| 트랙 | 백엔드 | 기본 모델 |
|---|---|---|
| A (Mac M칩) | mlx-whisper | `mlx-community/whisper-large-v3-turbo` |
| 그 외 | faster-whisper | `large-v3-turbo` (CPU int8 / CUDA fp16) |

- 첫 실행 때 모델 다운로드(약 1.5GB). 변경: `CAPCUT_ASR_BACKEND=mlx|faster`, `CAPCUT_ASR_MODEL=...`
- 대본 캐시: `.cache/asr/<content-hash>.<backend>.<model>.json` → 같은 영상은 재추출 안 함
- whisper는 동시에 호출하면 죽을 수 있어서 서버에서 `asyncio.Lock` + 모듈 락으로 한 번에 하나만 실행
- 무음 구간 환각(no_speech_prob > 0.6 & avg_logprob < −1) 세그먼트는 버림
- 자막: 토큰을 어절로 합침 → 세로 16자 / 가로 24자 이내로 나눔. 의존명사 앞(`할 | 수`), `수/줄/적(+조사)` 뒤에서는 끊지 않음
- 자막 위치 `transform_y` = −0.55(세로) / −0.8(가로). 단위는 캔버스 높이의 절반, 음수가 아래
- CLI: `python -m capcut_agent 영상.mp4` (자막 없이 1단처럼 쓰려면 `--no-asr`)
- 테스트: `pytest -q tests`

### 캡컷에서 확인할 것
- [ ] `자막` 텍스트 트랙이 생기고 한글이 깨지지 않음
- [ ] 자막 싱크: 말과 자막이 같이 나오고 컷 이후에도 밀리지 않음
- [ ] 위치/크기 적당함 (세로 영상은 쇼츠 UI에 안 가려짐)
- [ ] `할 수 있어요` 같은 표현이 두 자막으로 쪼개지지 않음

## 4단 — 잔말 / NG 통합 컷 + 대본 노출

단어 하나만 보고 판단하지 않고 **전체 대본을 문장 단위로** 보고 결정한다 (`capcut_agent/filler.py`).

| 종류 | 규칙 |
|---|---|
| 잔말 | `음/어/으음/흠…`은 항상 컷. `그/저/뭐/이제/막`은 뒤에 0.3s 이상 쉼이 있거나 문장 끝일 때만 컷 (`그 사람`의 `그`는 유지) |
| 반복 | 같은 어절이 연속 (`그래서 그래서`) → 앞의 것 컷 |
| NG | 앞 문장이 바로 다음 문장의 앞부분과 거의 같음 = 말하다 다시 시작 → 앞 시도 컷. 앞부분만 같고 내용이 다른 문장은 유지 |
| NG 마커 | `다시 할게요`, `NG`, `처음부터` 같은 문장 → 그 문장 + 바로 앞 문장 컷 |

- 최종 컷 = 무음 기준으로 남긴 구간 − 잔말/NG 구간. 컷 때문에 생긴 말 없는 짧은 조각(<0.6s)도 같이 버려서 미세한 점프를 막음
- 자막은 잘린 단어를 자동으로 뺌
- whisper는 기본적으로 `음/어`를 지워버림 → 프롬프트에 군말을 넣어서 받아 적게 유도 (`ASR_VERSION=v2`, 이전 캐시는 무효화)
- 화면: `잔말 컷` / `NG 컷` 토글, 결과 카드에 결과 / 무음 / 잔말·NG 3칸 + 대본 (잘린 부분은 취소선)
- CLI: `--no-filler`, `--no-ng`. 실행하면 컷 목록을 출력함

### 캡컷에서 확인할 것
- [ ] 잘린 잔말 자리에서 말이 부자연스럽게 붙지 않음
- [ ] NG 문장이 통째로 빠지고 다시 말한 버전만 남음
- [ ] 필요한 말이 잘못 잘리지 않음 (특히 `그/저`, 비슷하게 시작하는 연속 문장)
- [ ] 대본 카드의 취소선과 실제 캡컷 결과가 일치함

## 로드맵
1. ✅ silence_detect + build_draft
2. ✅ FastAPI + 정적 HTML (drag/drop + SSE stepper)
3. ✅ whisper Transcript + 세그먼트 자막
4. ✅ filler/NG 통합 컷 + transcript 카드
5. 영상 프리뷰 + 보존 구간 마킹 (`[` / `]`)
