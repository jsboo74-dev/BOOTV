from capcut_agent.asr import Segment, Transcript, Word
from capcut_agent.filler import Cut, analyze, apply_cuts, is_retake


def seg(text, start=0.0, step=0.4, gaps=None):
    """어절마다 step 초. gaps={index: 추가 쉼(초)}."""
    ws, t = [], start
    for i, w in enumerate(text.split()):
        t += (gaps or {}).get(i, 0)
        ws.append(Word(t, t + step - 0.05, " " + w))
        t += step
    return Segment(ws[0].start, ws[-1].end, " " + text, ws)


def tr(*segs):
    return Transcript("ko", "t", "t", 60.0, list(segs))


def cut_words(a, kind=None):
    return [w.t for s in a.sentences for w in s.words if w.cut and (kind is None or w.cut == kind)]


def test_always_fillers():
    a = analyze(tr(seg("음 오늘은 어 편집을 해볼게요.")))
    assert cut_words(a) == ["음", "어"]


def test_pause_only_filler_needs_pause():
    # "그 사람" 은 지시어 → 유지.  "그... 편집" 은 쉼이 있어서 잔말
    a = analyze(tr(seg("그 사람이 그 편집을 했어요.", gaps={3: 0.5})))
    assert cut_words(a) == ["그"]
    assert a.sentences[0].words[0].cut is None      # 첫 "그"는 유지
    assert a.sentences[0].words[2].cut == "filler"  # 쉼 앞 "그"는 컷


def test_stutter():
    a = analyze(tr(seg("그래서 그래서 저는 시작했어요.")))
    assert cut_words(a, "stutter") == ["그래서"]
    assert a.sentences[0].words[1].cut is None


def test_retake_cuts_earlier_attempt():
    a = analyze(tr(
        seg("오늘은 캡컷으로 자막을", 0.0),
        seg("오늘은 캡컷으로 자막을 자동으로 넣어볼게요.", 3.0),
        seg("먼저 영상을 불러옵니다.", 7.0),
    ))
    assert [s.text() for s in a.sentences if all(w.cut == "ng" for w in s.words)] == ["오늘은 캡컷으로 자막을"]
    assert a.counts()["ng"] == 1


def test_similar_start_but_different_sentence_is_kept():
    a = analyze(tr(
        seg("그래서 저는 편집을 좋아해요.", 0.0),
        seg("그래서 저는 매일 영상을 올리고 있어요.", 3.0),
    ))
    assert a.counts()["ng"] == 0


def test_marker_cuts_itself_and_previous():
    a = analyze(tr(
        seg("이건 제가 만든 영상인데요", 0.0),
        seg("아 다시 할게요.", 3.0),
        seg("이건 제가 직접 만든 영상이에요.", 5.0),
    ))
    ng = [s.text() for s in a.sentences if all(w.cut for w in s.words)]
    assert ng == ["이건 제가 만든 영상인데요", "아 다시 할게요."]


def test_is_retake_threshold():
    assert is_retake("안녕하세요여러분오늘", "안녕하세요여러분오늘은편집", False)
    assert not is_retake("안녕", "안녕하세요", False)   # 너무 짧음


def test_apply_cuts_subtracts():
    keeps = [(0.0, 5.0), (6.0, 9.0)]
    cuts = [Cut("filler", 1.0, 1.3, "음"), Cut("ng", 6.0, 7.5, "...")]
    assert apply_cuts(keeps, cuts) == [(0.0, 1.0), (1.3, 5.0), (7.5, 9.0)]


def test_orphan_slivers_dropped():
    keeps = [(0.0, 2.2), (3.4, 6.0)]
    cuts = [Cut("ng", 0.3, 1.8, "..."), Cut("filler", 3.5, 3.8, "어")]
    words = [(0.0, 0.25), (3.85, 5.8)]
    # 1.8–2.2 (말 없는 자투리) 와 3.4–3.5 (min_keep 미만) 버림, 0–0.3 은 말이 있어서 유지
    assert apply_cuts(keeps, cuts, words) == [(0.0, 0.3), (3.8, 6.0)]
