from capcut_agent.asr import Segment, Transcript, Word
from capcut_agent.subtitles import Piece, build_cues, chunk, is_dependent_noun, to_eojeols


def words(*items):
    """(text, start, end) → Word 리스트. whisper 처럼 새 어절은 앞 공백."""
    return [Word(s, e, t) for t, s, e in items]


def test_tokens_merge_into_eojeol():
    seg = Segment(0, 2, " 안녕하세요", words((" 안녕", 0.0, 0.4), ("하세요", 0.4, 0.9), (" 여러분", 1.0, 1.5)))
    es = to_eojeols(seg)
    assert [e.text for e in es] == ["안녕하세요", "여러분"]
    assert (es[0].start, es[0].end) == (0.0, 0.9)


def test_dependent_noun_detection():
    for t in ["수", "것", "거예요", "게", "때문에", "줄", "적이", "수도"]:
        assert is_dependent_noun(t), t
    for t in ["수업", "것들이다른", "여러분", "게임"]:
        assert not is_dependent_noun(t), t


def _eo(text):
    seg = Segment(0, 9, text, words(*[(" " + w, i, i + 0.5) for i, w in enumerate(text.split())]))
    return to_eojeols(seg)


def test_chunk_never_splits_before_dependent_noun_or_after_su():
    # max_chars 가 작아도 "할 수 있어요" 는 붙어 있어야 한다
    cs = chunk(_eo("저도 이제 편집을 할 수 있어요"), max_chars=8)
    texts = [" ".join(e.text for e in c) for c in cs]
    assert all(c[0].text not in ("수", "있어요") for c in cs), texts
    assert any("할 수 있어요" in t for t in texts), texts
    assert len(cs) > 1, texts   # 다른 곳에서는 정상적으로 끊긴다


def test_chunk_breaks_at_sentence_end():
    cs = chunk(_eo("좋아요. 다음으로 넘어갈게요"), max_chars=40)
    assert [" ".join(e.text for e in c) for c in cs] == ["좋아요.", "다음으로 넘어갈게요"]


def test_cues_map_through_cuts():
    # 원본: 0~1s 말, 1~3s 무음(컷), 3~4s 말
    tr = Transcript("ko", "t", "t", 4.0, [
        Segment(0, 1, " 첫 문장.", words((" 첫", 0.1, 0.4), (" 문장.", 0.4, 0.9))),
        Segment(3, 4, " 두 번째", words((" 두", 3.1, 3.4), (" 번째", 3.4, 3.9))),
    ])
    pieces = [Piece(0, 0, 1_000_000), Piece(3_000_000, 1_000_000, 1_000_000)]
    cues = build_cues(tr, pieces)
    assert [c.text for c in cues] == ["첫 문장.", "두 번째"]
    assert cues[1].start == 1_100_000           # 3.1s → 타임라인 1.1s
    assert cues[1].end == 1_900_000 or cues[1].end == 2_000_000
    assert cues[0].end <= cues[1].start


def test_words_inside_cut_are_dropped():
    tr = Transcript("ko", "t", "t", 3.0, [
        Segment(0, 3, " 음 안녕", words((" 음", 1.2, 1.6), (" 안녕", 2.1, 2.6))),
    ])
    pieces = [Piece(2_000_000, 0, 1_000_000)]   # 2~3s 만 남김
    cues = build_cues(tr, pieces)
    assert [c.text for c in cues] == ["안녕"]


def test_bind_next_with_particle():
    cs = chunk(_eo("처음 해 본 적이 있는데 생각보다 어렵더라고요"), max_chars=12)
    texts = [" ".join(e.text for e in c) for c in cs]
    assert any("적이 있는데" in t for t in texts), texts
