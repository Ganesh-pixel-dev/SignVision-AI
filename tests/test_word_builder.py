from core.word_builder import WordBuilder


def feed(wb, label, n, conf=0.95):
    for _ in range(n):
        wb.update(label, conf)


def make():
    return WordBuilder(hold_frames=5, confidence_threshold=0.85, release_frames=3)


def test_letter_added_after_hold():
    wb = make()
    feed(wb, "A", 4)
    assert wb.current_word == ""
    feed(wb, "A", 1)
    assert wb.current_word == "A"


def test_long_hold_types_once():
    wb = make()
    feed(wb, "A", 60)
    assert wb.current_word == "A"


def test_low_confidence_never_types():
    wb = make()
    feed(wb, "A", 30, conf=0.5)
    assert wb.current_word == ""


def test_single_flicker_inside_hold_does_not_double_type():
    wb = make()
    feed(wb, "A", 5)
    wb.update("B", 0.95)
    feed(wb, "A", 20)
    assert wb.current_word == "A"


def test_double_letter_needs_hand_to_leave():
    wb = make()
    feed(wb, "L", 5)
    feed(wb, None, 3, conf=0.0)
    feed(wb, "L", 5)
    assert wb.current_word == "LL"


def test_different_letters_in_sequence():
    wb = make()
    feed(wb, "A", 5)
    feed(wb, "B", 5)
    feed(wb, "C", 5)
    assert wb.current_word == "ABC"


def test_interrupted_hold_restarts_count():
    wb = make()
    feed(wb, "A", 4)
    wb.update("B", 0.95)
    feed(wb, "A", 4)
    assert wb.current_word == ""


def test_commit_returns_word_and_clears():
    wb = make()
    feed(wb, "A", 5)
    assert wb.commit_word() == "A"
    assert wb.current_word == "" and wb.history == ["A"]
    assert wb.commit_word() == ""
    assert wb.history == ["A"]


def test_backspace_and_reset():
    wb = make()
    feed(wb, "A", 5)
    feed(wb, "B", 5)
    wb.backspace()
    assert wb.current_word == "A"
    wb.reset()
    assert wb.current_word == ""
    wb.backspace()
    assert wb.current_word == ""
