from app.models.jersey_ocr import JerseyNumberAggregator


def test_no_readings_returns_none():
    agg = JerseyNumberAggregator()
    for _ in range(5):
        agg.add(None)
    assert agg.best_guess() is None


def test_consistent_majority_reading_wins():
    agg = JerseyNumberAggregator()
    for reading in ["23", "23", "23", "23", None, "7"]:
        agg.add(reading)
    assert agg.best_guess() == "23"


def test_too_few_votes_in_absolute_terms_is_rejected():
    agg = JerseyNumberAggregator()
    # Only 2 total readings, both agreeing - below MIN_VOTES=3 even though
    # it's a unanimous vote, so we shouldn't report a guess.
    agg.add("9")
    agg.add("9")
    assert agg.best_guess() is None


def test_many_unreadable_frames_dont_sink_an_otherwise_confident_vote():
    agg = JerseyNumberAggregator()
    # Most frames legitimately can't be read at all (blur/angle/occlusion) -
    # that alone shouldn't count against a number the readable frames agree
    # on. This mirrors a real clip: 24 samples, only 5 yielded any digits,
    # but 3 of those 5 agreed on "9".
    for reading in ["9", "9", "9", "0", "2"]:
        agg.add(reading)
    for _ in range(19):
        agg.add(None)
    assert agg.best_guess() == "9"


def test_scattered_disagreement_among_actual_readings_is_rejected():
    agg = JerseyNumberAggregator()
    # Of the frames that did read a digit, no number has majority agreement
    # (3 out of 9 readings = 33%, below MIN_FRACTION) - genuine disagreement,
    # not just a lot of unreadable frames, so this should stay unresolved.
    for reading in ["9", "9", "9", "0", "0", "0", "2", "2", "2"]:
        agg.add(reading)
    assert agg.best_guess() is None


def test_split_votes_pick_the_plurality():
    agg = JerseyNumberAggregator()
    for reading in ["11", "11", "11", "4", "4"]:
        agg.add(reading)
    assert agg.best_guess() == "11"
