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


def test_low_fraction_relative_to_total_frames_is_rejected():
    agg = JerseyNumberAggregator()
    for reading in ["5", "5", "5"]:
        agg.add(reading)
    # 3 votes for "5" but padded out with a lot of misses/None, so the
    # fraction of frames agreeing drops below MIN_FRACTION.
    for _ in range(20):
        agg.add(None)
    assert agg.best_guess() is None


def test_split_votes_pick_the_plurality():
    agg = JerseyNumberAggregator()
    for reading in ["11", "11", "11", "4", "4"]:
        agg.add(reading)
    assert agg.best_guess() == "11"
