from eval_leak_detector import CorpusIndex, Record, Thresholds, scan
from eval_leak_detector.detector import CLEAN, CONTAMINATED, PARTIAL, TOO_SHORT

LEAKED = ("A train leaves the station at 9 am travelling at 60 km per hour and a second train "
          "leaves the same station at 10 am travelling at 80 km per hour")
CLEAN_ITEM = ("Explain how a B-tree index speeds up range queries in a relational database and "
              "when a hash index would be a better choice")


def rec(text, rid="x", source="src"):
    return Record(rid, text, source)


def test_exact_copy_is_contaminated():
    report = scan([rec(LEAKED, "q1")], [rec("Homework: " + LEAKED + ". Answer: 1 pm.", source="forum:1")])
    item = report.items[0]
    assert item.verdict == CONTAMINATED
    assert item.ngram_overlap == 1.0
    assert item.best_match == "forum:1"


def test_reformatted_copy_still_matches():
    reformatted = LEAKED.upper().replace(" at ", ",  at ") + "!!!"
    report = scan([rec(LEAKED)], [rec(reformatted)])
    assert report.items[0].verdict == CONTAMINATED


def test_unrelated_item_is_clean():
    report = scan([rec(CLEAN_ITEM)], [rec(LEAKED), rec("Today I baked bread and it was great.")])
    item = report.items[0]
    assert item.verdict == CLEAN
    assert item.best_match is None


def test_single_shared_span_is_partial():
    long_item = CLEAN_ITEM + " and also describe covering indexes composite keys and the cost of writes in detail"
    corpus = [rec("blog post: " + " ".join(CLEAN_ITEM.split()[:13]) + " and then something else entirely")]
    item = scan([rec(long_item)], corpus).items[0]
    assert item.verdict == PARTIAL
    assert 0 < item.ngram_overlap < 0.5


def test_partial_threshold_can_be_raised():
    long_item = CLEAN_ITEM + " and also describe covering indexes composite keys and the cost of writes in detail"
    corpus = [rec(" ".join(CLEAN_ITEM.split()[:13]))]
    t = Thresholds(partial_overlap=0.5)
    assert scan([rec(long_item)], corpus, thresholds=t).items[0].verdict == CLEAN


def test_short_item_is_marked_too_short():
    report = scan([rec("What is the capital of France?")], [rec(LEAKED)])
    assert report.items[0].verdict == TOO_SHORT
    assert report.contamination_rate == 0.0


def test_short_exact_duplicate_caught_by_near_duplicate():
    report = scan([rec("What is the capital of France?")], [rec("what is the capital of france")])
    assert report.items[0].verdict == CONTAMINATED


def test_contamination_rate_and_counts():
    report = scan([rec(LEAKED, "a"), rec(CLEAN_ITEM, "b")], [rec(LEAKED)])
    assert report.counts[CONTAMINATED] == 1
    assert report.counts[CLEAN] == 1
    assert report.contamination_rate == 0.5
    d = report.to_dict()
    assert d["items_checked"] == 2 and d["contamination_rate"] == 0.5


def test_index_rejects_bad_band_config():
    import pytest
    with pytest.raises(ValueError):
        CorpusIndex(num_perm=100, bands=32)
