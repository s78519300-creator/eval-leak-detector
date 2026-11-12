from eval_leak_detector.text import MinHasher, ngram_hashes, shingles, stable_hash, tokenize


def test_tokenize_ignores_case_and_punctuation():
    assert tokenize("Hello,  World!") == tokenize("hello world") == ["hello", "world"]


def test_tokenize_normalises_unicode():
    assert tokenize("Ｃａｆé") == ["café"]


def test_stable_hash_is_deterministic():
    assert stable_hash("abc") == stable_hash("abc")
    assert stable_hash("abc") != stable_hash("abd")


def test_ngram_hashes_counts():
    tokens = "a b c d e".split()
    assert len(ngram_hashes(tokens, 3)) == 3
    assert ngram_hashes(tokens, 6) == set()


def test_minhash_similarity_tracks_jaccard():
    mh = MinHasher(256)
    a = shingles(tokenize("the quick brown fox jumps over the lazy dog near the river bank today"))
    b = shingles(tokenize("the quick brown fox jumps over the lazy dog near the river bank tonight"))
    c = shingles(tokenize("completely different sentence about databases and indexes and queries"))
    sim_ab = MinHasher.similarity(mh.signature(a), mh.signature(b))
    sim_ac = MinHasher.similarity(mh.signature(a), mh.signature(c))
    true_ab = len(a & b) / len(a | b)
    assert abs(sim_ab - true_ab) < 0.15
    assert sim_ac < 0.1


def test_minhash_identical_sets():
    mh = MinHasher(64)
    s = {1, 2, 3}
    assert MinHasher.similarity(mh.signature(s), mh.signature(set(s))) == 1.0
