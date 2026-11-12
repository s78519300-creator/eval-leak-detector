"""Text normalisation, n-grams and MinHash signatures."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, List, Sequence, Set

_WORD = re.compile(r"\w+", re.UNICODE)
_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def tokenize(text: str) -> List[str]:
    """Lower-case, NFKC-normalise and split text into word tokens.

    Punctuation and whitespace differences are ignored, so "Hello,  World!"
    and "hello world" produce the same tokens.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    return _WORD.findall(text)


def stable_hash(value: str) -> int:
    """A 64-bit hash that is the same in every process (unlike ``hash``)."""
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def ngram_hashes(tokens: Sequence[str], n: int) -> Set[int]:
    """Hashes of every contiguous word n-gram in ``tokens``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return set()
    return {stable_hash(" ".join(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}


def shingles(tokens: Sequence[str], k: int = 3) -> Set[int]:
    """Word k-shingles used for near-duplicate (MinHash) matching."""
    if len(tokens) < k:
        return {stable_hash(" ".join(tokens))} if tokens else set()
    return ngram_hashes(tokens, k)


class MinHasher:
    """MinHash with universal hashing: ``(a * x + b) mod p``."""

    def __init__(self, num_perm: int = 128, seed: int = 1):
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        self.num_perm = num_perm
        params = []
        for i in range(num_perm):
            digest = hashlib.blake2b(f"{seed}:{i}".encode(), digest_size=16).digest()
            a = int.from_bytes(digest[:8], "big") % (_MERSENNE - 1) + 1
            b = int.from_bytes(digest[8:], "big") % _MERSENNE
            params.append((a, b))
        self._params = params

    def signature(self, items: Iterable[int]) -> List[int]:
        items = [x & _MAX_HASH for x in items]
        if not items:
            return [_MAX_HASH] * self.num_perm
        return [min(((a * x + b) % _MERSENNE) & _MAX_HASH for x in items) for a, b in self._params]

    @staticmethod
    def similarity(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
        if len(sig_a) != len(sig_b):
            raise ValueError("signatures must have the same length")
        if not sig_a:
            return 0.0
        return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)
