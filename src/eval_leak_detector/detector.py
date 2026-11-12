"""Index a corpus and score eval items for contamination."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .loaders import Record
from .text import MinHasher, ngram_hashes, shingles, tokenize

CLEAN, PARTIAL, CONTAMINATED, TOO_SHORT = "clean", "partial", "contaminated", "too_short"


@dataclass
class Thresholds:
    contaminated_overlap: float = 0.5
    partial_overlap: float = 0.0  # any shared n-gram counts, as in the GPT-3 contamination study
    contaminated_similarity: float = 0.8
    partial_similarity: float = 0.5


@dataclass
class ItemResult:
    id: str
    source: str
    preview: str
    tokens: int
    ngram_overlap: Optional[float]
    near_duplicate: float
    verdict: str
    best_match: Optional[str] = None
    matched_ngrams: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    n: int
    corpus_records: int
    items: List[ItemResult] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        c = Counter(i.verdict for i in self.items)
        return {k: c.get(k, 0) for k in (CONTAMINATED, PARTIAL, CLEAN, TOO_SHORT)}

    @property
    def contamination_rate(self) -> float:
        scored = [i for i in self.items if i.verdict != TOO_SHORT]
        if not scored:
            return 0.0
        return sum(1 for i in scored if i.verdict == CONTAMINATED) / len(scored)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "corpus_records": self.corpus_records,
            "items_checked": len(self.items),
            "counts": self.counts,
            "contamination_rate": round(self.contamination_rate, 4),
            "items": [i.to_dict() for i in self.items],
        }


class CorpusIndex:
    """N-gram index plus MinHash LSH buckets over corpus records.

    * Exact leakage: share of an eval item's word n-grams that appear anywhere
      in the corpus (the test used for GPT-3 / PaLM style contamination checks).
    * Near duplicates: MinHash over word 3-shingles with LSH banding, which
      catches paraphrase-light copies such as re-punctuated or lightly edited items.
    """

    def __init__(self, n: int = 13, num_perm: int = 128, bands: int = 32, min_tokens: int = 8):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.n = n
        self.min_tokens = min_tokens
        self.bands = bands
        self.rows = num_perm // bands
        self.hasher = MinHasher(num_perm)
        self._grams: Dict[int, int] = {}
        self._buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
        self._sigs: List[List[int]] = []
        self._sources: List[str] = []

    def __len__(self) -> int:
        return len(self._sources)

    def add(self, record: Record) -> None:
        tokens = tokenize(record.text)
        if not tokens:
            return
        rid = len(self._sources)
        self._sources.append(record.source)
        for h in ngram_hashes(tokens, self.n):
            self._grams.setdefault(h, rid)
        sig = self.hasher.signature(shingles(tokens))
        self._sigs.append(sig)
        for b in range(self.bands):
            self._buckets[(b, tuple(sig[b * self.rows : (b + 1) * self.rows]))].append(rid)

    def add_all(self, records: Iterable[Record]) -> "CorpusIndex":
        for r in records:
            self.add(r)
        return self

    def _near_duplicate(self, sig: List[int]) -> Tuple[float, Optional[int]]:
        candidates = set()
        for b in range(self.bands):
            candidates.update(self._buckets.get((b, tuple(sig[b * self.rows : (b + 1) * self.rows])), ()))
        best, best_id = 0.0, None
        for rid in candidates:
            s = MinHasher.similarity(sig, self._sigs[rid])
            if s > best:
                best, best_id = s, rid
        return best, best_id

    def check(self, item: Record, thresholds: Thresholds = Thresholds()) -> ItemResult:
        tokens = tokenize(item.text)
        preview = " ".join(item.text.split())[:90]
        sig = self.hasher.signature(shingles(tokens))
        sim, sim_id = self._near_duplicate(sig) if tokens else (0.0, None)

        overlap: Optional[float] = None
        matched = 0
        gram_best: Optional[int] = None
        grams = ngram_hashes(tokens, self.n)
        if grams:
            hits = Counter(self._grams[h] for h in grams if h in self._grams)
            matched = sum(hits.values())
            overlap = matched / len(grams)
            if hits:
                gram_best = hits.most_common(1)[0][0]

        t = thresholds
        if overlap is None and len(tokens) < self.min_tokens and sim < t.contaminated_similarity:
            verdict = TOO_SHORT
        elif (overlap or 0) >= t.contaminated_overlap or sim >= t.contaminated_similarity:
            verdict = CONTAMINATED
        elif (matched and (overlap or 0) >= t.partial_overlap) or sim >= t.partial_similarity:
            verdict = PARTIAL
        else:
            verdict = CLEAN

        best_id = gram_best if gram_best is not None and (overlap or 0) >= sim else sim_id
        if verdict in (CLEAN, TOO_SHORT):
            best_id = None
        return ItemResult(
            id=item.id,
            source=item.source,
            preview=preview,
            tokens=len(tokens),
            ngram_overlap=None if overlap is None else round(overlap, 4),
            near_duplicate=round(sim, 4),
            verdict=verdict,
            best_match=self._sources[best_id] if best_id is not None else None,
            matched_ngrams=matched,
        )


def scan(eval_items: Iterable[Record], corpus: Iterable[Record], n: int = 13,
         thresholds: Thresholds = Thresholds(), num_perm: int = 128, bands: int = 32) -> Report:
    index = CorpusIndex(n=n, num_perm=num_perm, bands=bands).add_all(corpus)
    report = Report(n=n, corpus_records=len(index))
    for item in eval_items:
        report.items.append(index.check(item, thresholds))
    return report
