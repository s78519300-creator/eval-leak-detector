# eval-leak-detector

Is your benchmark score real, or did the model see the test?

`eval-leak-detector` checks every item in an eval set against a training corpus, a fine-tuning dataset or a RAG/prompt corpus, and tells you which items leaked. It uses two signals:

- **N-gram overlap**: the share of an item's 13-word sequences that appear anywhere in the corpus. This is the contamination test used in the GPT-3 and PaLM papers. Case, punctuation and spacing are ignored, so reformatted copies still match.
- **Near-duplicate search**: MinHash over word 3-shingles with LSH banding. It catches short items and lightly edited copies that exact n-grams miss.

Pure Python standard library. No model, no GPU, no API key.

## Quick start

```bash
git clone https://github.com/s78519300-creator/eval-leak-detector
cd eval-leak-detector
pip install -e ".[dev]"

eval-leak-detector scan --eval examples/eval.jsonl --corpus examples/corpus
```

```text
eval-leak-detector: 6 eval items vs 8 corpus records (n=13)
  contaminated 2 | partial 1 | clean 2 | too short 1
  contamination rate: 40.0%

VERDICT       N-GRAM  NEAR-DUP  ID / MATCH
contaminated    100%       85%  q1
                                  "A train leaves the station at 9 am travelling at 60 km per hour. A second train leaves the"
                                  -> examples/corpus/forum_dump.jsonl:1
contaminated    100%       65%  q5
                                  "Given a list of meeting time intervals, determine the minimum number of conference rooms r"
                                  -> examples/corpus/forum_dump.jsonl:3
partial           9%       43%  q2
                                  "Write a function that returns the length of the longest substring without repeating charac"
                                  -> examples/corpus/blog.md:5
```

`q5` was copied into a forum post with different punctuation ("time-intervals", an exclamation mark) and is still caught. `q2` shares one 13-word span with a blog post, so it is flagged as partial for a human to review.

## Verdicts

| Verdict | Rule (defaults) |
| --- | --- |
| `contaminated` | n-gram overlap ≥ 50% or near-duplicate similarity ≥ 0.8 |
| `partial` | at least one shared 13-gram, or similarity ≥ 0.5 |
| `clean` | neither |
| `too_short` | fewer than 8 words and no near-duplicate; not counted in the rate (items of 8 to 12 words are judged by similarity alone) |

Every threshold is a flag: `--contaminated-overlap`, `--partial-overlap`, `--contaminated-similarity`, `--partial-similarity`, and `-n` for the n-gram size.

## Inputs

**Eval set** (`--eval`): `.jsonl`, `.json`, `.csv`, `.tsv` or `.txt` (one item per line). The text is taken from the first of `question`, `prompt`, `input`, `problem`, `text`, `instruction`, `query`, or from the fields you name with `--field context,question`. IDs come from `id`, `task_id`, `uid` or `qid`.

**Corpus** (`--corpus`): any mix of files and directories. Each JSONL line, CSV row or text paragraph (`.txt`, `.md`, `.py`, `.html`, `.rst`) is one record, so matches point to a line number.

## Use it in CI

Block a merge when a new eval set or training shard leaks:

```bash
eval-leak-detector scan --eval evals/test.jsonl --corpus data/train/ --fail-above 0 --format md -o leak-report.md
```

Exit codes: `0` clean, `1` contamination rate above `--fail-above`, `2` bad input. The included GitHub Actions workflow runs the tests and writes the demo report to the job summary.

## Use it from Python

```python
from eval_leak_detector import iter_corpus, load_eval, scan

report = scan(load_eval("evals/test.jsonl"), iter_corpus(["data/train"]))
print(report.contamination_rate)
for item in report.items:
    if item.verdict == "contaminated":
        print(item.id, item.ngram_overlap, item.best_match)
```

## Limits

- It finds textual leakage. A paraphrase that changes most of the wording, or a translation, will not be caught.
- The n-gram index is held in memory: roughly 100 bytes per unique 13-gram. A few GB of text is fine on a laptop; for web-scale corpora, shard the corpus and merge reports.
- Near-duplicate similarity is a MinHash estimate (128 permutations, about ±0.05).

## License

MIT
