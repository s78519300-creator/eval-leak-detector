"""Command line interface: ``eval-leak-detector scan``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .detector import Thresholds, scan
from .loaders import iter_corpus, load_eval
from .report import render


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval-leak-detector",
        description="Find benchmark items that leaked into a training or prompt corpus.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="check an eval set against a corpus")
    s.add_argument("--eval", required=True, dest="eval_path", help="eval set (.jsonl, .json, .csv, .tsv or .txt)")
    s.add_argument("--corpus", required=True, nargs="+", help="corpus files or directories")
    s.add_argument("--field", help="comma-separated eval fields to check (default: question/prompt/input/...)")
    s.add_argument("-n", "--ngram", type=int, default=13, help="n-gram size in words (default 13)")
    s.add_argument("--contaminated-overlap", type=float, default=0.5)
    s.add_argument("--partial-overlap", type=float, default=0.0,
                   help="minimum overlap for a partial verdict; 0 means any shared n-gram")
    s.add_argument("--contaminated-similarity", type=float, default=0.8)
    s.add_argument("--partial-similarity", type=float, default=0.5)
    s.add_argument("--format", choices=["text", "md", "json"], default="text")
    s.add_argument("--output", "-o", help="write the report to a file instead of stdout")
    s.add_argument("--show-clean", action="store_true", help="list clean items too")
    s.add_argument("--fail-above", type=float, default=None,
                   help="exit 1 if the contamination rate is above this fraction (for CI), e.g. 0 or 0.02")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    fields = [f.strip() for f in args.field.split(",")] if args.field else None
    try:
        items = load_eval(args.eval_path, fields)
        thresholds = Thresholds(args.contaminated_overlap, args.partial_overlap,
                                args.contaminated_similarity, args.partial_similarity)
        report = scan(items, iter_corpus(args.corpus), n=args.ngram, thresholds=thresholds)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = render(report, args.format, args.show_clean)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"Report written to {args.output} "
              f"(contamination rate {report.contamination_rate * 100:.1f}%)", file=sys.stderr)
    else:
        sys.stdout.write(output)

    if args.fail_above is not None and report.contamination_rate > args.fail_above:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
