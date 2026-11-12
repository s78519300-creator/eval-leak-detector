"""Render a contamination report as text, Markdown or JSON."""

from __future__ import annotations

import json

from .detector import CLEAN, CONTAMINATED, PARTIAL, TOO_SHORT, Report

_ORDER = {CONTAMINATED: 0, PARTIAL: 1, TOO_SHORT: 2, CLEAN: 3}


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def _sorted(report: Report, show_clean: bool):
    items = [i for i in report.items if show_clean or i.verdict in (CONTAMINATED, PARTIAL)]
    return sorted(items, key=lambda i: (_ORDER[i.verdict], -(i.ngram_overlap or 0), -i.near_duplicate))


def render_text(report: Report, show_clean: bool = False) -> str:
    c = report.counts
    lines = [
        f"eval-leak-detector: {len(report.items)} eval items vs {report.corpus_records} corpus records (n={report.n})",
        f"  contaminated {c[CONTAMINATED]} | partial {c[PARTIAL]} | clean {c[CLEAN]} | too short {c[TOO_SHORT]}",
        f"  contamination rate: {report.contamination_rate * 100:.1f}%",
        "",
    ]
    items = _sorted(report, show_clean)
    if items:
        lines.append(f"{'VERDICT':<13}{'N-GRAM':>7}{'NEAR-DUP':>10}  ID / MATCH")
        for i in items:
            lines.append(f"{i.verdict:<13}{_pct(i.ngram_overlap):>7}{_pct(i.near_duplicate):>10}  {i.id}")
            lines.append(f"{'':<32}  \"{i.preview}\"")
            if i.best_match:
                lines.append(f"{'':<32}  -> {i.best_match}")
    else:
        lines.append("No leaked items found.")
    return "\n".join(lines) + "\n"


def render_markdown(report: Report, show_clean: bool = False) -> str:
    c = report.counts
    out = [
        "# Eval contamination report",
        "",
        f"**{c[CONTAMINATED]} contaminated** and **{c[PARTIAL]} partial** out of {len(report.items)} eval items "
        f"(contamination rate {report.contamination_rate * 100:.1f}%), checked against {report.corpus_records} "
        f"corpus records with {report.n}-gram overlap and MinHash near-duplicate search.",
        "",
    ]
    items = _sorted(report, show_clean)
    if items:
        out += ["| Verdict | Item | N-gram overlap | Near-duplicate | Best match |", "| --- | --- | --- | --- | --- |"]
        for i in items:
            preview = i.preview.replace("|", "\\|")
            out.append(f"| {i.verdict} | `{i.id}` {preview} | {_pct(i.ngram_overlap)} | {_pct(i.near_duplicate)} | "
                       f"{i.best_match or '-'} |")
    return "\n".join(out) + "\n"


def render_json(report: Report) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n"


def render(report: Report, fmt: str, show_clean: bool = False) -> str:
    if fmt == "json":
        return render_json(report)
    if fmt in ("md", "markdown"):
        return render_markdown(report, show_clean)
    return render_text(report, show_clean)
