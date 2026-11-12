"""Read eval sets and training corpora from common file formats."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

DEFAULT_FIELDS = ("question", "prompt", "input", "problem", "text", "instruction", "query")
CORPUS_EXTENSIONS = {".txt", ".md", ".jsonl", ".json", ".csv", ".tsv", ".py", ".html", ".rst"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox"}
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass
class Record:
    id: str
    text: str
    source: str


def _pick_text(obj, fields: Optional[Sequence[str]]) -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)
    if fields:
        parts = [str(obj[f]) for f in fields if f in obj and obj[f] is not None]
        return "\n".join(parts)
    for f in DEFAULT_FIELDS:
        if isinstance(obj.get(f), str):
            return obj[f]
    return "\n".join(str(v) for v in obj.values() if isinstance(v, str))


def _item_id(obj, fallback: str) -> str:
    if isinstance(obj, dict):
        for key in ("id", "task_id", "uid", "qid"):
            if key in obj:
                return str(obj[key])
    return fallback


def load_eval(path: str, fields: Optional[Sequence[str]] = None) -> List[Record]:
    """Load benchmark items. Supports .jsonl, .json, .csv/.tsv and .txt (one item per line)."""
    p = Path(path)
    suffix = p.suffix.lower()
    records: List[Record] = []
    if suffix == ".jsonl":
        with p.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                records.append(Record(_item_id(obj, f"line-{i}"), _pick_text(obj, fields), f"{p}:{i}"))
    elif suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        for i, obj in enumerate(data, 1):
            records.append(Record(_item_id(obj, f"item-{i}"), _pick_text(obj, fields), f"{p}#{i}"))
    elif suffix in (".csv", ".tsv"):
        with p.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ",")
            for i, row in enumerate(reader, 2):
                records.append(Record(_item_id(row, f"row-{i}"), _pick_text(row, fields), f"{p}:{i}"))
    else:
        with p.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if line.strip():
                    records.append(Record(f"line-{i}", line.strip(), f"{p}:{i}"))
    return [r for r in records if r.text.strip()]


def _walk(paths: Iterable[str]) -> Iterator[Path]:
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            yield p
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for name in sorted(files):
                    fp = Path(root) / name
                    if fp.suffix.lower() in CORPUS_EXTENSIONS:
                        yield fp
        else:
            raise FileNotFoundError(raw)


def iter_corpus(paths: Iterable[str]) -> Iterator[Record]:
    """Yield corpus records: one per JSONL line, CSV row or text paragraph."""
    for fp in _walk(paths):
        suffix = fp.suffix.lower()
        try:
            if suffix == ".jsonl":
                with fp.open(encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            obj = line
                        yield Record(f"{fp}:{i}", _pick_text(obj, None), f"{fp}:{i}")
            elif suffix in (".csv", ".tsv"):
                with fp.open(encoding="utf-8", errors="replace", newline="") as fh:
                    reader = csv.reader(fh, delimiter="\t" if suffix == ".tsv" else ",")
                    for i, row in enumerate(reader, 1):
                        yield Record(f"{fp}:{i}", " ".join(row), f"{fp}:{i}")
            else:
                text = fp.read_text(encoding="utf-8", errors="replace")
                line = 1
                for para in _PARAGRAPH.split(text):
                    if para.strip():
                        yield Record(f"{fp}:{line}", para, f"{fp}:{line}")
                    line += para.count("\n") + 2
        except UnicodeDecodeError:
            continue
