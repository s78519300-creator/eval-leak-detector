import json
from pathlib import Path

from eval_leak_detector.cli import main
from eval_leak_detector.loaders import iter_corpus, load_eval

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_load_eval_formats(tmp_path):
    (tmp_path / "a.csv").write_text("id,prompt\n7,hello there friend\n", encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"data": [{"question": "q one"}, "q two"]}), encoding="utf-8")
    (tmp_path / "c.txt").write_text("line one\n\nline two\n", encoding="utf-8")
    assert [r.id for r in load_eval(str(tmp_path / "a.csv"))] == ["7"]
    assert [r.text for r in load_eval(str(tmp_path / "b.json"))] == ["q one", "q two"]
    assert len(load_eval(str(tmp_path / "c.txt"))) == 2


def test_load_eval_custom_fields(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"id": 1, "context": "ctx", "ask": "why"}) + "\n", encoding="utf-8")
    assert load_eval(str(p), ["context", "ask"])[0].text == "ctx\nwhy"


def test_iter_corpus_splits_paragraphs(tmp_path):
    (tmp_path / "doc.md").write_text("para one\n\npara two\n", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    records = list(iter_corpus([str(tmp_path)]))
    assert [r.text.strip() for r in records] == ["para one", "para two"]


def test_cli_on_examples_fails_ci_gate(capsys):
    code = main(["scan", "--eval", str(EXAMPLES / "eval.jsonl"), "--corpus", str(EXAMPLES / "corpus"),
                 "--fail-above", "0"])
    out = capsys.readouterr().out
    assert code == 1
    assert "contaminated 2" in out and "q1" in out and "q5" in out


def test_cli_json_output(tmp_path):
    out = tmp_path / "r.json"
    code = main(["scan", "--eval", str(EXAMPLES / "eval.jsonl"), "--corpus", str(EXAMPLES / "corpus"),
                 "--format", "json", "-o", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert data["counts"]["contaminated"] == 2
    assert {i["id"] for i in data["items"]} == {"q1", "q2", "q3", "q4", "q5", "q6"}


def test_cli_markdown(capsys):
    main(["scan", "--eval", str(EXAMPLES / "eval.jsonl"), "--corpus", str(EXAMPLES / "corpus"), "--format", "md"])
    assert "| Verdict |" in capsys.readouterr().out


def test_cli_missing_file(capsys):
    assert main(["scan", "--eval", "nope.jsonl", "--corpus", "nope"]) == 2
