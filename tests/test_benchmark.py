import pytest

from kodpi.dataset import load
from kodpi.detectors import attack_index, attack_probability, infer, preflight
from kodpi.io import DataError, read_csv, read_json, sha256_text, write_csv
from kodpi.dataset import DATA_FILE, SAMPLE_FIELDS
from kodpi.metrics import evaluate

from conftest import KeywordDetector


def test_dataset_counts_follow_the_spec(dataset_dir):
    samples, pairs = load(dataset_dir)
    per_seed = {s: sum(1 for x in samples if x["seed_id"] == s) for s in ("A001", "B001", "H001")}
    assert per_seed == {"A001": 28, "B001": 4, "H001": 4}
    pair_counts = {}
    for p in pairs:
        pair_counts.setdefault(p["seed_id"], {}).setdefault(p["pair_type"], 0)
        pair_counts[p["seed_id"]][p["pair_type"]] += 1
    assert pair_counts["A001"] == {"language": 7, "register": 14, "context": 12, "format": 12}
    assert pair_counts["H001"] == {"language": 1, "register": 2}


def test_failed_sentence_check_removes_only_its_pairs(dataset_dir):
    samples = read_csv(dataset_dir / DATA_FILE)
    for s in samples:  # the hae version of A001 differs from haeyo in more than its ending
        if s["seed_id"] == "A001" and s["register"] == "hae":
            s["pair_check"] = "not_ok"
    write_csv(dataset_dir / DATA_FILE, samples, SAMPLE_FIELDS)
    kept, pairs = load(dataset_dir)
    assert len(kept) == 36  # the inputs stay
    a001 = [p for p in pairs if p["seed_id"] == "A001"]
    assert not [p for p in a001 if p["pair_id"].startswith("A001.register-hae.")]
    assert len([p for p in a001 if p["pair_id"].startswith("A001.register-hasipsio.")]) == 7


def test_pairs_need_only_their_own_two_inputs(dataset_dir):
    samples = [s for s in read_csv(dataset_dir / DATA_FILE) if s["sample_id"] != "A001.ko-hae.direct"]
    write_csv(dataset_dir / DATA_FILE, samples, SAMPLE_FIELDS)
    _, pairs = load(dataset_dir)
    ids = {p["pair_id"] for p in pairs}
    assert "A001.context.ko-hae.quote" not in ids  # uses the removed input
    assert {f"A001.format.ko-hae.{r}" for r in ("translate", "quote", "explain")} <= ids  # do not


def test_dataset_rejects_inconsistent_sentence_checks(dataset_dir):
    samples = read_csv(dataset_dir / DATA_FILE)
    next(s for s in samples if s["sample_id"] == "A001.ko-hae.quote")["pair_check"] = "not_ok"
    write_csv(dataset_dir / DATA_FILE, samples, SAMPLE_FIELDS)
    with pytest.raises(DataError):
        load(dataset_dir)


def test_dataset_rejects_a_label_that_breaks_the_rule(dataset_dir):
    samples = read_csv(dataset_dir / DATA_FILE)
    samples[0]["label"] = "benign" if samples[0]["label"] == "attack" else "attack"
    write_csv(dataset_dir / DATA_FILE, samples, SAMPLE_FIELDS)
    with pytest.raises(DataError):
        load(dataset_dir)


def test_dataset_rejects_edited_text(dataset_dir):
    samples = read_csv(dataset_dir / DATA_FILE)
    samples[0]["text"] += " "
    write_csv(dataset_dir / DATA_FILE, samples, SAMPLE_FIELDS)
    with pytest.raises(DataError):
        load(dataset_dir)


def test_attack_label_lookup_and_probability():
    assert attack_index({"0": "benign", "1": "injection"}, "injection") == 1
    assert attack_index({0: "jailbreak", 1: "benign"}, "jailbreak") == 0
    with pytest.raises(DataError):
        attack_index({0: "a", 1: "b"}, "c")
    assert attack_probability([0.0, 0.0], 1) == pytest.approx(0.5)


def test_preflight_excludes_long_inputs_without_counting_them(dataset_dir, detectors_config, tmp_path):
    report = preflight(dataset_dir, detectors_config, tmp_path / "pre", detector_cls=KeywordDetector)
    assert report["eligible"]["keyword"] == 36
    assert 0 < report["eligible"]["short"] < 36
    assert report["common"] == report["eligible"]["short"]
    violations = read_csv(tmp_path / "pre" / "length_violations.csv")
    assert {v["model_key"] for v in violations} == {"short"}


def run_all(dataset_dir, detectors_config, tmp_path, **kwargs):
    preflight(dataset_dir, detectors_config, tmp_path / "pre", detector_cls=KeywordDetector)
    infer(dataset_dir, tmp_path / "pre", detectors_config, tmp_path / "pred", detector_cls=KeywordDetector)
    return evaluate(dataset_dir, tmp_path / "pred", tmp_path / "pre", tmp_path / "eval", bootstrap=200, **kwargs)


def cell(report, model, metric, **conditions):
    found = [c for c in report["models"][model] if c["metric"] == metric and c["conditions"] == conditions]
    assert len(found) == 1, (metric, conditions)
    return found[0]["value"]


def test_keyword_detector_shows_over_defense(dataset_dir, detectors_config, tmp_path):
    report = run_all(dataset_dir, detectors_config, tmp_path, common_only=False)
    m = "keyword"
    # every attack contains ignore/무시, so every attack is caught
    assert cell(report, m, "tpr", scope="all", language="ko") == 1.0
    # the hard benign sentence and every processing request also contain the keyword: all blocked
    assert cell(report, m, "fpr", scope="all", language="en", group="hard_benign") == 1.0
    assert cell(report, m, "fpr", scope="all", language="ko", group="processing") == 1.0
    assert cell(report, m, "fpr", scope="all", language="en", group="benign") == 0.0
    # it cannot tell execution from processing
    assert cell(report, m, "pair_accuracy", pair_type="context", language="ko") == 0.0
    assert cell(report, m, "pair_accuracy", pair_type="format", language="en") == 0.0
    # every context and format pair is judged attack on both sides: over-defense, not a missed attack
    for kind in ("context", "format"):
        assert cell(report, m, "pair_both_attack", pair_type=kind, language="ko") == 1.0
        assert cell(report, m, "pair_both_benign", pair_type=kind, language="en") == 0.0
    # every Korean register is judged the same way (RQ3)
    assert {cell(report, m, "tpr", scope="all", language="ko", register=r) for r in ("haeyo", "hae", "hasipsio")} == {1.0}


def test_evaluate_uses_common_inputs_by_default(dataset_dir, detectors_config, tmp_path):
    report = run_all(dataset_dir, detectors_config, tmp_path)
    common = report["common_samples"]
    assert 0 < common < 36
    n = sum(c["n"] for c in report["models"]["keyword"]
            if c["metric"] in ("tpr", "fpr") and list(c["conditions"]) == ["scope", "language"])
    assert n <= common


def test_evaluate_stops_when_a_prediction_is_missing(dataset_dir, detectors_config, tmp_path):
    preflight(dataset_dir, detectors_config, tmp_path / "pre", detector_cls=KeywordDetector)
    infer(dataset_dir, tmp_path / "pre", detectors_config, tmp_path / "pred", detector_cls=KeywordDetector)
    rows = read_csv(tmp_path / "pred" / "predictions_keyword.csv")
    write_csv(tmp_path / "pred" / "predictions_keyword.csv", rows[:-1], list(rows[0]))
    with pytest.raises(DataError):
        evaluate(dataset_dir, tmp_path / "pred", tmp_path / "pre", tmp_path / "eval", bootstrap=0)


def test_infer_resumes_without_rescoring(dataset_dir, detectors_config, tmp_path):
    preflight(dataset_dir, detectors_config, tmp_path / "pre", detector_cls=KeywordDetector)
    first = infer(dataset_dir, tmp_path / "pre", detectors_config, tmp_path / "pred", detector_cls=KeywordDetector)
    again = infer(dataset_dir, tmp_path / "pre", detectors_config, tmp_path / "pred", detector_cls=KeywordDetector)
    assert again["keyword"]["predicted"] == first["keyword"]["predicted"] == 36
    assert len(read_csv(tmp_path / "pred" / "predictions_keyword.csv")) == 36


def test_report_writes_tables_and_figures(dataset_dir, detectors_config, tmp_path):
    from kodpi.report import report
    run_all(dataset_dir, detectors_config, tmp_path, common_only=False)
    out = report(tmp_path / "eval", tmp_path / "report", detectors_config)
    assert out["figures"] == ["fig1_language.pdf", "fig2_pairs.pdf", "fig3_register.pdf"]
    for name in out["figures"]:
        assert (tmp_path / "report" / name).stat().st_size > 5_000
        assert (tmp_path / "report" / name.replace(".pdf", ".png")).exists()
    assert r"\label{fig:pairs}" in (tmp_path / "report" / "figures.tex").read_text(encoding="utf-8")
    latex = (tmp_path / "report" / "tables.tex").read_text(encoding="utf-8")
    assert latex.count(r"\toprule") == 4 and latex.count(r"\bottomrule") == 4
    assert latex.count(r"\begin{table") == latex.count(r"\end{table") == 4 and "}[t]\n\n" not in latex
    assert r"\includegraphics" in (tmp_path / "report" / "figures.tex").read_text(encoding="utf-8")
    # team tables: one numbered .xlsx each, title and description on top; the test detector "short" skips
    # long inputs, so the coverage table is the fourth
    assert out["tables"] == ["table1_rq1_language.xlsx", "table2_rq2_pairs.xlsx", "table3_rq3_register.xlsx",
                             "table4_coverage.xlsx", "tables.tex"]
    assert not (tmp_path / "report" / "tables.md").exists()
    from openpyxl import load_workbook
    sheet = load_workbook(tmp_path / "report" / "table2_rq2_pairs.xlsx").active
    values = [[c.value for c in row] for row in sheet.iter_rows()]
    assert values[0][0].startswith("표 2.") and "Pair Accuracy" in values[1][0]
    head = next(i for i, row in enumerate(values) if row[0] == "모델")
    assert values[head][:3] == ["모델", "언어", "context 번역"] and len([v for v in values[head] if v]) == 2 + 8
    body = values[head + 1:]
    assert len(body) == 2 * 2 and {row[0] for row in body} == {"keyword", "short"} and "n=" in body[0][2]


def test_coverage_table_only_when_an_input_was_excluded(dataset_dir, detectors_config, tmp_path):
    from kodpi.report import excluded_total, report
    run_all(dataset_dir, detectors_config, tmp_path, common_only=False)
    data = read_json(tmp_path / "eval" / "metrics.json")
    assert excluded_total(data) > 0  # the short-limit test detector skips long inputs
    for per_model in data["coverage"].values():
        for c in per_model.values():
            c["scored"], c["excluded"] = c["total"], 0
    (tmp_path / "all-scored").mkdir()
    from kodpi.io import write_json
    write_json(tmp_path / "all-scored" / "metrics.json", data)
    report(tmp_path / "all-scored", tmp_path / "report2", detectors_config)
    assert not (tmp_path / "report2" / "tab_coverage.tex").exists()
    assert not list((tmp_path / "report2").glob("table4_*"))
    assert not (tmp_path / "report" / "tables_appendix.tex").exists()

