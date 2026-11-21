"""Paper tables and figures from metrics.json (the output of evaluate), for an ICML or ICLR paper.

What each piece shows, in the order of the research questions:

  fig1_language  RQ1. English -> Korean change of TPR and of FPR on processing requests and hard benign
                 inputs, per detector (same sources).
  fig2_pairs     main result (RQ2). For context and format pairs, the share of pairs judged right, judged
                 attack on both sides (over-defense), judged benign on both sides (missed attack) or judged
                 the other way round, per detector and language.
  table 1        RQ1. TPR and FPR (benign, hard benign, processing requests) in English and Korean on the
                 same sources.
  table 2        RQ2. Pair Accuracy per processing request type (translate, quote, explain), per language.
  fig3_register  RQ3. TPR and FPR of Korean inputs in each register (해요체, 해체, 하십시오체), per detector.
  table 3        RQ3. The numbers behind fig3_register.
  table 4        setup. Inputs each detector could score, per language and input type; written only when some
                 input was excluded (otherwise one sentence says every input was scored).

Files
  tables.tex                        LaTeX (booktabs), English captions and labels
  figures.tex                       figure environments with captions
  fig1_*.pdf, fig1_*.png ...        vector figures (TrueType fonts embedded) and 300 dpi previews
  table1_*.xlsx, table2_*.xlsx ...  the same tables in Korean for the team, one numbered file each with its
                                    title and description on top (Excel keeps Korean intact). metrics.csv
                                    has every number.

Figures are sized to the venue's text width (ICML: 6.75 in for two columns, 3.25 in for one; ICLR and
NeurIPS: 5.5 in) and use a Times-like serif at 7-8 pt like the paper body. They carry no title; the caption does.
"""

from __future__ import annotations

from pathlib import Path

from .io import read_json, require, write_csv

# Categorical slots 1-3 of the validated default palette (all-pairs safe for three series): detectors.
# Detectors: steel blue, sand, coral. Pastel fills fail the lightness/contrast checks meant for borderless
# marks, so every bar gets a black edge; the hues pass the CVD and normal-vision separation checks.
SERIES_COLORS = ["#4E79A7", "#F2C88F", "#E06666"]
EDGE = "#1a1a1a"
# Pair outcomes in the same muted palette as the other figures (detectors are on the x axis here, so the hues
# carry the outcome): darker means worse: over-defense orange, missed attack red, reversed (both wrong) dark red; correct light grey. (face, hatch, label ink)
OUTCOME_STYLE = {"pair_accuracy": ("#D9D9D9", None, "#0b0b0b"),
                 "pair_both_attack": ("#F28E2B", None, "#0b0b0b"),
                 "pair_both_benign": ("#D62728", None, "white"),
                 "pair_reversed": ("#6B0F0F", None, "white")}
OUTCOME_NAMES = {"pair_accuracy": ("정답", "Correct"),
                 "pair_both_attack": ("둘 다 attack (과잉 방어)", "Both Attack (Over-Defense)"),
                 "pair_both_benign": ("둘 다 benign (탐지 실패)", "Both Benign (Missed Attack)"),
                 "pair_reversed": ("반대로 판정", "Reversed")}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LANGUAGES = ("en", "ko")
KO_LANG, EN_LANG = {"en": "영어", "ko": "한국어"}, {"en": "EN", "ko": "KO"}
REQUESTS = (("translate", "번역", "Transl."), ("quote", "인용", "Quote"), ("explain", "설명", "Explain"))
KIND_EN = {"context": "Context pair", "format": "Format contrast"}
REGISTERS = (("hae", "해체"), ("hasipsio", "하십시오체"))
# Where the register changes decides what kind of input it is (position -> label -> names):
#   request  the sentence itself: an attack sentence, or a benign / hard benign sentence
#   quoted   only the sentence inside the quotes of a processing request; the attack side carries the
#            execution demand
REGISTER_INPUTS = {
    ("request", "attack"): ("공격 문장", "Attack sentence", "Attack sent."),
    ("quoted", "attack"): ("처리 요청+실행 요구", "Processing request + exec.", "Proc.\\ + exec."),
    ("request", "benign"): ("일반·hard benign 문장", "Benign / hard benign sentence", "Benign sent."),
    ("quoted", "benign"): ("처리 요청", "Processing request", "Proc.\\ request"),
}
INPUT_GROUPS = [("attack", "attack", "Attack"), ("benign", "일반 benign", "Benign"),
                ("hard_benign", "hard benign", "Hard ben."), ("processing", "처리 요청", "Proc."),
                ("processing_execute", "처리+실행", "Proc.+exec")]
# Only metrics a detector cannot win by calling everything benign get a bold best value; FPR is 0 for a
# detector that never says attack.
BOLD_METRICS = {"tpr", "pair_accuracy"}
VENUES = {"icml": {"full": 6.75, "column": 3.25}, "iclr": {"full": 5.5, "column": 3.6},
          "neurips": {"full": 5.5, "column": 3.6}}
UP, DOWN = r"$\uparrow$", r"$\downarrow$"


class Cells:
    def __init__(self, report: dict):
        self.models = report["models"]

    def get(self, model: str, metric: str, **conditions) -> dict | None:
        found = [c for c in self.models[model] if c["metric"] == metric and c["conditions"] == conditions]
        if not found:
            return None
        # every metric is a rate; tables and figures show it in percent
        cell = dict(found[0], value=100 * found[0]["value"])
        if cell.get("ci95"):
            cell["ci95"] = [100 * v for v in cell["ci95"]]
        return cell


# ---- table definitions --------------------------------------------------------------------------

def col(ko: str, en: str, metric: str, **conditions) -> dict:
    return {"ko": ko, "en": en, "metric": metric, "conditions": conditions}


def per_language_rows() -> list[dict]:
    return [{"ko": KO_LANG[l], "en": EN_LANG[l], "conditions": {"language": l}} for l in LANGUAGES]


def main_tables() -> list[dict]:
    rq1 = [("TPR " + UP, [col(f"TPR {KO_LANG[l]}", EN_LANG[l], "tpr", scope="matched", language=l) for l in LANGUAGES])]
    for group, ko, en in (("benign", "일반 benign", "benign"), ("hard_benign", "hard benign", "hard benign"),
                          ("processing", "처리 요청", "processing")):
        rq1.append((f"FPR, {en} {DOWN}", [col(f"FPR {ko} {KO_LANG[l]}", EN_LANG[l], "fpr", scope="matched",
                                              language=l, group=group) for l in LANGUAGES]))
    rq2 = [(f"{KIND_EN[kind]} {UP}",
            [col(f"{kind} {ko}", en, "pair_accuracy", pair_type=kind, request=r) for r, ko, en in REQUESTS]
            + [col(f"{kind} 전체", "All", "pair_accuracy", pair_type=kind)])
           for kind in ("context", "format")]
    # RQ3: the numbers behind fig3_register, Korean inputs per register
    all_registers = (("haeyo", "해요체"), ("hae", "해체"), ("hasipsio", "하십시오체"))
    rq3 = [("TPR " + UP, [col(f"TPR {ko}", ko, "tpr", scope="all", language="ko", register=r) for r, ko in all_registers])]
    for group, ko_group, en in (("processing", "처리 요청", "processing"), ("hard_benign", "hard benign", "hard benign")):
        rq3.append((f"FPR, {en} {DOWN}", [col(f"FPR {ko_group} {ko}", ko, "fpr", scope="all", language="ko", group=group,
                                              register=r) for r, ko in all_registers]))
    return [
        {"key": "rq1_language", "groups": rq1,
         "ko_title": "표 1. 영어와 한국어의 탐지율·오탐률 — RQ1",
         "ko_note": "같은 원본의 영어 입력과 한국어 해요체 입력만 씁니다(번역 요청 제외). TPR은 높을수록, FPR은 낮을수록 "
                    "좋습니다.",
         "caption": "RQ1. Detection rate (TPR) and false positive rate (FPR) in English (EN) and Korean (KO, 해요체) on the "
                    "same source sentences, translation requests excluded. Processing: requests that only translate, "
                    "quote or explain an attack sentence (benign). Values in \\%."},
        {"key": "rq2_pairs", "groups": rq2, "rows": per_language_rows(),
         "ko_title": "표 2. 처리 유형별 Pair Accuracy — RQ2",
         "ko_note": "context: 공격 문장 ↔ 그 문장의 처리 요청. format: 처리 요청 ↔ 같은 요청 + 실행 요구. Pair Accuracy: 기본 "
                    "임계값에서 두 입력을 모두 맞힌 쌍의 비율. 높을수록 좋습니다. 한국어는 말끝 3종을 모두 씁니다.",
         "caption": "RQ2. Pair Accuracy by processing request type. Context pair: an attack sentence versus a request to "
                    "process it. Format contrast: a processing request versus the same request with a demand to carry the "
                    "sentence out. Pair Accuracy: share of pairs with both inputs judged correctly at the default threshold. Values in \\%."},
        {"key": "rq3_register", "groups": rq3,
         "ko_title": "표 3. 한국어 말투별 탐지율·오탐률 — RQ3",
         "ko_note": "한국어 입력을 말투(해요체, 해체, 하십시오체)별로 나눈 TPR과 FPR. 처리 요청은 번역·인용·설명 요청 전체이고, "
                    "말투는 따옴표 안 문장의 말끝입니다. TPR은 높을수록, FPR은 낮을수록 좋습니다.",
         "caption": "RQ3. TPR and FPR of Korean inputs in each register (해요체, 해체, 하십시오체). Processing: requests "
                    "that translate, quote or explain an attack sentence, whose quoted sentence carries the register. Values in \\%."},
    ]


def columns_of(spec):
    return [c for _, cols in spec["groups"] for c in cols]


def rows_of(spec, models, names):
    """(row label Korean, row label English, model, extra conditions) for each table row."""
    out = []
    for m in models:
        for r in spec.get("rows", [{"ko": "", "en": "", "conditions": {}}]):
            out.append((names[m], r["ko"], r["en"], m, r["conditions"]))
    return out


def grid_of(spec, cells, rows):
    return [[cells.get(m, c["metric"], **c["conditions"], **extra) for c in columns_of(spec)]
            for _, _, _, m, extra in rows]


def bold_marks(spec, rows, grid) -> list[list[bool]]:
    """Highest TPR or Pair Accuracy among the detectors, within each column and row condition."""
    marks = [[False] * len(columns_of(spec)) for _ in rows]
    for j, c in enumerate(columns_of(spec)):
        if c["metric"] not in BOLD_METRICS:
            continue
        for key in {r[2] for r in rows}:
            idx = [i for i, r in enumerate(rows) if r[2] == key and grid[i][j]]
            values = {grid[i][j]["value"] for i in idx}
            if len(values) > 1:
                for i in idx:
                    marks[i][j] = grid[i][j]["value"] == max(values)
    return marks


def column_n(rows, grid, j, row_key) -> int | None:
    ns = {grid[i][j]["n"] for i, r in enumerate(rows) if r[2] == row_key and grid[i][j]}
    return ns.pop() if len(ns) == 1 else None


# ---- Markdown and CSV (Korean, for the team) ----------------------------------------------------

def _cell_text(cell, with_n):
    if not cell:
        return "–"
    text = f"{cell['value']:.0f}" + (f" ({cell['ci95'][0]:.0f}–{cell['ci95'][1]:.0f})" if cell["ci95"] else "")
    return text + (f", n={cell['n']}" if with_n else "")


def coverage_rows(report, models, names) -> list[tuple[str, str, list[str]]]:
    out = []
    for m in models:
        for lang in LANGUAGES:
            cells = []
            for group, _, _ in INPUT_GROUPS:
                c = report["coverage"][m].get(f"{lang}|{group}")
                cells.append(f"{c['scored']}/{c['total']}" if c else "–")
            out.append((names[m], lang, cells))
    return out


def excluded_total(report) -> int:
    """Inputs some detector could not score (token limit or failure), over all detectors."""
    return sum(c["excluded"] for per_model in report["coverage"].values() for c in per_model.values())


# Sheet colors, as in the reviewer sheets: navy, sky blue and white only.
NAVY, SKY, WHITE = "1F3864", "DDEBF7", "FFFFFF"


def team_notes(report) -> list[str]:
    """Lines every team table carries under its own description."""
    scored = (f"세 탐지기 모두 입력 {report['samples']}개를 빠짐없이 처리했습니다(토큰 한도 초과·추론 실패 0개)."
              if not excluded_total(report) else
              f"비교에 쓴 입력: {report['common_samples'] if report['common_only'] else '모델별 전체'} / 전체 "
              f"{report['samples']}개. 처리하지 못한 입력 수는 표 4에 있습니다.")
    return [f"칸: 값 % (95% 신뢰구간), n=쌍 또는 입력 수. 신뢰구간은 원본 단위 부트스트랩 {report['bootstrap']}회.",
            "굵게: 같은 조건에서 가장 높은 TPR, Pair Accuracy. FPR은 모든 입력을 benign으로 판정하는 "
            "탐지기도 0이 되므로 굵게 하지 않습니다.", scored]


def write_sheet(path: Path, title: str, description: str, notes: list[str], header: list[str],
                body: list[list[str]], bold: list[list[bool]] | None = None) -> None:
    """One table per .xlsx: title, description and notes on top, then the table (navy header, sky-blue
    first columns). Excel opens it with Korean intact, unlike a CSV."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    book = Workbook()
    sheet = book.active
    sheet.title = "표"
    width = len(header)
    thin = Side(style="thin", color=NAVY)
    sheet.append([title])
    sheet.cell(1, 1).font = Font(bold=True, size=13, color=NAVY)
    for text in [description, *notes]:
        sheet.append([text])
        row = sheet.max_row
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=width)
        sheet.cell(row, 1).alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[row].height = 15 * max(1, -(-len(text) // (14 * width)))
    sheet.append([])
    sheet.append(header)
    top = sheet.max_row
    for j in range(1, width + 1):
        c = sheet.cell(top, j)
        c.font, c.fill = Font(bold=True, color=WHITE), PatternFill("solid", fgColor=NAVY)
        c.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    sheet.row_dimensions[top].height = 32
    labels = width - len(bold[0]) if bold and bold[0] else 2  # model (and language) columns
    for i, row in enumerate(body):
        sheet.append(row)
        r = sheet.max_row
        for j in range(1, width + 1):
            c = sheet.cell(r, j)
            c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
            if j <= labels:
                c.fill = PatternFill("solid", fgColor=SKY)
            else:
                c.alignment = Alignment(horizontal="right")
                if bold and bold[i][j - 1 - labels]:
                    c.font = Font(bold=True)
    for j in range(1, width + 1):
        sheet.column_dimensions[sheet.cell(top, j).column_letter].width = 15 if j <= 2 else 24
    book.save(path)


def write_team_tables(cells, models, names, report, output: Path) -> list[str]:
    """Korean tables for the team, one numbered .xlsx each (table1_..., table2_...), each with its own
    description; the paper uses the LaTeX tables."""
    for old in ["tables.md", *(p.name for p in output.glob("table_*.csv")), *(p.name for p in output.glob("table*.xlsx"))]:
        (output / old).unlink(missing_ok=True)
    written = []
    for number, spec in enumerate(main_tables(), start=1):
        rows, cols = rows_of(spec, models, names), columns_of(spec)
        grid = grid_of(spec, cells, rows)
        marks = bold_marks(spec, rows, grid)
        has_rows = "rows" in spec
        header = ["모델"] + (["언어"] if has_rows else []) + [c["ko"] for c in cols]
        body = [[name] + ([row_ko] if has_rows else []) + [_cell_text(grid[i][j], with_n=True) for j in range(len(cols))]
                for i, (name, row_ko, _, _, _) in enumerate(rows)]
        name = f"table{number}_{spec['key']}.xlsx"
        write_sheet(output / name, spec["ko_title"], spec["ko_note"], team_notes(report), header, body,
                    [[marks[i][j] for j in range(len(cols))] for i in range(len(rows))])
        written.append(name)
    if excluded_total(report):
        header = ["모델", "언어"] + [ko for _, ko, _ in INPUT_GROUPS]
        body = [[name, KO_LANG[lang], *c] for name, lang, c in coverage_rows(report, models, names)]
        name = f"table{len(written) + 1}_coverage.xlsx"
        write_sheet(output / name, f"표 {len(written) + 1}. 탐지기별 처리 가능 입력 (처리한 수 / 전체)",
                    "토큰 한도를 넘거나 추론에 실패한 입력은 오탐·미탐으로 세지 않고 뺍니다.", [], header, body)
        written.append(name)
    return written


# ---- LaTeX (English, for the paper) -------------------------------------------------------------

def latex_table(spec, cells, models, names) -> str:
    rows, cols = rows_of(spec, models, names), columns_of(spec)
    grid = grid_of(spec, cells, rows)
    marks = bold_marks(spec, rows, grid)
    has_rows = "rows" in spec
    lead = 2 if has_rows else 1
    widths = [len(c) for _, c in spec["groups"]]
    env = "table*" if len(cols) > 6 else "table"  # wide tables span both ICML columns
    label, caption = spec["key"], spec["caption"]
    lines = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}", r"\small",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{" + "l" * lead + "c" * len(cols) + "}", r"\toprule",
             " & " * lead + " & ".join(rf"\multicolumn{{{w}}}{{c}}{{{t}}}" for (t, _), w in zip(spec["groups"], widths))
             + r" \\"]
    start, rules = lead + 1, []
    for w in widths:
        rules.append(rf"\cmidrule(lr){{{start}-{start + w - 1}}}")
        start += w
    lines += [" ".join(rules), "Detector & " + ("Lang. & " if has_rows else "") + " & ".join(c["en"] for c in cols)
              + r" \\", r"\midrule"]
    for i, (name, _, row_en, model, _) in enumerate(rows):
        if has_rows and i and rows[i - 1][3] != model:
            lines.append(r"\addlinespace")
        texts = []
        for j in range(len(cols)):
            cell = grid[i][j]
            if not cell:
                texts.append("--")
                continue
            text = rf"\textbf{{{cell['value']:.0f}}}" if marks[i][j] else f"{cell['value']:.0f}"
            texts.append(text)
        first = name if not has_rows or i == 0 or rows[i - 1][3] != model else ""
        lines.append(" & ".join([first] + ([row_en] if has_rows else []) + texts) + r" \\")
    lines.append(r"\midrule")
    for key in dict.fromkeys(r[2] for r in rows):
        ns = [column_n(rows, grid, j, key) for j in range(len(cols))]
        lines.append(r"\textit{n} & " + (f"{key} & " if has_rows else "")
                     + " & ".join(rf"\textit{{{n}}}" if n else "--" for n in ns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", rf"\end{{{env}}}", ""]
    return "\n".join(lines)


def latex_coverage(report, models, names) -> str:
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Inputs each detector could score (scored / total). Inputs over a detector's token limit or "
             r"failing at inference are left out, not counted as errors; model comparisons use the inputs every "
             r"detector scored.}", r"\label{tab:coverage}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{ll" + "c" * len(INPUT_GROUPS) + "}", r"\toprule",
             "Detector & Lang. & " + " & ".join(en for _, _, en in INPUT_GROUPS) + r" \\", r"\midrule"]
    previous = None
    for name, lang, cells in coverage_rows(report, models, names):
        lines.append(" & ".join([name if name != previous else "", EN_LANG[lang]] + cells) + r" \\")
        previous = name
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def write_latex(cells, models, names, report, output: Path) -> None:
    head = ["% Generated by `kodpi report`. Needs \\usepackage{booktabs}, \\usepackage{adjustbox} and \\usepackage{kotex}.",
            f"% Inputs: {report['common_samples'] if report['common_only'] else 'per detector'} of {report['samples']}; "
            f"bootstrap: {report['bootstrap']} resamples over source sentences.",
            "% Bold: highest TPR or Pair Accuracy among detectors under the same condition. FPR is",
            "% not bolded: a detector that never says attack scores 0 on both.", ""]
    keys = [s["key"] for s in main_tables()]
    main = [latex_table(s, cells, models, names) for s in main_tables()]
    if excluded_total(report):  # the coverage table only says something when an input was left out
        keys.append("coverage")
        main.append(latex_coverage(report, models, names))
    else:
        (output / "tab_coverage.tex").unlink(missing_ok=True)
    (output / "tables.tex").write_text("\n".join(head + main), encoding="utf-8")
    for key, table in zip(keys, main):  # one file per table, to place each where the text needs it
        (output / f"tab_{key}.tex").write_text("\n".join(head[:1] + [table]), encoding="utf-8")


# ---- figures ------------------------------------------------------------------------------------

FIGURES_TEX = {
    "fig1_language": ("main", "full",
                      "English (EN, circle) versus Korean (KO, square; 해요체) on the same source sentences, "
                      "translation requests excluded. (a) TPR on attack inputs. (b) FPR on requests that only quote or "
                      "explain an attack sentence. (c) FPR on hard benign sentences. Error bars: 95\\% CI, bootstrap "
                      "over source sentences."),
    "fig2_pairs": ("main", "full",
                  "Outcomes of context pairs (an attack sentence versus a request to process it) and format contrasts "
                  "(a processing request versus the same request with a demand to carry it out), in English (EN) and "
                  "Korean (KO; all three registers). Both attack: the benign input is blocked (over-defense). Both benign: the attack is missed."),
    "fig3_register": ("main", "full",
                      "Korean inputs in 해요체 (circle), 해체 (triangle) and 하십시오체 (square), per detector. (a) TPR on "
                      "attack inputs. (b) FPR on requests that translate, quote or explain an attack sentence. (c) FPR on "
                      "hard benign sentences. Same axis as Figure~\\ref{fig:language}. Error bars: 95\\% CI, bootstrap "
                      "over source sentences."),
}


KOREAN_SERIFS = ["Nanum Myeongjo", "AppleMyungjo", "Noto Serif KR", "Noto Serif CJK KR", "Batang"]


def _available(names: list[str]) -> list[str]:
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    found = [n for n in names if n in installed]
    return found[:1] or ["DejaVu Serif"]


def paper_style() -> dict:
    # Times for Latin; Hangul falls back glyph by glyph to a Korean serif (Myeongjo) that matches it
    return {"font.family": _available(["Times New Roman", "Times", "STIXGeneral"]) + _available(KOREAN_SERIFS),
            "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
            "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
            "hatch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02, "axes.edgecolor": EDGE, "axes.labelcolor": INK, "text.color": INK,
            "xtick.color": INK, "ytick.color": INK}


def _style(ax, ylabel="", ylim=(0, 1.05)):
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#bdbdbd", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.6)


# Every panel of every figure has the same plot area, so x and y axes are the same length across figures.
# Two panels per row at the text width; a figure with more panels continues on the next row.
PANEL_COLUMNS = 2
PANEL_MARGINS = {"left": 0.5, "right": 0.1, "gap": 0.35, "top": 0.3, "bottom": 0.5, "row_gap": 0.55, "height": 1.35}


def _panels(n: int, size: dict, columns: int = PANEL_COLUMNS, y_labels_everywhere: bool = False, top: float | None = None):
    """(fig, axes): n panels of identical size, `columns` per row; y tick labels only in the first column.
    Panel height is common to all figures; the width is common to all rows of up to PANEL_COLUMNS panels."""
    import matplotlib.pyplot as plt

    m = dict(PANEL_MARGINS, **({"top": top} if top is not None else {}))  # a taller top for two-line titles
    # the common panel width; a row with more panels than PANEL_COLUMNS narrows them to stay in the text width
    per_row = max(columns, PANEL_COLUMNS)
    gap = m["left"] if y_labels_everywhere else m["gap"]  # room for each panel's own y tick labels and label
    panel_w = (size["full"] - m["left"] - m["right"] - gap * (per_row - 1)) / per_row
    width = m["left"] + columns * panel_w + (columns - 1) * gap + m["right"]
    rows = -(-n // columns)
    height = m["top"] + rows * m["height"] + (rows - 1) * m["row_gap"] + m["bottom"]
    fig = plt.figure(figsize=(width, height))
    axes = []
    for i in range(n):
        row, col = divmod(i, columns)
        left = m["left"] + col * (panel_w + gap)
        bottom = height - m["top"] - (row + 1) * m["height"] - row * m["row_gap"]
        ax = fig.add_axes([left / width, bottom / height, panel_w / width, m["height"] / height])
        if col and not y_labels_everywhere:
            ax.tick_params(labelleft=False)
        axes.append(ax)
    return fig, axes


def _panel_title(ax, title: str) -> None:
    """Centered title, and the panel letter in bold at the panel's top-left corner, apart from the title:
    "(a) Attack" -> title "Attack", letter "(a)"."""
    letter, _, rest = title.partition(" ")
    if not letter.startswith("("):
        ax.set_title(title)
        return
    ax.set_title(rest)
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-24, 6), textcoords="offset points",
                ha="left", va="bottom", fontweight="bold", fontsize=plt_rc("axes.titlesize"))


def plt_rc(key):
    import matplotlib

    return matplotlib.rcParams[key]


def _legend(fig, handles, ncol):
    legend = fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=True, bbox_to_anchor=(0.5, 1.01),
                        handlelength=1.6, columnspacing=1.6, borderaxespad=0, fancybox=False, framealpha=1)
    legend.get_frame().set_edgecolor(EDGE)
    legend.get_frame().set_linewidth(0.6)


def fig_pairs(cells, models, names, size):
    """Stacked outcome bars: x = detector, two bars (EN, KO) per detector."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = _panels(2, size)
    for ax, (kind, title) in zip(axes, (("context", "(a) Attack Sentence vs. Processing Request"), ("format", "(b) Processing Request vs. + Execute"))):
        _style(ax, "Share of pairs (%)" if kind == "context" else "", ylim=(0, 100))
        ticks, labels = [], []
        for i, model in enumerate(models):
            for k, lang in enumerate(LANGUAGES):
                x = i * 1.0 + (k - 0.5) * 0.38
                bottom = 0.0
                for metric, (face, hatch, ink) in OUTCOME_STYLE.items():
                    cell = cells.get(model, metric, pair_type=kind, language=lang)
                    value = cell["value"] if cell else 0.0
                    ax.bar(x, value, 0.34, bottom=bottom, color=face, hatch=hatch, edgecolor=EDGE, linewidth=0.6)
                    if value >= 12:  # label the segments tall enough to hold a number
                        ax.text(x, bottom + value / 2, f"{value:.0f}", ha="center", va="center", fontsize=5.5,
                                color=ink, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6}
                                if hatch else None)
                    bottom += value
                ticks.append(x)
                labels.append(EN_LANG[lang])
        ax.set_xticks(ticks, labels)
        for i, model in enumerate(models):
            ax.text(i, -0.17, names[model], ha="center", va="top", transform=ax.get_xaxis_transform(),
                    fontsize=7)
        ax.set_xlim(-0.6, len(models) - 0.4)
        _panel_title(ax, title)
    _legend(fig, [Patch(facecolor=face, hatch=hatch, edgecolor=EDGE, linewidth=0.6, label=OUTCOME_NAMES[m][1])
                  for m, (face, hatch, _) in OUTCOME_STYLE.items()], 4)
    return fig


def fig_register(cells, models, names, size):
    """RQ3: Korean 해요체 vs 해체 vs 하십시오체, per detector: TPR, FPR on processing requests, FPR on hard benign
    inputs. Same layout and axis as fig_language, so the flat register lines read against its steep language lines."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = (("tpr", {}, f"(a) Attack {UP}"),
              ("fpr", {"group": "processing"}, f"(b) Benign (Processing Request) {DOWN}"),
              ("fpr", {"group": "hard_benign"}, f"(c) Hard Benign {DOWN}"))
    registers = (("haeyo", "o", "해요체"), ("hae", "^", "해체"), ("hasipsio", "s", "하십시오체"))
    gap = 2.0
    fig, axes = _panels(3, size, columns=3, y_labels_everywhere=True)
    for ax, (metric, conditions, title) in zip(axes, panels):
        _style(ax, f"{metric.upper()} (%)", ylim=(0, 100))  # an arrow here would turn sideways with the label
        for i, model in enumerate(models):
            xs, ys = [], []
            for k, (register, marker, _) in enumerate(registers):
                cell = cells.get(model, metric, scope="all", language="ko", register=register, **conditions)
                if not cell:
                    continue
                x = i * gap + (k - 1) * 0.45
                xs.append(x)
                ys.append(cell["value"])
                if cell["ci95"]:
                    ax.errorbar(x, cell["value"], yerr=[[cell["value"] - cell["ci95"][0]], [cell["ci95"][1] - cell["value"]]],
                                fmt="none", ecolor=EDGE, elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=2)
                ax.plot(x, cell["value"], marker=marker, markersize=4.5, color=SERIES_COLORS[i % 3],
                        markeredgecolor=EDGE, markeredgewidth=0.6, linestyle="none", zorder=3)
            ax.plot(xs, ys, color=SERIES_COLORS[i % 3], linewidth=1.0, zorder=2)
        ax.set_xticks([i * gap for i in range(len(models))], [names[m].replace(" ", "\n", 1) if len(names[m]) > 8 else names[m]
                                                              for m in models])
        ax.set_xlim(-1.0, (len(models) - 1) * gap + 1.0)
        _panel_title(ax, title)
    _legend(fig, [Line2D([], [], marker=marker, color="white", markerfacecolor="white", markeredgecolor=EDGE,
                         markersize=5, label=label) for _, marker, label in registers], 3)
    for text in fig.legends[0].get_texts():  # one Korean serif for the Hangul labels
        text.set_fontfamily(_available(KOREAN_SERIFS)[:1] or None)
    return fig

# Figure files are numbered in the order the paper shows them; the LaTeX label stays the short name.
def fig_language(cells, models, names, size):
    """RQ1: English -> Korean, per detector: TPR, FPR on processing requests, FPR on hard benign inputs."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # y axis: the metric; title: the inputs it is measured on
    panels = (("tpr", {"scope": "matched"}, f"(a) Attack {UP}"),
              ("fpr", {"scope": "matched", "group": "processing"}, f"(b) Benign (Quote/Explain) {DOWN}"),
              ("fpr", {"scope": "matched", "group": "hard_benign"}, f"(c) Hard Benign {DOWN}"))
    markers, gap = {"en": "o", "ko": "s"}, 2.0
    fig, axes = _panels(3, size, columns=3, y_labels_everywhere=True)
    for ax, (metric, conditions, title) in zip(axes, panels):
        _style(ax, f"{metric.upper()} (%)", ylim=(0, 100))  # an arrow here would turn sideways with the label
        for i, model in enumerate(models):
            xs, ys = [], []
            for k, lang in enumerate(LANGUAGES):
                cell = cells.get(model, metric, language=lang, **conditions)
                if not cell:
                    continue
                x = i * gap + (k - 0.5) * 0.7
                xs.append(x)
                ys.append(cell["value"])
                if cell["ci95"]:
                    ax.errorbar(x, cell["value"], yerr=[[cell["value"] - cell["ci95"][0]], [cell["ci95"][1] - cell["value"]]],
                                fmt="none", ecolor=EDGE, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=2)
                ax.plot(x, cell["value"], marker=markers[lang], markersize=4.5, color=SERIES_COLORS[i % 3],
                        markeredgecolor=EDGE, markeredgewidth=0.6, linestyle="none", zorder=3)
                # English above its error bar, Korean below, so labels never meet the next detector's;
                # a value with no room above (near 1) or below (near 0) goes beside its point instead
                low, high = cell["ci95"] if cell["ci95"] else (cell["value"], cell["value"])
                text = f"{cell['value']:.0f}"
                if lang == "en" and high + 9 < 100:
                    ax.text(x, high + 2, text, ha="center", va="bottom", fontsize=5.5)
                elif lang == "ko" and low - 9 > 0:
                    ax.text(x, low - 2, text, ha="center", va="top", fontsize=5.5)
                else:
                    ax.text(x + (-0.2 if lang == "en" else 0.2), cell["value"], text, va="center", fontsize=5.5,
                            ha="right" if lang == "en" else "left")
            ax.plot(xs, ys, color=SERIES_COLORS[i % 3], linewidth=1.0, zorder=2)
        # narrow panels: two-line names so neighbouring labels do not touch
        ax.set_xticks([i * gap for i in range(len(models))], [names[m].replace(" ", "\n", 1) if len(names[m]) > 8 else names[m]
                                                              for m in models])
        ax.set_xlim(-1.3, (len(models) - 1) * gap + 1.3)  # room for side labels
        _panel_title(ax, title)
    _legend(fig, [Line2D([], [], marker=markers[l], color="white", markerfacecolor="white", markeredgecolor=EDGE,
                         markersize=5, label=EN_LANG[l] + (" (English)" if l == "en" else " (Korean)"))
                  for l in LANGUAGES], 2)
    return fig


DRAW = {"fig1_language": fig_language, "fig2_pairs": fig_pairs, "fig3_register": fig_register}


def write_figures(cells, models, names, output: Path, venue: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = VENUES[venue]
    written, tex = [], [f"% Generated by `kodpi report --venue {venue}`. Needs \\usepackage{{graphicx}} and "
                        "\\usepackage{kotex} (Korean in captions).", ""]
    for old in output.glob("fig*"):  # figure files from earlier versions under other names
        if old.stem not in DRAW and old.name != "figures.tex":
            old.unlink()
    with plt.rc_context(paper_style()):
        for name, (part, width, caption) in FIGURES_TEX.items():
            fig = DRAW[name](cells, models, names, size)
            fig.savefig(output / f"{name}.pdf")
            fig.savefig(output / f"{name}.png", dpi=300)
            plt.close(fig)
            written.append(f"{name}.pdf")
            env = "figure*" if venue == "icml" and width == "full" else "figure"
            block = [rf"\begin{{{env}}}[t]", r"\centering", (rf"\includegraphics{{{name}.pdf}}" if width == "natural" else rf"\includegraphics[width=\linewidth]{{{name}.pdf}}"),
                     rf"\caption{{{caption}}}", rf"\label{{fig:{name.split('_', 1)[1]}}}", rf"\end{{{env}}}", ""]
            tex += [f"% {part}"] + block
            (output / f"{name}.tex").write_text("\n".join(block), encoding="utf-8")
    (output / "figures.tex").write_text("\n".join(tex), encoding="utf-8")
    return written


# ---- entry --------------------------------------------------------------------------------------

def report(eval_dir: str | Path, output: str | Path, detectors_config: str | Path | None = None,
           venue: str = "icml") -> dict:
    require(venue in VENUES, f"venue must be one of {sorted(VENUES)}")
    metrics_path = Path(eval_dir) / "metrics.json"
    require(metrics_path.exists(), f"{metrics_path} not found; run evaluate first")
    data = read_json(metrics_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    specs = read_json(detectors_config)["models"] if detectors_config and Path(detectors_config).exists() else []
    names = {k: k for k in data["models"]} | {s["key"]: s.get("name", s["key"]) for s in specs if s["key"] in data["models"]}
    models = [s["key"] for s in specs if s["key"] in data["models"]]
    models += sorted(k for k in data["models"] if k not in models)
    cells = Cells(data)
    team = write_team_tables(cells, models, names, data, output)
    write_latex(cells, models, names, data, output)
    figures = write_figures(cells, models, names, output, venue)
    return {"output": str(output), "venue": venue, "tables": team + ["tables.tex"],
            "figures": figures, "captions": "figures.tex"}
