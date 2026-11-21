"""kodpi: evaluate prompt injection detectors on the Korean over-defense benchmark.

  kodpi preflight  count tokens; list the inputs each detector can take without truncation
  kodpi infer      run the detectors on those inputs
  kodpi evaluate   TPR, FPR and pair accuracy with bootstrap confidence intervals
  kodpi report     result tables (LaTeX, Excel) and figures (PDF, PNG) for the paper
"""

from __future__ import annotations

import argparse
import json
import sys

from .io import DataError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kodpi", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", default="data")
        p.add_argument("--detectors", default="configs/detectors.json")
        p.add_argument("--output", required=True)
        p.add_argument("--cache-dir", help="Hugging Face cache folder (default: the standard HF cache)")
        p.add_argument("--offline", action="store_true", help="use only files already in the cache")
        if name == "infer":
            p.add_argument("--preflight", required=True, help="preflight output folder")
            p.add_argument("--models", nargs="+", help="detector keys to run (default: all)")
            p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
            p.add_argument("--batch-size", type=int, default=8)

    p = sub.add_parser("evaluate")
    p.add_argument("--dataset", default="data")
    p.add_argument("--predictions", required=True, help="infer output folder")
    p.add_argument("--preflight", required=True, help="preflight output folder")
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--per-model-inputs", action="store_true",
                   help="use every input each detector scored (default: only inputs all detectors scored)")

    p = sub.add_parser("report")
    p.add_argument("--metrics", required=True, help="evaluate output folder")
    p.add_argument("--detectors", default="configs/detectors.json", help="for model names and order")
    p.add_argument("--venue", default="icml", choices=("icml", "iclr", "neurips"), help="figure widths (default: icml)")
    p.add_argument("--output", required=True)
    return parser


def run(args) -> dict:
    if args.command == "preflight":
        from .detectors import preflight
        report = preflight(args.dataset, args.detectors, args.output, args.cache_dir, args.offline)
        return {k: report[k] for k in ("samples", "eligible", "common")}
    if args.command == "infer":
        from .detectors import infer
        summary = infer(args.dataset, args.preflight, args.detectors, args.output, args.cache_dir, args.offline,
                        args.models, args.device, args.batch_size)
        return {k: {"predicted": v["predicted"], "failed": v["failed"], "predictions": v["predictions"]}
                for k, v in summary.items()}
    if args.command == "report":
        from .report import report
        return report(args.metrics, args.output, args.detectors, args.venue)
    from .metrics import evaluate
    report = evaluate(args.dataset, args.predictions, args.preflight, args.output, args.bootstrap, args.seed,
                      common_only=not args.per_model_inputs)
    return {"samples": report["samples"], "common_samples": report["common_samples"], "output": args.output}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except DataError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
