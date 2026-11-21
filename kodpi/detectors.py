"""Load the detectors, check input lengths, and run inference.

Inputs are given to each detector exactly as written: no truncation, no chat template
and no added system text. An input longer than a detector's token limit is skipped
for that detector and listed in length_violations.csv. It is never counted as a hit
or a miss.
"""

from __future__ import annotations

import collections
import math
import os
from pathlib import Path

from .dataset import load
from .io import canonical_json, now_iso, read_csv, read_json, require, write_csv, write_json

PREDICTION_FIELDS = ["model_key", "sample_id", "text_sha256", "token_count", "attack_score", "prediction", "logits"]


def load_specs(config_path: str | Path) -> list[dict]:
    specs = read_json(config_path)["models"]
    keys = [s["key"] for s in specs]
    require(len(keys) == len(set(keys)), "duplicate detector keys")
    return specs


def attack_index(id2label: dict | None, attack_label: str) -> int:
    labels = {int(k): v for k, v in (id2label or {}).items()}
    require(set(labels) == {0, 1}, f"expected a binary classifier, got {labels}")
    matches = [i for i, name in labels.items() if name == attack_label]
    require(len(matches) == 1, f"attack_label {attack_label!r} not in {labels}")
    return matches[0]


def attack_probability(logits: list[float], index: int) -> float:
    require(len(logits) == 2 and all(math.isfinite(v) for v in logits), f"bad logits {logits}")
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    return exps[index] / sum(exps)


class Detector:
    """One Hugging Face sequence classifier pinned to a revision."""

    def __init__(self, spec: dict, cache_dir: str | Path | None = None, offline: bool = False):
        from transformers import AutoConfig, AutoTokenizer

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        self.spec = spec
        self.key = spec["key"]
        self.options = {"revision": spec["revision"], "trust_remote_code": bool(spec.get("trust_remote_code", False)),
                        "local_files_only": offline}
        if cache_dir:
            self.options["cache_dir"] = str(cache_dir)
        self.config = AutoConfig.from_pretrained(spec["model_id"], **self.options)
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], use_fast=True, **self.options)
        self.attack_index = attack_index(getattr(self.config, "id2label", None), spec["attack_label"])
        self.max_tokens = int(spec["max_tokens"])
        self.threshold = float(spec["threshold"])
        self.model = None
        self.device = None

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def load_model(self, device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.spec["model_id"], torch_dtype=torch.float32, **self.options)
        if self.model.config.pad_token_id is None:
            require(self.tokenizer.pad_token_id is not None, f"{self.key}: no pad token")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(device).eval()
        self.device = device

    def logits(self, texts: list[str]) -> list[list[float]]:
        import torch

        encoded = self.tokenizer(texts, add_special_tokens=True, truncation=False, padding=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            return self.model(**encoded).logits.float().cpu().tolist()

    def metadata(self) -> dict:
        import torch
        import transformers

        return {"key": self.key, "model_id": self.spec["model_id"], "revision": self.spec["revision"],
                "id2label": {str(k): v for k, v in (self.config.id2label or {}).items()},
                "attack_label": self.spec["attack_label"], "attack_index": self.attack_index,
                "max_tokens": self.max_tokens, "threshold": self.threshold,
                "tokenizer_class": type(self.tokenizer).__name__,
                "torch": torch.__version__, "transformers": transformers.__version__}


def preflight(dataset_dir: str | Path, config_path: str | Path, output: str | Path,
              cache_dir: str | Path | None = None, offline: bool = False, detector_cls=Detector) -> dict:
    """Count tokens per detector and list the inputs each detector can take."""
    samples, _ = load(dataset_dir)
    output = Path(output)
    lengths, violations, metadata, eligible = [], [], {}, {}
    for spec in load_specs(config_path):
        det = detector_cls(spec, cache_dir, offline)
        metadata[det.key] = det.metadata()
        ok = []
        for s in samples:
            count = det.token_count(s["text"])
            lengths.append({"model_key": det.key, "sample_id": s["sample_id"], "token_count": count})
            if count <= det.max_tokens:
                ok.append(s["sample_id"])
            else:
                violations.append({"model_key": det.key, "sample_id": s["sample_id"], "token_count": count,
                                   "max_tokens": det.max_tokens})
        eligible[det.key] = ok
    common = sorted(set.intersection(*(set(v) for v in eligible.values())))
    write_csv(output / "token_lengths.csv", lengths, ["model_key", "sample_id", "token_count"])
    write_csv(output / "length_violations.csv", violations, ["model_key", "sample_id", "token_count", "max_tokens"])
    report = {"created_at": now_iso(), "dataset": str(dataset_dir), "samples": len(samples),
              "detectors": metadata, "eligible": {k: len(v) for k, v in eligible.items()},
              "common": len(common), "eligible_ids": eligible, "common_ids": common}
    write_json(output / "preflight.json", report)
    return report


def infer(dataset_dir: str | Path, preflight_dir: str | Path, config_path: str | Path, output: str | Path,
          cache_dir: str | Path | None = None, offline: bool = False, models: list[str] | None = None,
          device: str = "auto", batch_size: int = 8, detector_cls=Detector) -> dict:
    """Score every input a detector can take. Finished rows are kept, so a stopped run resumes."""
    samples = {s["sample_id"]: s for s in load(dataset_dir)[0]}
    eligible = read_json(Path(preflight_dir) / "preflight.json")["eligible_ids"]
    specs = [s for s in load_specs(config_path) if not models or s["key"] in models]
    require(specs, "no detector selected")
    output = Path(output)
    summary = {}
    for spec in specs:
        path = output / f"predictions_{spec['key']}.csv"
        rows = read_csv(path) if path.exists() else []
        done = {r["sample_id"] for r in rows}
        todo = sorted((samples[i] for i in eligible[spec["key"]] if i not in done), key=lambda s: len(s["text"]))
        det = detector_cls(spec, cache_dir, offline)
        det.load_model(device)
        failures = []
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            try:
                scored = list(zip(batch, det.logits([s["text"] for s in batch])))
            except Exception:  # retry one by one so a single bad input is isolated
                scored = []
                for s in batch:
                    try:
                        scored.append((s, det.logits([s["text"]])[0]))
                    except Exception as error:
                        failures.append({"model_key": det.key, "sample_id": s["sample_id"],
                                         "error": f"{type(error).__name__}: {error}"[:500]})
            for s, values in scored:
                score = attack_probability(values, det.attack_index)
                rows.append({"model_key": det.key, "sample_id": s["sample_id"], "text_sha256": s["text_sha256"],
                             "token_count": det.token_count(s["text"]), "attack_score": f"{score:.8f}",
                             "prediction": "attack" if score >= det.threshold else "benign",
                             "logits": canonical_json([round(v, 6) for v in values])})
            write_csv(path, rows, PREDICTION_FIELDS)
        write_csv(path, rows, PREDICTION_FIELDS)
        write_csv(output / f"failures_{det.key}.csv", failures, ["model_key", "sample_id", "error"])
        summary[det.key] = {"predicted": len(rows), "failed": len(failures), "device": det.device,
                            "batch_size": batch_size, "detector": det.metadata(),
                            "predictions": dict(collections.Counter(r["prediction"] for r in rows))}
        write_json(output / f"run_{det.key}.json", {"created_at": now_iso(), **summary[det.key]})
    return summary
