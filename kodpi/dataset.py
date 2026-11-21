"""The benchmark file and its rules.

kod_pi_benchmark_dataset.csv  one row per input; the only data file
  sample_id    unique id: {seed}.{en | ko-<register>}.{request}[+exec], e.g. A001.ko-hae.quote+exec
  text         the input exactly as given to a detector
  label        attack | benign
  source_kind  attack | benign | hard_benign (kind of the source sentence)
  language     en | ko
  register     none (English) | haeyo | hae | hasipsio
  request      direct | translate | quote | explain
               direct: the sentence itself; the others wrap the sentence in a processing request
  execute      1 when the processing request ends with a demand to carry out the quoted sentence
  attack_type  PIDS-Bench attack type (attack sources only)
  pair_check   check of the Korean sentence this input is built from (ko rows only): ending-only rule,
               majority of four LLM judges, or two human reviewers on a tie
               haeyo row: same meaning as the English source; hae / hasipsio row: differs from the
               haeyo sentence only in sentence endings. ok | not_ok
  seed_id      the PIDS-Bench source sentence the input was built from (A001, B001, H001, ...)
  pids_parent_seed_id, pids_file, pids_row   where the source sentence is in PIDS-Bench v3
  text_sha256  sha256 of text

Comparison pairs are not stored; pairs() builds them from the ids by fixed rules:
  language  en <-> ko haeyo, same request and execute     expected same       (haeyo pair_check ok)
  register  ko haeyo <-> ko hae | hasipsio, same request   expected same       (variant pair_check ok)
  context   direct attack <-> processing request          expected different
  format    processing request <-> + execute demand       expected different
A pair exists only when both of its inputs are in the file.
"""

from __future__ import annotations

from pathlib import Path

from .io import read_csv, require, sha256_text

SAMPLE_FIELDS = ["sample_id", "text", "label", "source_kind", "language", "register", "request", "execute",
                 "attack_type", "pair_check", "seed_id", "pids_parent_seed_id", "pids_file", "pids_row",
                 "text_sha256"]
DATA_FILE = "kod_pi_benchmark_dataset.csv"
PAIR_FIELDS = ["pair_id", "seed_id", "pair_type", "base_id", "variant_id", "expected"]

ALLOWED = {
    "source_kind": {"attack", "benign", "hard_benign"},
    "language": {"en", "ko"},
    "register": {"none", "haeyo", "hae", "hasipsio"},
    "request": {"direct", "translate", "quote", "explain"},
    "execute": {"0", "1"},
    "label": {"attack", "benign"},
}
PAIR_TYPES = {"language": "same", "register": "same", "context": "different", "format": "different"}
REQUESTS = ("translate", "quote", "explain")


def input_group(s: dict) -> str:
    """What an input is, for reporting: the source kind for direct inputs, processing for a processing
    request around an attack sentence (the over-defense case), processing_execute when the request also
    demands execution."""
    if s["request"] == "direct":
        return s["source_kind"]
    return "processing_execute" if s["execute"] == "1" else "processing"


def form(language: str, register: str) -> str:
    return "en" if language == "en" else f"ko-{register}"


def tag(request: str, execute: str) -> str:
    return request + ("+exec" if execute == "1" else "")


def sample_id(seed: str, language: str, register: str, request: str, execute: str) -> str:
    return f"{seed}.{form(language, register)}.{tag(request, execute)}"


def pairs(samples: list[dict]) -> list[dict]:
    """All comparison pairs whose two inputs are present, by the rules in the module docstring."""
    by_id = {s["sample_id"]: s for s in samples}
    out = []

    def add(pair_id, pair_type, base, variant, check=None):
        if base in by_id and variant in by_id and (check is None or by_id[check]["pair_check"] == "ok"):
            out.append({"pair_id": pair_id, "seed_id": by_id[base]["seed_id"], "pair_type": pair_type,
                        "base_id": base, "variant_id": variant, "expected": PAIR_TYPES[pair_type]})

    for s in samples:
        seed, request, execute = s["seed_id"], s["request"], s["execute"]
        t = tag(request, execute)
        if s["language"] == "ko" and s["register"] == "haeyo":
            add(f"{seed}.language.{t}", "language", sample_id(seed, "en", "none", request, execute), s["sample_id"],
                check=s["sample_id"])
            for register in ("hae", "hasipsio"):
                variant = sample_id(seed, "ko", register, request, execute)
                add(f"{seed}.register-{register}.{t}", "register", s["sample_id"], variant, check=variant)
        if s["source_kind"] == "attack" and request != "direct" and execute == "0":
            # each pair is built from its processing request, so it only needs its own two inputs: a removed
            # direct input must not take the format pairs of the same sentence with it
            f = form(s["language"], s["register"])
            add(f"{seed}.context.{f}.{request}", "context", sample_id(seed, s["language"], s["register"], "direct", "0"),
                s["sample_id"])
            add(f"{seed}.format.{f}.{request}", "format", s["sample_id"],
                sample_id(seed, s["language"], s["register"], request, "1"))
    return out


def check_samples(samples: list[dict]) -> None:
    require(samples, "no samples")
    seen, checks = set(), {}
    for s in samples:
        sid = s.get("sample_id", "?")
        require(set(SAMPLE_FIELDS) <= set(s), f"{DATA_FILE} needs columns {SAMPLE_FIELDS}")
        for field, allowed in ALLOWED.items():
            require(s[field] in allowed, f"{sid}: {field}={s[field]!r} not in {sorted(allowed)}")
        require(sid not in seen, f"duplicate sample_id {sid}")
        seen.add(sid)
        require(sid == sample_id(s["seed_id"], s["language"], s["register"], s["request"], s["execute"]),
                f"{sid}: id does not match its seed, language, register, request and execute")
        require(sha256_text(s["text"]) == s["text_sha256"], f"{sid}: text_sha256 does not match text")
        require((s["language"] == "en") == (s["register"] == "none"), f"{sid}: register must be none only for en")
        require(s["execute"] == "0" or s["request"] != "direct", f"{sid}: execute needs a processing request")
        expected = "attack" if s["execute"] == "1" or (s["request"] == "direct" and s["source_kind"] == "attack") \
            else "benign"
        require(s["label"] == expected, f"{sid}: label {s['label']} breaks the construction rule")
        if s["language"] == "ko":
            require(s["pair_check"] in ("ok", "not_ok"), f"{sid}: pair_check must be ok or not_ok for Korean")
            key = (s["seed_id"], s["register"])
            require(checks.setdefault(key, s["pair_check"]) == s["pair_check"],
                    f"{sid}: pair_check differs between inputs built from the same Korean sentence")
        else:
            require(s["pair_check"] == "", f"{sid}: pair_check is only for Korean inputs")


def load(dataset_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """(samples, pairs) from dataset_dir/DATA_FILE, after checking every rule."""
    samples = read_csv(Path(dataset_dir) / DATA_FILE)
    check_samples(samples)
    return samples, pairs(samples)
