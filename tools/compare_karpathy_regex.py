"""
Compare our BPETokenizer with a Karpathy-style RegexTokenizer under the same
regex split pattern and vocab size, then check encode/decode consistency.

This script now imports the Karpathy-style reference tokenizer from
tools/karpathy_regex_tokenizer.py to keep comparison logic separate from the
reference implementation.

Usage:
  assignment1-basics/.venv/bin/python assignment1-basics/tools/compare_karpathy_regex.py 1500
"""

import sys
import time
import importlib.util
from pathlib import Path
from typing import Any
import regex as re


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "taylorswift.txt"
OUR_IMPL = ROOT / "cs336_basics" / "basic_bpe.py"
REF_IMPL = ROOT / "tools" / "karpathy_regex_tokenizer.py"

# Use the same split pattern for both tokenizers
SPLIT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


def main(vocab_size: int = 3000) -> None:
    # load our implementation
    mod = load_module("our_bpe", OUR_IMPL)
    OurBPETok = getattr(mod, "BPETokenizer")

    # load reference Karpathy-style implementation (extracted module)
    ref_mod = load_module("karpathy_ref", REF_IMPL)
    KarpathyRegexTokenizer = getattr(ref_mod, "KarpathyRegexTokenizer")


    # read corpus once
    text = DATA.read_text(encoding="utf-8")

    # train both under the same regex pattern
    t0 = time.time()
    ours = OurBPETok(pattern=SPLIT_PATTERN)
    ours.train(str(DATA), vocab_size)
    t_ours = time.time() - t0

    t0 = time.time()
    ref = KarpathyRegexTokenizer(pattern=SPLIT_PATTERN)
    ref.train(text, vocab_size)
    t_ref = time.time() - t0

    # optional: register the same special token for both and test encode
    next_id = max(ours.vocab) + 1 if ours.vocab else 256
    specials = {"<|endoftext|>": next_id}
    ours.register_special_tokens(specials)
    ref.register_special_tokens(specials)

    samples = [
        "Hello world!",
        "你好，世界！",
        "A test with <|endoftext|> inside",
        text.splitlines()[0] if text.splitlines() else "",
    ]
    all_ok = True
    for s in samples:
        ids_a = ours.encode(s)
        ids_b = ref.encode(s, allowed_special="all")
        same_ids = list(ids_a) == list(ids_b)
        same_dec = ours.decode(ids_a) == ref.decode(ids_b) == s
        print(f"sample: {s[:40]!r} ... same_ids={same_ids} same_decode={same_dec}")
        all_ok &= same_ids and same_dec

    print("vocab_size=", vocab_size)
    print("ours_time_sec=", round(t_ours, 4))
    print("karpathy_time_sec=", round(t_ref, 4))
    print("overall_match=", all_ok)


if __name__ == "__main__":
    vs = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    main(vs)
