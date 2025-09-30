"""
Compare our BPETokenizer with a minimal Karpathy-style RegexTokenizer under the
same regex split pattern and vocab size, then check encode/decode consistency.

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


# Minimal helpers mirroring Karpathy's encoder.py
def get_stats(ids: list[int], stats: dict[tuple[int, int], int] | None = None) -> dict[tuple[int, int], int]:
    out = stats if stats is not None else {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        out[pair] = out.get(pair, 0) + 1
    return out


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    a, b = pair
    new_ids: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class KarpathyRegexTokenizer:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

    def register_special_tokens(self, special: dict[str, int]) -> None:
        # basic collision check with vocab
        for tid in special.values():
            if tid in self.vocab:
                raise ValueError(f"special id {tid} collides with vocab")
        self.special_tokens = special
        self.inverse_special_tokens = {v: k for k, v in special.items()}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        chunks = re.findall(self.compiled_pattern, text)
        ids_list = [list(ch.encode("utf-8")) for ch in chunks]
        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for ids in ids_list:
                get_stats(ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids_list = [merge(ids, pair, idx) for ids in ids_list]
            self.merges[pair] = idx
            a, b = pair
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} had {stats[pair]} occurrences")

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in re.findall(self.compiled_pattern, text):
            ids.extend(self._encode_chunk(ch.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str | set[str] = "none_raise") -> list[int]:
        # align with karpathy's RegexTokenizer handling
        if not self.special_tokens:
            return self.encode_ordinary(text)
        if allowed_special == "none_raise":
            assert all(tok not in text for tok in self.special_tokens)
            return self.encode_ordinary(text)
        if allowed_special == "none":
            return self.encode_ordinary(text)
        if allowed_special != "all" and not isinstance(allowed_special, set):
            raise ValueError("allowed_special must be 'all'|'none'|'none_raise' or a set")
        if allowed_special == "all":
            special = self.special_tokens
        else:
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        if not special:
            return self.encode_ordinary(text)
        pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        parts = re.split(pattern, text)
        ids: list[int] = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[bytes] = []
        for t in ids:
            if t in self.vocab:
                out.append(self.vocab[t])
            elif t in self.inverse_special_tokens:
                out.append(self.inverse_special_tokens[t].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id {t}")
        return b"".join(out).decode("utf-8", errors="replace")


def main(vocab_size: int = 3000) -> None:
    # load our implementation
    mod = load_module("our_bpe", OUR_IMPL)
    OurBPETok = getattr(mod, "BPETokenizer")

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

