"""
Minimal Karpathy-style regex-based BPE tokenizer extracted into a standalone
module so comparison scripts can import it instead of embedding the code.

This mirrors the essential behavior from Karpathy's educational implementation:
- same split pattern behavior (caller provides the pattern)
- same merge statistics logic
- same special token handling semantics for encode
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Set
import regex as re


def get_stats(ids: List[int], stats: Dict[Tuple[int, int], int] | None = None) -> Dict[Tuple[int, int], int]:
    """Count adjacent pair frequencies in a sequence of token ids."""
    out = stats if stats is not None else {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        out[pair] = out.get(pair, 0) + 1
    return out


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Replace all occurrences of `pair` in `ids` with the new token id `idx`."""
    a, b = pair
    new_ids: List[int] = []
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
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}

    def register_special_tokens(self, special: Dict[str, int]) -> None:
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
            stats: Dict[Tuple[int, int], int] = {}
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

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for ch in re.findall(self.compiled_pattern, text):
            ids.extend(self._encode_chunk(ch.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str | Set[str] = "none_raise") -> List[int]:
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
        ids: List[int] = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: List[int]) -> str:
        out: List[bytes] = []
        for t in ids:
            if t in self.vocab:
                out.append(self.vocab[t])
            elif t in self.inverse_special_tokens:
                out.append(self.inverse_special_tokens[t].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id {t}")
        return b"".join(out).decode("utf-8", errors="replace")

