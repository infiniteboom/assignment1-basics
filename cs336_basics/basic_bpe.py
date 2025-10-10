import regex as re
from collections import Counter
from collections.abc import Sequence
import math

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, pattern: str = None):
        self.pattern: str = PAT if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        # vocab maps token id -> token bytes (set during training)
        self.vocab: dict[int, bytes] | None = None
        self.inverse_vocab: dict[bytes, int] | None = None
        # merge maps a pair of token ids -> rank (creation order) / token id mapping
        self.pair2rank: dict[tuple[int, int], int] | None = None
        self.pair2token: dict[tuple[int, int], int] | None = None

        self.merge = []

        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def update_inverse_vocab(self) -> None:
        assert self.vocab is not None
        self.inverse_vocab = {bt:token for token,bt in self.vocab.items()}

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        # special_tokens is a dictionary of str -> int
        for _, token in special_tokens.items():
            if self.vocab is not None and token in self.vocab:
                raise ValueError(f"special token id {token} collides with existing vocab id")
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def get_rank(self, byte_word_counter: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
        """
        Return frequency counts of adjacent pairs over a word->count map.
        Note: despite the name, this returns pair->count (not creation order).
        """
        rank: dict[tuple[int, int], int] = {}
        for byte_tuple, count in byte_word_counter.items():
            for ind in range(len(byte_tuple) - 1):
                pair = (byte_tuple[ind], byte_tuple[ind + 1])
                rank[pair] = rank.get(pair, 0) + count
        return rank

    def merge_pair(self, byte_tuple: Sequence[int], pair: tuple[int, int], new_token: int) -> tuple[int, ...]:
        new_list: list[int] = []
        i = 0
        a, b = pair
        while i < len(byte_tuple):
            if i < len(byte_tuple) - 1 and (byte_tuple[i] == a and byte_tuple[i + 1] == b):
                new_list.append(new_token)
                i += 2
            else:
                new_list.append(byte_tuple[i])
                i += 1
        return tuple(new_list)

    def update_counter(
        self,
        byte_word_counter: dict[tuple[int, ...], int],
        pair: tuple[int, int],
        new_token: int,
    ) -> dict[tuple[int, ...], int]:
        updated_counter: dict[tuple[int, ...], int] = {}
        for byte_tuple, count in byte_word_counter.items():
            new_tuple = self.merge_pair(byte_tuple, pair, new_token)
            updated_counter[new_tuple] = updated_counter.get(new_tuple, 0) + count
        return updated_counter

    def get_byte_word_counter(self, text_path: str) -> dict[tuple[int, ...], int]:
        word_counter: Counter[str] = Counter()
        with open(text_path, "r", encoding="utf-8") as file:
            line = file.readline()
            while line:
                word_counter.update(self.compiled_pattern.findall(line))
                line = file.readline()

        byte_word_counter: dict[tuple[int, ...], int] = {
            tuple(key.encode("utf-8")): value for key, value in word_counter.items()
        }
        return byte_word_counter

    def train(self, text_path: str, vocab_size: int) -> None:
        byte_word_counter: dict[tuple[int, ...], int] = self.get_byte_word_counter(text_path=text_path)
        if vocab_size < 256:
            vocab_size = 256
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        counter: dict[tuple[int, ...], int] = byte_word_counter
        pair2rank: dict[tuple[int, int], int] = {}
        pair2token: dict[tuple[int, int], int] = {}
        merge = []
        for idx in range(vocab_size - 256):
            rank = self.get_rank(counter)
            if not rank:
                break
            pair = max(rank.items(), key=lambda x: x[1])[0]
            merge.append(pair)
            a, b = pair
            new_token = 256 + idx
            vocab[new_token] = vocab[a] + vocab[b]
            pair2rank[pair] = idx
            pair2token[pair] = new_token
            counter = self.update_counter(counter, pair, new_token)
        self.vocab = vocab
        self.update_inverse_vocab()
        self.merge = merge
        self.pair2rank = pair2rank
        self.pair2token = pair2token

    def find_best_pair(self, seq: Sequence[int]) -> tuple[tuple[int, int] | None, int | None]:
        if self.pair2rank is None:
            raise RuntimeError("not train yet")

        best_score = math.inf
        best_pair = None
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            score = self.pair2rank.get(pair, math.inf)
            if score < best_score:
                best_score = score
                best_pair = pair
        merged_token = (
            self.pair2token.get(best_pair, None) if (self.pair2token is not None and best_pair is not None) else None
        )
        return best_pair, merged_token

    def encode_chunk(self, text: str) -> tuple[int, ...]:
        if self.inverse_vocab is None:
            raise RuntimeError("not train yet")
        seq: tuple[int, ...] = tuple(text.encode("utf-8"))
        seq = tuple([self.inverse_vocab[bytes([bt])] for bt in seq])
        while True:
            pair, merged_token = self.find_best_pair(seq=seq)
            if pair is None:
                break
            assert merged_token is not None
            seq = self.merge_pair(byte_tuple=seq, pair=pair, new_token=merged_token)
        return seq

    def encode_ordinary(self, text: str) -> list[int]:
        res: list[int] = []
        for m in self.compiled_pattern.finditer(text):
            chunk = m.group(0)
            if not chunk:
                continue
            seq = self.encode_chunk(chunk)
            res.extend(seq)
        return res

    def decode(self, tokens: Sequence[int]) -> str:
        if not self.vocab:
            raise RuntimeError("not train yet")
        part_bytes = []
        for ids in tokens:
            if ids in self.vocab:
                part_bytes.append(self.vocab[ids])
            elif ids in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[ids].encode('utf-8'))
            else:
                raise ValueError(f"invalid token: {ids}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode('utf-8',errors='replace')
        return text


    def encode(self, text: str) -> list[int]:
        special = self.special_tokens
        if not special:
            return self.encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
