import regex as re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple
import math


class BPETokenizerList:
    """BPE tokenizer (training based on list-of-sequences approach).

    - Train: maintain a list of byte-id sequences; each step counts pairs over all
      sequences, picks the most frequent pair, assigns a new token id, then merges
      all occurrences in-place. Vocabulary stores bytes expansion for fast decode.
    - Encode: same merging logic as training, using pair2rank to choose pairs and
      pair2id to know replacement id; chunked by regex pattern to avoid crossing
      boundaries.
    """

    def __init__(self, pattern: str):
        self.PAT: str = pattern
        self._pat = re.compile(self.PAT)
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.pair2rank: Dict[Tuple[int, int], int] = {}
        self.pair2id: Dict[Tuple[int, int], int] = {}

    def _count_pairs(self, seqs: Sequence[Sequence[int]]) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = {}
        for s in seqs:
            # skip empty/length-1 sequences
            if len(s) < 2:
                continue
            for i in range(len(s) - 1):
                p = (s[i], s[i + 1])
                counts[p] = counts.get(p, 0) + 1
        return counts

    @staticmethod
    def _merge_seq(seq: Sequence[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        a, b = pair
        out: List[int] = []
        i = 0
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def train(self, input_path: str, vocab_size: int) -> None:
        # 1) Build list of sequences from corpus using regex chunks
        seqs: List[List[int]] = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                for m in self._pat.finditer(line):
                    chunk = m.group(0)
                    if not chunk:
                        continue
                    seqs.append(list(chunk.encode('utf-8')))

        # 2) BPE loop
        steps = max(0, vocab_size - 256)
        for i in range(steps):
            counts = self._count_pairs(seqs)
            if not counts:
                break
            (a, b), _ = max(counts.items(), key=lambda x: x[1])
            new_id = 256 + i
            # update vocab and maps
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.pair2rank[(a, b)] = i
            self.pair2id[(a, b)] = new_id
            # merge across all sequences
            for k, s in enumerate(seqs):
                if len(s) < 2:
                    continue
                seqs[k] = self._merge_seq(s, (a, b), new_id)

    def _find_best_pair(self, seq: Sequence[int]) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        # choose the present pair with minimal rank
        best_pair: Optional[Tuple[int, int]] = None
        best_rank = math.inf
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            r = self.pair2rank.get(p, math.inf)
            if r < best_rank:
                best_rank = r
                best_pair = p
        new_id = self.pair2id.get(best_pair) if best_pair is not None else None
        return best_pair, new_id

    def encode_chunk(self, text: str) -> List[int]:
        seq: List[int] = list(text.encode('utf-8'))
        while True:
            pair, new_id = self._find_best_pair(seq)
            if pair is None:
                break
            assert new_id is not None
            seq = self._merge_seq(seq, pair, new_id)
        return seq

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for m in self._pat.finditer(text):
            chunk = m.group(0)
            if not chunk:
                continue
            out.extend(self.encode_chunk(chunk))
        return out

    def decode(self, tokens: Sequence[int]) -> str:
        text_bytes = b"".join(self.vocab[t] for t in tokens)
        return text_bytes.decode('utf-8', errors='replace')

