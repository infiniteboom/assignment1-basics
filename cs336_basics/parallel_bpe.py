import math
from collections import Counter
from collections.abc import Sequence
import regex as re
import os
from typing import BinaryIO
from typing import Generator
from pathlib import Path
import multiprocessing as mp
from multiprocessing.pool import Pool
from functools import partial

g_compiled_pattern = None
g_special_pattern = None
g_special_token_set = None

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def read_chunks(
        path:Path,
        num_chunks:int,
        split_token:bytes | None,
) -> Generator[str,None,None]:
    with open(path,"rb") as fh:
        if split_token is None:
            fh.seek(0,2)
            boundaries = [0,fh.tell()]
        else:
            boundaries = find_chunk_boundaries(fh,num_chunks,split_token)

        for start,end in zip(boundaries[:-1],boundaries[1:]):
            fh.seek(start)
            chunk_bytes = fh.read(end - start)
            if not chunk_bytes:
                continue
            yield chunk_bytes.decode("utf-8",errors="ignore")

def _init_worker(pattern_str,special_tokens:list[str]):
    global g_compiled_pattern,g_special_pattern,g_special_token_set

    g_compiled_pattern = re.compile(pattern=pattern_str)
    if special_tokens:
        tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern_str = "(" + "|".join(re.escape(tok) for tok in tokens) + ")"
        g_special_pattern = re.compile(special_pattern_str)
        g_special_token_set = set(special_tokens)

def _process_chunk(task_args:tuple) -> Counter:
    """
    这是一个独立的 worker 函数，负责处理单个文本块。
    """
    text_path,start,end = task_args

    with open(text_path,'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end-start)

        if not chunk_bytes:
            return Counter()
        
        text_chunk = chunk_bytes.decode("utf-8",errors="ignore")

    counter = Counter()
    if g_special_pattern is None:
        counter.update(g_compiled_pattern.findall(text_chunk))
    else:
        for part in g_special_pattern.split(text_chunk):
            if not part or (g_special_token_set and part in g_special_token_set):
                continue
            counter.update(g_compiled_pattern.findall(part))
    return counter


def _process_rank_chunk(byte_word_counter_chunk: list[tuple[tuple[int, ...], int]]) -> dict[tuple[int, int], int]:
    # Local rank computation over a chunk of (byte_tuple, count) items
    rank: dict[tuple[int, int], int] = {}
    for byte_tuple, count in byte_word_counter_chunk:
        for ind in range(len(byte_tuple) - 1):
            pair = (byte_tuple[ind], byte_tuple[ind + 1])
            rank[pair] = rank.get(pair, 0) + count
    return rank

def _merge_pair(byte_tuple: Sequence[int], pair: tuple[int, int], new_token: int) -> tuple[int, ...]:
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

def _update_counter_chunk(byte_word_counter_chunk: list[tuple[tuple[int, ...], int]],
                          pair: tuple[int, int],
                          new_token: int,) -> dict[tuple[int, int], int]:
    # Apply a single merge to a chunk of (byte_tuple, count) items
    updated_counter: dict[tuple[int, ...], int] = {}
    for byte_tuple, count in byte_word_counter_chunk:
        new_tuple = _merge_pair(byte_tuple, pair, new_token)
        updated_counter[new_tuple] = updated_counter.get(new_tuple, 0) + count
    return updated_counter

def _update_counter_chunk_args(byte_word_counter_chunk,pair,new_token) -> dict[tuple[int, int], int]:
    # Wrapper to avoid functools.partial pickling quirks on some platforms
    return _update_counter_chunk(byte_word_counter_chunk, pair, new_token)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizerParallel:
    def __init__(
        self,
        pattern: str | None = None,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        self.pattern: str = PAT if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

        # vocab maps token id -> token bytes (set during training)
        self.vocab: dict[int, bytes] | None = vocab
        self.inverse_vocab: dict[bytes, int] | None
        self.update_inverse_vocab()

        self.merges: list[tuple[int, int]] = []
        self.merge_ranks: dict[tuple[int, int], int] = {}
        if merges is not None:
            self._load_merges(merges)

        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}
        self.special_pattern = None
        self._special_token_set = None
        self.register_special_tokens(special_tokens=special_tokens)

    def update_inverse_vocab(self) -> None:
        if self.vocab is None:
            self.inverse_vocab = None
        else:
            self.inverse_vocab = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

    def register_special_tokens(self, special_tokens: list[str] | None) -> None:
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        if not special_tokens:
            return

        if self.vocab is None:
            self.vocab = {}
        if self.inverse_vocab is None:
            self.update_inverse_vocab()

        next_id = max(self.vocab.keys(), default=-1) + 1
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            token_id = self.inverse_vocab.get(token_bytes)
            if token_id is None:
                token_id = next_id
                next_id += 1
                self.vocab[token_id] = token_bytes
            self.special_tokens[token] = token_id

        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.update_inverse_vocab()

    def _load_merges(self, merges: list[tuple[bytes, bytes]]) -> None:
        if self.inverse_vocab is None:
            self.update_inverse_vocab()
        if self.inverse_vocab is None:
            raise ValueError("Cannot load merges without an initialized vocabulary.")
        prepared: list[tuple[int, int]] = []
        for left_bytes, right_bytes in merges:
            left = self.inverse_vocab[left_bytes]
            right = self.inverse_vocab[right_bytes]
            prepared.append((left, right))
        self._set_trained_merges(prepared)

    def _set_trained_merges(self, merges: list[tuple[int, int]]) -> None:
        self.merges = merges
        self.merge_ranks = {pair: idx for idx, pair in enumerate(merges)}

    def get_rank_parallel(
            self,
            byte_word_counter: dict[tuple[int, ...], int],
            pool: Pool,
    ) -> dict[tuple[int, int], int]:
        # Small input fallback
        if len(byte_word_counter) < 5000:
            return self.get_rank(byte_word_counter=byte_word_counter)

        all_items = list(byte_word_counter.items())
        procs = getattr(pool, "_processes", 1) or 1
        chunk_size = (len(all_items) + procs - 1) // procs
        chunks = [all_items[i:i + chunk_size] for i in range(0, len(all_items), chunk_size)]

        total_counter = Counter()
        for local_counter in pool.imap_unordered(_process_rank_chunk, chunks):
            total_counter.update(local_counter)
        return dict(total_counter)
    
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
    
    def update_counter_parallel(
        self,
        byte_word_counter: dict[tuple[int, ...], int],
        pair: tuple[int, int],
        new_token: int,
        pool: Pool,
    ) -> dict[tuple[int, ...], int]:
        # Derive worker count from the provided pool
        procs = getattr(pool, "_processes", 1) or 1
        if not byte_word_counter or procs <= 1:
            return self.update_counter(byte_word_counter, pair, new_token)

        n = len(byte_word_counter)
        threshold = max(50_000, 10_000 * procs)
        if n < threshold:
            return self.update_counter(byte_word_counter, pair, new_token)

        all_items = list(byte_word_counter.items())
        chunk_size = (n + procs - 1) // procs
        chunks = [all_items[i:i + chunk_size] for i in range(0, n, chunk_size)]
        if not chunks:
            return {}

        total_counter: Counter = Counter()
        arg_iter = ((chunk, pair, new_token) for chunk in chunks)
        for local_counter in pool.imap_unordered(_update_counter_chunk_args, arg_iter, chunksize=1):
            total_counter.update(local_counter)
        return dict(total_counter)

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
    

    def get_byte_word_counter_by_chunk(self, 
                                       text_path: str,
                                       split_token: bytes | None,
                                       num_chunks: int,
                                       pool: Pool | None = None) -> dict[tuple[int, ...], int]:
        # Small-input fast path: avoid Pool overhead
        if pool is None:
            # Fall back to serial path (still respects special token filtering)
            return self.get_byte_word_counter(text_path)
        
        path = Path(text_path)
        with open(path, "rb") as f:
            if split_token is None:
                f.seek(0, 2)
                boundaries = [0, f.tell()]
            else:
                boundaries = find_chunk_boundaries(f, num_chunks, split_token)

        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start < end:
                tasks.append((text_path, start, end))

        if not tasks:
            return {}

        # Small-input fast path: avoid Pool overhead when only one chunk
        if len(tasks) <= 1:
            return self.get_byte_word_counter(text_path)

        total_counter = Counter() 
        counter_iterator = pool.imap_unordered(_process_chunk, tasks)
        for counter in counter_iterator:
            total_counter.update(counter)

        return {tuple(token.encode("utf-8")): freq for token, freq in total_counter.items()}

    def get_byte_word_counter(self, text_path: str) -> dict[tuple[int, ...], int]:
        word_counter: Counter[str] = Counter()
        special_pattern = self.special_pattern
        special_set = self._special_token_set
        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read()
            if special_pattern is None:
                word_counter.update(self.compiled_pattern.findall(text))
            else:
                for part in special_pattern.split(text):
                    if not part or (special_set is not None and part in special_set):
                        continue
                    word_counter.update(self.compiled_pattern.findall(part))

        byte_word_counter: dict[tuple[int, ...], int] = {
            tuple(key.encode("utf-8")): value for key, value in word_counter.items()
        }
        return byte_word_counter
    
    def set_special_pattern(self, special_tokens: list[str]) -> None:
        if special_tokens:
            tokens = sorted(special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(tok) for tok in tokens) + ")"
            self.special_pattern = re.compile(pattern)
            self._special_token_set = set(tokens)
        else:
            self.special_pattern = None
            self._special_token_set = None

    def train(
        self,
        text_path: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        num_workers:int = 8
    ) -> None:
        special_tokens = special_tokens or []
        self.set_special_pattern(special_tokens=special_tokens)

        target_vocab = max(vocab_size, 256 + len(special_tokens))
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        # Initialize pool lazily (here we create once to reuse across phases)

        # Prefer 'fork' to reduce startup and tear-down overhead when available (Linux)
        try:
            ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else mp.get_context()
        except Exception:
            ctx = mp
        pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(self.pattern, list(self._special_token_set) if self._special_token_set else [])
        )
        try:
            byte_word_counter: dict[tuple[int, ...], int] = self.get_byte_word_counter_by_chunk(
                text_path=text_path,
                split_token=b"<|endoftext|>",
                num_chunks=num_workers,
                pool=pool,
            )
            counter: dict[tuple[int, ...], int] = byte_word_counter

            merges: list[tuple[int, int]] = []
            for idx in range(target_vocab - 256 - len(special_tokens)):
                # Use pooled parallel rank with small-input fallback inside
                rank = self.get_rank_parallel(counter, pool=pool)
                if not rank:
                    break
                # Tiebreak on equal counts using bytes lexicographic order of (left,right).
                # Compare by the actual byte expansions, not token ids.
                pair = max(
                    rank.items(),
                    key=lambda item: (item[1], (vocab[item[0][0]], vocab[item[0][1]])),
                )[0]
                merges.append(pair)
                a, b = pair
                new_token = 256 + idx
                vocab[new_token] = vocab[a] + vocab[b]

                # Parallel update with pool; function falls back to serial when too small
                counter = self.update_counter_parallel(counter, pair, new_token, pool=pool)
        finally:
            # Graceful shutdown to avoid expensive terminate() path in context manager __exit__
            try:
                pool.close()
            finally:
                pool.join()

        # Finalize tokenizer state
        self.vocab = vocab
        self.update_inverse_vocab()
        self._set_trained_merges(merges)
        if special_tokens:
            self.register_special_tokens(special_tokens)

    def get_score(self, pair: tuple[int, int]) -> int:
        return self.merge_ranks.get(pair, math.inf)

    def get_new_token(self, score: int) -> int:
        return 256 + score

    def find_best_pair(self, seq: Sequence[int]) -> tuple[tuple[int, int] | None, int | None]:
        best_score = math.inf
        best_pair = None
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            score = self.get_score(pair=pair)
            if score < best_score:
                best_score = score
                best_pair = pair

        merged_token = (
            self.get_new_token(score=best_score) if not math.isinf(best_score) else None
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
        part_bytes: list[bytes] = []
        for token_id in tokens:
            if token_id in self.vocab:
                part_bytes.append(self.vocab[token_id])
            elif token_id in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[token_id].encode("utf-8"))
            else:
                raise ValueError(f"invalid token: {token_id}")
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        special = self.special_tokens
        if not special:
            return self.encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids: list[int] = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
