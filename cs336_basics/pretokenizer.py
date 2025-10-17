"""
PreTokenizer utilities: split text into regex chunks ("words"), optionally
respecting special-token boundaries, and build byte-level counters for BPE
training. File-chunk helpers are provided to align large-file reads to a
chosen split token (e.g., "<|endoftext|>").

Notes
- If a ``split_token`` is provided to file-chunk helpers but does not occur in
  the file, the entire file is yielded as a single chunk (expected behavior).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO, Optional
import os
import regex as re


# Default regex pattern for GPT2-like pre-tokenization.
PAT: str = (
    "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
)

# Type aliases used by callers (Trainer, etc.).
Word = tuple[int, ...]


class PreTokenizer:
    """Regex-based pre-tokenizer with optional special-token boundaries.

    Responsibilities:
    - Split raw text into regex chunks ("words").
    - Optionally avoid crossing a special-token boundary while counting.
    - Provide helpers to read large files in aligned chunks for parallelizable counting.
    """

    def __init__(
        self,
        pattern: Optional[str] = None,
        special_tokens: Optional[list[str]] = None,
    ) -> None:
        """Initialize a pre-tokenizer.

        pattern: Unicode regex string used to split ordinary text into chunks.
        special_tokens: If provided, occurrences are treated as atomic separators
                        (do not cross them when counting ordinary chunks).
        """
        self.pattern: str = PAT if pattern is None else pattern
        self._re: re.Pattern[str] = re.compile(self.pattern)

        self.special_tokens: list[str] = []
        self._special_re: Optional[re.Pattern[str]] = None
        self._special_set: Optional[set[str]] = None
        self.set_special_tokens(special_tokens or [])

    def set_special_tokens(self, special_tokens: list[str]) -> None:
        """Register special tokens and prepare separator regex.

        The split regex should be built in a way that prevents crossing special
        tokens when iterating ordinary chunks.
        """
        if special_tokens:
            tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens = tokens
            pattern = "(" + "|".join(re.escape(tok) for tok in tokens) + ")"
            self._special_re = re.compile(pattern)
            self._special_set = set(tokens)
        else:
            self._special_re = None
            self._special_set = None


    # ---------- Large-file helpers (optional but useful) ----------
    @staticmethod
    def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """Return byte offsets that align chunks on the given split token.

        The returned list contains sorted, unique offsets with first=0 and
        last=file_size. Implementations may return fewer than desired chunks.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        # Defensive bounds; avoid division by zero and degenerate chunk sizes
        desired_num_chunks = max(1, int(desired_num_chunks))
        if file_size <= 0:
            return [0, 0]
        chunk_size = max(1, file_size // desired_num_chunks)

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
        self,
        path: Path | str,
        num_chunks: int,
        split_token: Optional[bytes],
    ) -> Iterator[str]:
        """Yield UTF-8 decoded text chunks from a file.

        If split_token is not None, chunk boundaries align to occurrences of
        that token; otherwise the whole file is yielded as a single chunk.
        """
        with open(path,"rb") as fh:
            if split_token is None:
                fh.seek(0,2)
                boundaries = [0,fh.tell()]
            else:
                boundaries = self.find_chunk_boundaries(fh,num_chunks,split_token)

            for start,end in zip(boundaries[:-1],boundaries[1:]):
                fh.seek(start)
                chunk_bytes = fh.read(end - start)
                if not chunk_bytes:
                    continue
                yield chunk_bytes.decode("utf-8",errors="ignore")

    # ---------- Counting APIs used by Trainer ----------
    def iter_chunks(self, text: str) -> Iterator[str]:
        """Yield regex chunks from a given text string (ordinary path).

        Implementations should use self._re and honor special-token splitting
        if self._special_re is not None.
        """
        if self._special_re is None:
            for m in self._re.finditer(text):
                tok = m.group(0)
                if tok:
                    yield tok
            return
        
        for part in self._special_re.split(text):
            if not part:
                continue
            if self._special_set and part in self._special_set:
                continue
            for m in self._re.finditer(part):
                tok = m.group(0)
                if tok:
                    yield tok


    def count_words(self, 
                    path: Path | str,
                    num_chunks:int = 1,
                    split_token:bytes|None = None) -> Counter[str]:
        """Return Counter over regex chunks (string form) from a file.

        Special tokens (if any) are excluded from ordinary chunk counting.
        If ``split_token`` is provided, the file is first split into
        ``num_chunks`` best-effort chunks aligned on that token; otherwise the
        whole file is processed as a single chunk.
        """
        word_counter = Counter()
        for chunk in self.read_chunks(path=path,num_chunks=num_chunks,split_token=split_token):
            word_counter.update(self.iter_chunks(chunk))
        return word_counter

    def build_byte_word_counter(self, 
                                path: Path | str,
                                num_chunks:int = 1,
                                split_token:bytes|None = b"<|endoftext|>") -> dict[Word, int]:
        """Return mapping from byte-level word tuples to their frequencies.

        This converts each chunk's UTF-8 bytes into a tuple[int,...] suitable
        for BPE training, aggregating identical byte tuples.
        """
        desired_chunks = max(1,num_chunks)
        word_counter = self.count_words(path=path,num_chunks = desired_chunks,split_token = split_token)
        byte_word_counter: dict[tuple[int, ...], int] = {
            to_byte_word(key): value for key, value in word_counter.items()
        }
        return byte_word_counter


# Optional utility: convert a string to a byte-level Word tuple.
def to_byte_word(s: str) -> Word:
    """Encode a string as UTF-8 and return a tuple of byte values (0..255)."""
    return tuple(s.encode("utf-8"))
