import io
from pathlib import Path

import pytest

from cs336_basics.pretokenizer import PreTokenizer, PAT, to_byte_word


def test_iter_chunks_without_specials():
    pt = PreTokenizer(pattern=PAT, special_tokens=None)
    text = "Hello, world!"
    toks = list(pt.iter_chunks(text))
    # Basic sanity: should produce non-empty tokens and join equals original after stripping spaces
    assert toks, "iter_chunks should yield tokens"
    # Ensure punctuation and words are present in some form
    assert any("Hello" in t for t in toks)
    assert any("world" in t for t in toks)


def test_iter_chunks_with_special_boundary():
    pt = PreTokenizer(pattern=PAT, special_tokens=["<|endoftext|>"])
    text = "A<|endoftext|>B"
    toks = list(pt.iter_chunks(text))
    # Special token should not be yielded; only 'A' and 'B' chunks expected
    assert all("<|endoftext|>" not in t for t in toks)
    assert any("A" in t for t in toks)
    assert any("B" in t for t in toks)


def test_read_chunks_alignment(tmp_path: Path):
    pt = PreTokenizer(pattern=PAT, special_tokens=["<|endoftext|>"])
    # Place two split tokens; ensure one occurs after the mid-point so the
    # boundary search (which starts at ~file_size//2) can find it.
    content = b"alpha<|endoftext|>" + b"X" * 100 + b"<|endoftext|>omega"
    p = tmp_path / "sample.txt"
    p.write_bytes(content)
    chunks = list(pt.read_chunks(p, num_chunks=2, split_token=b"<|endoftext|>"))
    # Should split into two non-empty chunks whose concatenation equals original decoded content
    assert len(chunks) == 2
    assert all(isinstance(c, str) and c for c in chunks)
    assert (chunks[0] + chunks[1]).encode("utf-8") == content


def test_count_words_and_byte_counter(tmp_path: Path):
    pt = PreTokenizer(pattern=PAT, special_tokens=["<|endoftext|>"])
    # Two occurrences of 'abc' separated by special token
    text = "abc<|endoftext|>abc"
    p = tmp_path / "two_abc.txt"
    p.write_text(text, encoding="utf-8")

    wc = pt.count_words(p, num_chunks=2, split_token=b"<|endoftext|>")
    assert wc["abc"] == 2

    bwc = pt.build_byte_word_counter(p, num_chunks=2, split_token=b"<|endoftext|>")
    assert bwc[to_byte_word("abc")] == 2
