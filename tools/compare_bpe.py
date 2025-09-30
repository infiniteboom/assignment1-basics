import sys
import time
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'taylorswift.txt'
BASIC = ROOT / 'cs336_basics' / 'basic_bpe.py'
LISTB = ROOT / 'cs336_basics' / 'basic_bpe_list.py'

PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def main(vocab_size: int = 3000):
    basic = load_module('basic_bpe_mod', BASIC)
    listb = load_module('basic_bpe_list_mod', LISTB)

    t0 = time.time()
    bt = basic.BPETokenizer(PATTERN)
    bt.train(str(DATA), vocab_size)
    t_basic = time.time() - t0

    t0 = time.time()
    lt = listb.BPETokenizerList(PATTERN)
    lt.train(str(DATA), vocab_size)
    t_list = time.time() - t0

    # Sanity: encode the same sample and compare decode
    sample = "Hello world! 你好，世界！ This is a small test."
    ids0 = bt.encode(sample)
    ids1 = lt.encode(sample)
    same_ids = list(ids0) == list(ids1)
    dec0 = bt.decode(ids0)
    dec1 = lt.decode(ids1)
    same_decode = (dec0 == dec1 == sample)

    print("vocab_size=", vocab_size)
    print("basic_counter_time_sec=", round(t_basic, 4))
    print("list_based_time_sec=", round(t_list, 4))
    print("same_ids=", same_ids)
    print("same_decode=", same_decode)

if __name__ == '__main__':
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    main(vocab_size)
