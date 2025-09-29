import regex as re
from collections import Counter

from test import byte_word_counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:

    def __init__(self,pattern:str):
        self.PAT = pattern
        self.vocab = None

    def get_rank(self,byte_word_counter:dict) -> list:
        rank = {}
        for byte_tuple, count in byte_word_counter.items():
            for ind in range(len(byte_tuple) - 1):
                pair = (byte_tuple[ind], byte_tuple[ind + 1])
                rank[pair] = rank.get(pair, 0) + count
        return sorted(rank.items(), key=lambda x: x[1], reverse=True)

    def merge_pair(self,byte_tuple,pair,new_token) -> tuple:
        new_list = []
        i = 0
        a,b = pair
        while i < len(byte_tuple):
            if i < len(byte_tuple) - 1 and (byte_tuple[i] == a and byte_tuple[i+1] == b):
                new_list.append(new_token)
                i += 2
            else:
                new_list.append(byte_tuple[i])
                i += 1
        return tuple(new_list)

    def update_counter(self,byte_word_counter:dict,pair:tuple,new_token:int) -> dict:
        updated_counter = {}
        for byte_tuple,count in byte_word_counter.items():
            new_tuple = self.merge_pair(byte_tuple,pair,new_token)
            updated_counter[new_tuple] = count
        return updated_counter

    def get_byte_word_counter(self,text_path):
        word_counter = Counter()
        with open("data/taylorswift.txt",'r',encoding = 'utf-8') as file:
            line = file.readline()
            while line:
                word_counter.update(re.findall(self.PAT,line))
                line = file.readline()

        byte_word_counter = {
            tuple(key.encode('utf-8')):value for key,value in word_counter.items()
        }
        return byte_word_counter

    def train(self,text_path:str,vocab_size:int) -> dict:
        byte_word_counter = self.get_byte_word_counter(text_path=text_path)
        if vocab_size < 256:
            vocab_size = 256
        vocab = {idx:bytes([idx]) for idx in range(256)}
        counter = byte_word_counter
        for i in range(vocab_size - 256):
            rank = self.get_rank(counter)
            if not rank:
                break
            pair = rank[0][0]
            new_token = 256 + i
            vocab[new_token] = pair
            counter = self.update_counter(counter,pair,new_token)
        self.vocab = vocab

    