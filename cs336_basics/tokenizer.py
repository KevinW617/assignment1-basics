from typing import Iterable, Iterator
import json
import regex

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

        # 为了在pre-tokenize时优先把special token分出来
        pat_str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        self.pat = regex.compile(pat_str)
        self.inv_vocab = {v : k for k, v in self.vocab.items()}

        # cache for speed up
        self.encode_cache = {}

        # merges rank
        self.merges_ranks = {mg: i for i, mg in enumerate(self.merges)}

    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            str_vocab = json.load(f)
        vocab = {int(idx):(token.encode('utf-8')) for idx, token in str_vocab.items()}

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    p1, p2 = cleaned_line.split(" ")
                    merges.append((p1.encode('utf-8'), p2.encode('utf-8')))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:

        int_seq = []

        if self.special_tokens != None:

            # split by special tokens first
            special_pat = "|".join(regex.escape(st) for st in self.special_tokens)
            # 创建捕获组，保留special token分隔符
            chunks = regex.split(f"({special_pat})", text)
        else:
            chunks = [text]


        # pre-tokenize
        for chunk in chunks:
            # special token
            if self.special_tokens and chunk in self.special_tokens:
                int_seq.append(self.inv_vocab[chunk.encode("utf-8")])
                continue

            for match in self.pat.finditer(chunk):
                # print("match:", match.group())
                word = match.group()
                if word in self.encode_cache:
                    int_seq.extend(self.encode_cache[word])
                    continue

                tk_bytes = word.encode("utf-8")

                # merging
                bytes_tp = tuple(bytes([ch]) for ch in tk_bytes)
                
                # 每个合并规则刷一遍word，大概M * N的复杂度
                # for mg in self.merges:
                #     bts0, bts1 = mg
                #     new_bytes_ls = []
                #     i = 0
                #     while i < len(bytes_tp):
                #         if i + 1 < len(bytes_tp) and bytes_tp[i] == bts0 and bytes_tp[i + 1] == bts1:
                #             new_bytes_ls.append(bts0 + bts1)
                #             i += 2
                #         else:
                #             new_bytes_ls.append(bytes_tp[i])
                #             i += 1
                #     bytes_tp = tuple(new_bytes_ls)

                #     if len(bytes_tp) == 1:
                #         break

                # 刷word找到rank最高的pair进行merge，直到没有可以merge的pair为止
                # 字典查找是O(1)，大概N^2的复杂度
                while True:
                    pairs = [(bytes_tp[i], bytes_tp[i + 1]) for i in range(len(bytes_tp) - 1)]
                    # 找出所有可以merge的pair
                    candidate_pairs = [pair for pair in pairs if pair in self.merges_ranks]
                    if not candidate_pairs:
                        break
                    # 选择rank最高的pair进行merge
                    best_pair = min(candidate_pairs, key=lambda pair: self.merges_ranks[pair])
                    bts0, bts1 = best_pair
                    new_bytes_ls = []
                    i = 0
                    while i < len(bytes_tp):
                        if i + 1 < len(bytes_tp) and bytes_tp[i] == bts0 and bytes_tp[i + 1] == bts1:
                            new_bytes_ls.append(bts0 + bts1)
                            i += 2
                        else:
                            new_bytes_ls.append(bytes_tp[i])
                            i += 1
                    bytes_tp = tuple(new_bytes_ls)
                    if len(bytes_tp) == 1:
                        break
                
                ids_ls = []
                for bts in bytes_tp:
                    id = self.inv_vocab[bts]
                    ids_ls.append(id)
                    int_seq.append(id)
                self.encode_cache[word] = ids_ls


        return int_seq
        
    # iterable[str] like a list of string or a file handle of lines of string
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            # 省去 for id in self.encode(line): yield id
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab.get(token_id, b'') for token_id in ids]
        full_bytes = b"".join(bytes_list)
        return full_bytes.decode("utf-8", errors="replace")
