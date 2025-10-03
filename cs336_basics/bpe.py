import os
import multiprocessing
from typing import BinaryIO
from collections import Counter, defaultdict
import regex as re

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    assert isinstance(split_special_tokens, list) and all(isinstance(t, bytes) for t in split_special_tokens), "Must represent special tokens as a list of bytestrings"

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
            # found_at = mini_chunk.find(split_special_tokens)
            found_at = -1
            for token in split_special_tokens:
                fd_at = mini_chunk.find(token)
                if fd_at != -1 and (found_at == -1 or found_at > fd_at):
                    found_at = fd_at

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _process_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str]) -> defaultdict[tuple[bytes, ...], int]:

    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    
    text = chunk_bytes.decode('utf-8')
    # split as special tokens
    special_pat = "|".join(re.escape(st) for st in special_tokens)
    sub_chunks = re.split(f"{special_pat}", text)

    pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    compiled_pat = re.compile(pat_str)

    pre_tk_cnt = defaultdict(int)
    for sub_chunk in sub_chunks:
        if not sub_chunk:
            continue
        for match in compiled_pat.finditer(sub_chunk):
            tk_bytes = match.group(0).encode("utf-8")
            tp = tuple(bytes([ch]) for ch in tk_bytes)
            pre_tk_cnt[tp] += 1
    
    return pre_tk_cnt

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # init vocab
    vocab: dict[int, bytes] = {x : bytes([x]) for x in range(256)}
    for i, sp_tk in enumerate(special_tokens):
        vocab[256 + i] = sp_tk.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    # parallelize pre-tokenization
    with open(input_path, 'rb') as f:
        num_processes = min(os.cpu_count() or 1, 8)
        boundaries = find_chunk_boundaries(f, num_processes, [t.encode('utf-8') for t in special_tokens])

    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with multiprocessing.Pool(len(boundaries)) as pool:
        list_of_pre_tk_cnts = pool.starmap(_process_chunk, chunk_args)

    agtd_pre_tk_cnts = defaultdict(int)
    for pre_tk_cnts in list_of_pre_tk_cnts:
        for tk, cnt in pre_tk_cnts.items():
            agtd_pre_tk_cnts[tk] += cnt

    # BPE merge
    merge_times = vocab_size - len(vocab)
    
    # used to index the pre-tokens of a bytes to efficiently merge
    pre_tk_of_bytes = defaultdict(set[tuple[bytes, ...]])
    freq_dict = defaultdict(int)

    for i in range(merge_times):
        pre_tk_of_bytes.clear()
        freq_dict.clear()

        for tk_bytes_tp, cnt in agtd_pre_tk_cnts.items():
            for bt0, bt1 in zip(tk_bytes_tp[:-1], tk_bytes_tp[1:]):
                mg_tp = (bt0, bt1)
                freq_dict[mg_tp] += cnt
                pre_tk_of_bytes[mg_tp].add(tk_bytes_tp)

        most_freq_bp = max(freq_dict, key = lambda k : (freq_dict[k], k))
        bytes0, bytes1 = most_freq_bp
        common_idxes = pre_tk_of_bytes[most_freq_bp]

        for tk_bytes_tp in common_idxes:
            new_tk_bytes : list[bytes] = []

            j = 0
            while j < len(tk_bytes_tp):
                bt = tk_bytes_tp[j]
                if j + 1 < len(tk_bytes_tp) and bt == bytes0 and tk_bytes_tp[j+1] == bytes1:
                    new_tk_bytes.append(bt + tk_bytes_tp[j+1])
                    j += 2   
                else:
                    new_tk_bytes.append(bt)
                    j += 1
            
            # update pre-token bytes tuple - cnt dict
            new_tk_bytes_tp = tuple(new_tk_bytes)
            cnt = agtd_pre_tk_cnts.pop(tk_bytes_tp)
            agtd_pre_tk_cnts[new_tk_bytes_tp] = cnt

        # merge
        vocab[len(vocab)] = bytes0 + bytes1
        merges.append(most_freq_bp)

    return vocab, merges
