import os
import multiprocessing
from typing import BinaryIO, TypeAlias, Optional
from collections import Counter, defaultdict
from typing import DefaultDict
import regex as re

BpPos : TypeAlias = tuple[tuple[bytes, ...], int, int]
Bp : TypeAlias = tuple[bytes, bytes]

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
        num_processes = min(os.cpu_count() or 1, 64)
        boundaries = find_chunk_boundaries(f, num_processes, [t.encode('utf-8') for t in special_tokens])

    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with multiprocessing.Pool(len(boundaries)) as pool:
        list_of_pre_tk_cnts = pool.starmap(_process_chunk, chunk_args)

    # bytes tuple(like split a word into bytes) : cnt
    agtd_pre_tk_cnts = defaultdict(int)
    for pre_tk_cnts in list_of_pre_tk_cnts:
        for tk, cnt in pre_tk_cnts.items():
            agtd_pre_tk_cnts[tk] += cnt

    # BPE merge
    merge_times = vocab_size - len(vocab)
    
    # freq_dict: bytes pair tuple : cnt
    freq_dict = defaultdict(int)
    # BpPos: (bytes tuple of a word, start, len)
    # pos_map: bytes pair : list[BpPos]
    pos_map: DefaultDict[
                        tuple[bytes, bytes], 
                        list[BpPos]
    ] = defaultdict(list)

    # prev_pos_map: pos : pos
    prev_pos_map: dict[
                        BpPos,
                        BpPos
    ] = {}

    # prev_bp_map: pos : bp
    prev_bp_map: dict[
                        BpPos,
                        Bp
    ] = {}

    # back_pos_map: pos : pos
    back_pos_map: dict[
                        BpPos,
                        BpPos
    ] = {}

    # back_bp_map: pos : pos
    back_bp_map: dict[
                        BpPos,
                        Bp
    ] = {}

    for wd_bytes_tp, cnt in agtd_pre_tk_cnts.items():
        for idx, (bt0, bt1) in enumerate(zip(wd_bytes_tp[:-1], wd_bytes_tp[1:])):
            mg_tp = (bt0, bt1)
            freq_dict[mg_tp] += cnt

            cur_pos : BpPos = (wd_bytes_tp, idx, 2)
            pos_map[mg_tp].append(cur_pos)

            if idx > 0:
                prev_pos_map[cur_pos] = (wd_bytes_tp, idx - 1, 2)
                prev_bp_map[cur_pos] = (wd_bytes_tp[idx - 1], bt0)
            if idx + 2 < len(wd_bytes_tp):
                back_pos_map[cur_pos] = (wd_bytes_tp, idx + 1, 2)
                back_bp_map[cur_pos] = (bt1, wd_bytes_tp[idx + 2])

    for i in range(merge_times):
        most_freq_bp = max(freq_dict, key = lambda k : (freq_dict[k], k))
        bytes0, bytes1 = most_freq_bp

        merged_bp = bytes0 + bytes1

        most_freq = freq_dict.pop(most_freq_bp)
        
        # 使用一个集合来跟踪本轮已经处理过的位置，防止重复处理
        # eg.  (b't', b'h', b'e', b'e', b'e')
        processed_positions_this_turn = set()

        positions_to_process = list(pos_map[most_freq_bp])
        for pos in positions_to_process:
            # 如果这个位置在本轮合并中已经被作为其他合并的一部分处理过，就跳过
            if pos in processed_positions_this_turn:
                continue

            times = agtd_pre_tk_cnts[pos[0]]
            # 处理前驱
            if pos in prev_pos_map:
                prev_pos: BpPos = prev_pos_map[pos]
                prev_bp:  Bp    = prev_bp_map[pos]
                
                # 创建新的前驱对和位置
                new_prev_bp = (prev_bp[0], merged_bp)
                new_prev_pos_tpl = (prev_pos[0], prev_pos[1], len(prev_bp[0]) + len(merged_bp))
                new_prev_pos = BpPos(new_prev_pos_tpl)

                # 更新频率
                freq_dict[prev_bp] -= times
                if freq_dict[prev_bp] == 0:
                    freq_dict.pop(prev_bp)
                freq_dict[new_prev_bp] += times
                
                # 更新 pos_map
                pos_map[prev_bp].remove(prev_pos)
                if not pos_map[prev_bp]:
                    pos_map.pop(prev_bp)
                pos_map[new_prev_bp].append(new_prev_pos)

                # 更新双向链表
                if prev_pos in prev_pos_map:
                    # 更新前驱的前驱的后继
                    prev_prev_pos = prev_pos_map[prev_pos]
                    prev_prev_bp = prev_bp_map[prev_pos]
                    back_pos_map[prev_prev_pos] = new_prev_pos
                    back_bp_map[prev_prev_pos] = new_prev_bp

                    prev_pos_map.pop(prev_pos)
                    prev_pos_map[new_prev_pos] = prev_prev_pos

                    prev_bp_map.pop(prev_pos)
                    prev_bp_map[new_prev_pos] = prev_prev_bp

                # 将旧位置标记为已处理
                processed_positions_this_turn.add(prev_pos)

            # 后继
            if pos in back_pos_map:
                back_pos: BpPos = back_pos_map[pos]
                back_bp:  Bp   = back_bp_map[pos]

                # 创建新的后继对和位置
                new_back_bp = (merged_bp, back_bp[1])
                new_back_pos_tpl = (back_pos[0], pos[1], len(merged_bp) + len(back_bp[1]))
                new_back_pos = BpPos(new_back_pos_tpl)

                # 更新频率
                freq_dict[back_bp] -= times
                if freq_dict[back_bp] == 0:
                    freq_dict.pop(back_bp)
                freq_dict[new_back_bp] += times

                # 更新 pos_map
                pos_map[back_bp].remove(back_pos)
                if not pos_map[back_bp]:
                    pos_map.pop(back_bp)
                pos_map[new_back_bp].append(new_back_pos)

                # 更新双向链表
                if back_pos in back_pos_map:
                    # 更新后继的后继的前驱
                    back_back_pos = back_pos_map[back_pos]
                    back_back_bp = back_bp_map[back_pos]
                    prev_pos_map[back_back_pos] = new_back_pos
                    prev_bp_map[back_back_pos] = new_back_bp

                    back_pos_map.pop(back_pos)
                    back_pos_map[new_back_pos] = back_back_pos

                    back_bp_map.pop(back_pos)
                    back_bp_map[new_back_pos] = back_back_bp

                # 将旧位置标记为已处理
                processed_positions_this_turn.add(back_pos)

            # 更新新节点之间的链接
            if pos in prev_pos_map and pos in back_pos_map:
                prev_pos_map.pop(pos)
                prev_bp_map.pop(pos)
                back_pos_map.pop(pos)
                back_bp_map.pop(pos)
                back_pos_map[new_prev_pos] = new_back_pos
                prev_pos_map[new_back_pos] = new_prev_pos
                back_bp_map[new_prev_pos] = new_back_bp
                prev_bp_map[new_back_pos] = new_prev_bp

        # 清理被合并的 bp
        pos_map.pop(most_freq_bp, None)

        vocab[len(vocab)] = bytes0 + bytes1
        merges.append(most_freq_bp)

    return vocab, merges
