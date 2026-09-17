import heapq
import multiprocessing
import os
import regex as re
from typing import BinaryIO, List
from collections import Counter, defaultdict

from .pretokenization_example import find_chunk_boundaries

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class _MaxPair:
    """
    Wraps a pair so a min-heap pops the lexicographically greatest pair among
    equal-frequency candidates, matching max(pair_freq, key=(pair_freq[k], k)).
    """
    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "_MaxPair") -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MaxPair) and self.pair == other.pair


def _process_chunk(
    input_path: str | os.PathLike,
    start: int, end: int, special_tokens: list[str]
) -> Counter:
    """
    Process a chunk of the file from start to end bytes.
    String of word to frequency.
    """
    # Boundaries come from a binary scan, so read bytes and decode here rather
    # than seeking to a byte offset on a text-mode stream.
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # 2(b). Remove special tokens by splitting the chunk.
        special_pattern = "|".join(map(re.escape, special_tokens))
        if special_pattern:
            segments = re.split(special_pattern, chunk)
        else:
            segments = [chunk]

        all_pre_tokens = []
        for segment in segments:
            pre_tokens = re.findall(GPT2_PRETOKENIZER_PATTERN, segment)
            all_pre_tokens.extend(pre_tokens)

        # 3&4. Construct original adjacent count map, no overlap across list items.
        pre_token_freq = Counter(all_pre_tokens)
        # Example: <apple: 5 times>

        return pre_token_freq


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE tokenizer using multiprocessing for pre-tokenization, then an
    index-accelerated merge loop:
      - pair_to_words maps a pair to the word ids that may contain it, so a
        merge only rescans affected words instead of every unique word.
      - a max-heap over pair frequencies (lazy invalidation) replaces the
        linear max() scan over all pairs on every merge.
      - each word is a doubly linked list of symbol nodes, so applying a merge
        splices two nodes together in O(1) instead of rebuilding a tuple.
    """
    # Use the first special token as the split token if available, otherwise default to space
    # This is used to find safe boundaries to split the file
    split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b" "

    with open(input_path, "rb") as f:
        num_chunks = multiprocessing.cpu_count()
        boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)

    # Prepare arguments for each chunk
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((input_path, start, end, special_tokens))

    # 1. Initialize vocab with with utf-8 256 characters and special tokens.
    vocab = {i: bytes([i]) for i in range(256)}

    # 2(a). Add special tokens to vocab
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    # Process chunks in parallel. Use "fork" explicitly: the default "spawn"
    # start method on macOS re-imports the entire parent process (including
    # the test runner and its plugins) in every worker, which dwarfs the
    # actual pre-tokenization work on a small corpus like this one.
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=len(chunk_args)) as pool:
        pre_token_freq_string_list = pool.starmap(_process_chunk, chunk_args)
    pre_token_freq_string = sum(pre_token_freq_string_list, Counter())  # sum all counters

    # Get pre-token frequency map
    # Example: <(b'a', b'p', b'p'): 5 times> <= one word
    pre_token_freqs = Counter()
    get_tuple_of_bytes = lambda pretoken: tuple(bytes([b]) for b in pretoken.encode("utf-8"))
    for pre_token, freq in pre_token_freq_string.items():
        pre_token_freqs[get_tuple_of_bytes(pre_token)] += freq

    # Build one doubly linked list of symbol nodes per unique word, stored as
    # parallel arrays indexed by node id.
    sym_value: list[bytes] = []
    sym_prev: list[int | None] = []
    sym_next: list[int | None] = []

    word_head: list[int] = []
    word_freq: list[int] = []

    pair_freq: Counter = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for word_id, (pre_token_tuple, freq) in enumerate(pre_token_freqs.items()):
        word_freq.append(freq)
        prev_id = None
        first_id = None
        for b in pre_token_tuple:
            nid = len(sym_value)
            sym_value.append(b)
            sym_prev.append(prev_id)
            sym_next.append(None)
            if prev_id is not None:
                sym_next[prev_id] = nid
            else:
                first_id = nid
            prev_id = nid
        word_head.append(first_id)

        nid = first_id
        while nid is not None and sym_next[nid] is not None:
            pair = (sym_value[nid], sym_value[sym_next[nid]])
            pair_freq[pair] += freq
            pair_to_words[pair].add(word_id)
            nid = sym_next[nid]

    heap = [(-freq, _MaxPair(pair)) for pair, freq in pair_freq.items()]
    heapq.heapify(heap)

    def push(pair: tuple[bytes, bytes]) -> None:
        heapq.heappush(heap, (-pair_freq[pair], _MaxPair(pair)))

    # 5. Keep iterate through map for merging.
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - 256 - len(special_tokens)

    for _ in range(num_merges):
        # Pop the best pair, skipping entries whose frequency changed since
        # they were pushed.
        while True:
            neg_freq, wrapped = heapq.heappop(heap)
            if pair_freq.get(wrapped.pair, 0) == -neg_freq:
                most_freq_pair = wrapped.pair
                break

        new_token_id = len(vocab)
        new_token = b"".join(most_freq_pair)
        vocab[new_token_id] = new_token
        merges.append(most_freq_pair)

        for word_id in pair_to_words.pop(most_freq_pair, ()):
            freq = word_freq[word_id]
            nid = word_head[word_id]
            while nid is not None and sym_next[nid] is not None:
                nxt = sym_next[nid]
                if sym_value[nid] != most_freq_pair[0] or sym_value[nxt] != most_freq_pair[1]:
                    nid = nxt
                    continue

                left_nb = sym_prev[nid]
                right_nb = sym_next[nxt]

                if left_nb is not None:
                    old_pair = (sym_value[left_nb], sym_value[nid])
                    pair_freq[old_pair] -= freq
                    if pair_freq[old_pair] <= 0:
                        del pair_freq[old_pair]
                    else:
                        push(old_pair)
                    new_pair = (sym_value[left_nb], new_token)
                    pair_freq[new_pair] += freq
                    pair_to_words[new_pair].add(word_id)
                    push(new_pair)

                if right_nb is not None:
                    old_pair = (sym_value[nxt], sym_value[right_nb])
                    pair_freq[old_pair] -= freq
                    if pair_freq[old_pair] <= 0:
                        del pair_freq[old_pair]
                    else:
                        push(old_pair)
                    new_pair = (new_token, sym_value[right_nb])
                    pair_freq[new_pair] += freq
                    pair_to_words[new_pair].add(word_id)
                    push(new_pair)

                pair_freq[most_freq_pair] -= freq
                if pair_freq[most_freq_pair] <= 0:
                    del pair_freq[most_freq_pair]

                # Splice nid and nxt into one merged node, in place.
                sym_value[nid] = new_token
                sym_next[nid] = right_nb
                if right_nb is not None:
                    sym_prev[right_nb] = nid
                # Stay at nid so overlapping repeats merge left-to-right in
                # this same pass.

    return (vocab, merges)
