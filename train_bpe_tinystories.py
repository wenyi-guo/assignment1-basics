"""
Train a byte-level BPE tokenizer on the full TinyStories dataset.

This is the `train_bpe_tinystories` deliverable from the assignment handout:
train BPE with vocab_size=10000 and a single special token "<|endoftext|>"
on data/TinyStoriesV2-GPT4-train.txt, then report:
  - total training time
  - peak resident memory usage
  - the longest token in the resulting vocabulary (and whether it makes sense)

Run with:
    uv run train_bpe_tinystories.py
"""

import cProfile
import json
import pickle
import pstats
import time
from pathlib import Path

import psutil

from cs336_basics.train_bpe import train_bpe

DATA_PATH = Path(__file__).parent / "data" / "TinyStoriesV2-GPT4-train.txt"
OUTPUT_DIR = Path(__file__).parent / "data" / "tinystories_bpe"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]
PROFILE = False  # set True to also dump a cProfile breakdown


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected training data at {DATA_PATH}. See README.md for the download command."
        )

    process = psutil.Process()
    start_rss = process.memory_info().rss

    start_time = time.time()
    profiler = cProfile.Profile() if PROFILE else None
    if profiler:
        profiler.enable()

    vocab, merges = train_bpe(
        input_path=DATA_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    if profiler:
        profiler.disable()
    end_time = time.time()

    peak_rss = process.memory_info().rss
    elapsed = end_time - start_time

    longest_token_id, longest_token = max(vocab.items(), key=lambda kv: len(kv[1]))

    print("=" * 60)
    print("train_bpe_tinystories results")
    print("=" * 60)
    print(f"Input file:        {DATA_PATH}")
    print(f"Vocab size:        {len(vocab)} (target {VOCAB_SIZE})")
    print(f"Num merges:        {len(merges)}")
    print(f"Training time:     {elapsed:.2f} s ({elapsed / 60:.2f} min)")
    print(f"RSS before:        {start_rss / (1024 ** 3):.2f} GiB")
    print(f"RSS after (peak-ish): {peak_rss / (1024 ** 3):.2f} GiB")
    print(f"Longest token id:  {longest_token_id}")
    print(f"Longest token:     {longest_token!r} (length {len(longest_token)} bytes)")
    try:
        print(f"Longest token (decoded): {longest_token.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print("Longest token (decoded): <not valid utf-8>")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(OUTPUT_DIR / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    # Also dump a human-readable vocab (bytes shown via latin-1 escaping) for quick inspection.
    with open(OUTPUT_DIR / "vocab_readable.json", "w") as f:
        json.dump(
            {str(k): v.decode("latin-1") for k, v in vocab.items()},
            f,
            indent=2,
        )
    print(f"\nSaved vocab/merges to {OUTPUT_DIR}/")

    if profiler:
        stats_path = OUTPUT_DIR / "profile.stats"
        profiler.dump_stats(str(stats_path))
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        print(f"\nSaved cProfile stats to {stats_path}; top 20 by cumulative time:")
        stats.print_stats(20)


if __name__ == "__main__":
    main()
