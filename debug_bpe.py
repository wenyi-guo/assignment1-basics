
import sys
from collections import Counter
import regex as re

from cs336_basics.train_bpe import train_bpe, _process_chunk

def test_debug():
    # Create a small file
    content = "lower newer"
    with open("debug_corpus.txt", "w") as f:
        f.write(content)
    
    # Run with small vocab
    vocab, merges = train_bpe("debug_corpus.txt", 256 + 3, [])
    
    print("Merges:", merges)
    # Expected: 
    # l o w e r n
    # 'er' (appears twice) -> merge
    # 'ow' (appears once? wait 'lower' 'newer'. 'w' 'e' in 'lower' is 'we'?)
    # l o w e r 
    # n e w e r
    # e r -> er (2)
    # l o w er
    # n e w er
    # ...

if __name__ == "__main__":
    test_debug()
