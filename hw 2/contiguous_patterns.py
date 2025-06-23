import itertools

def read_sequences(filename):
    """
    Read the reviews_sample.txt file and return a list of token lists (each review).
    """
    sequences = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = [tok.strip() for tok in line.strip().split() if tok.strip()]
            if tokens:
                sequences.append(tokens)
    return sequences


def mine_contiguous_patterns(sequences, min_support):
    """
    Mine all frequent contiguous sequential patterns with support >= min_support.
    Returns a dict mapping pattern tuples to their support counts.
    """
    freq_patterns = {}
    # length-1 patterns
    count1 = {}
    for seq in sequences:
        for item in set(seq):
            count1[(item,)] = count1.get((item,), 0) + 1
    L_prev = {p: c for p, c in count1.items() if c >= min_support}
    freq_patterns.update(L_prev)

    k = 2
    while L_prev:
        cand_counts = {}
        for seq in sequences:
            seen = set()
            for i in range(len(seq) - k + 1):
                pat = tuple(seq[i : i + k])
                seen.add(pat)
            for pat in seen:
                cand_counts[pat] = cand_counts.get(pat, 0) + 1
        Lk = {p: c for p, c in cand_counts.items() if c >= min_support}
        if not Lk:
            break
        freq_patterns.update(Lk)
        L_prev = Lk
        k += 1

    return freq_patterns


def write_patterns(patterns, filename):
    """
    Write patterns to patterns.txt in format:
    support:item1;item2;...
    Sorted by support descending, then length ascending, then lex.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for pat, sup in sorted(
            patterns.items(), key=lambda x: (-x[1], len(x[0]), x[0])
        ):
            f.write(f"{sup}:{';'.join(pat)}\n")


def main():
    sequences = read_sequences('reviews_sample.txt')
    min_support = int(0.01 * len(sequences))
    patterns = mine_contiguous_patterns(sequences, min_support)
    write_patterns(patterns, 'patterns.txt')


if __name__ == '__main__':
    main()
