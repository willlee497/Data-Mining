import itertools


def read_transactions(filename):
    """
    Read the categories.txt file and return a list of transactions,
    each transaction is a set of category strings.
    """
    transactions = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            items = [item.strip() for item in line.strip().split(';') if item.strip()]
            if items:
                transactions.append(set(items))
    return transactions


def get_frequent_itemsets(transactions, min_support):
    """
    Run the Apriori algorithm on the list of transactions and
    return a dict mapping each frequent itemset (frozenset) to its support count.
    """
    # Count single items (L1)
    item_counts = {}
    for trans in transactions:
        for item in trans:
            item_counts[item] = item_counts.get(item, 0) + 1
    L1 = {frozenset([item]): count for item, count in item_counts.items() if count > min_support}

    # Store all frequent itemsets
    frequent_itemsets = dict(L1)

    # Iteratively build larger itemsets
    Lk_prev = set(L1.keys())
    k = 2
    while Lk_prev:
        # Candidate generation (join step)
        candidates = set()
        Lk_prev_list = list(Lk_prev)
        for i in range(len(Lk_prev_list)):
            for j in range(i + 1, len(Lk_prev_list)):
                union_set = Lk_prev_list[i] | Lk_prev_list[j]
                if len(union_set) == k:
                    # Prune: all (k-1)-subsets must be frequent
                    subsets = itertools.combinations(union_set, k - 1)
                    if all(frozenset(sub) in Lk_prev for sub in subsets):
                        candidates.add(union_set)

        # Count supports for candidates
        candidate_counts = {c: 0 for c in candidates}
        for trans in transactions:
            for c in candidates:
                if c.issubset(trans):
                    candidate_counts[c] += 1

        # Filter by min_support
        Lk = {c: count for c, count in candidate_counts.items() if count > min_support}
        if not Lk:
            break

        # Add to overall frequent itemsets
        frequent_itemsets.update(Lk)
        Lk_prev = set(Lk.keys())
        k += 1

    return frequent_itemsets


def write_part1(frequent_itemsets, filename):
    """
    Write all length-1 frequent itemsets to part1.txt in the format support:category
    """
    with open(filename, 'w', encoding='utf-8') as f:
        # Sort by descending support, then alphabetically
        for itemset, count in sorted(
            ((iset, cnt) for iset, cnt in frequent_itemsets.items() if len(iset) == 1),
            key=lambda x: (-x[1], sorted(x[0]))
        ):
            item = next(iter(itemset))
            f.write(f"{count}:{item}\n")


def write_part2(frequent_itemsets, filename):
    """
    Write all frequent itemsets (length >= 1) to part2.txt in the format
    support:cat1;cat2;...
    """
    with open(filename, 'w', encoding='utf-8') as f:
        # Sort by size, then descending support, then lexicographically
        for itemset, count in sorted(
            frequent_itemsets.items(),
            key=lambda x: (len(x[0]), -x[1], sorted(x[0]))
        ):
            items = sorted(itemset)
            line = ";".join(items)
            f.write(f"{count}:{line}\n")


def main():
    transactions = read_transactions('categories.txt')
    # Minimum support threshold: 1% of total transactions (non-inclusive)
    min_support = int(0.01 * len(transactions))
    frequent_itemsets = get_frequent_itemsets(transactions, min_support)

    write_part1(frequent_itemsets, 'part1.txt')
    write_part2(frequent_itemsets, 'part2.txt')


if __name__ == '__main__':
    main()
