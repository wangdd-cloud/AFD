"""Compute popularity-drift across the official RSIR iteration files (train_1th..8th).

For each dataset: load the real train set, then each shipped iteration file. Report
(i) the combined-data (real + accumulated synthetic) concentration and (ii) the
per-round synthetic-only concentration (the multiset delta from the previous round),
which is directly comparable to a single-round generated pool.

Item counting follows the single-round convention: the TARGET items (field 2,
seq[1:]), so the "real" reference is comparable to the generated suffix.
"""
import os
from collections import Counter
import numpy as np
import torch


def gini(freq):
    x = np.asarray(sorted(freq), dtype=np.float64)
    if x.sum() == 0:
        return float('nan')
    x = x / x.sum()
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / n)


def load_counter(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    c = Counter()
    for row in data:
        # RSIR interaction row: [user_id, item_seq, target_seq, seq_len, label, domain_id]
        seq = row[2]
        if isinstance(seq, int):
            if seq != 0:
                c[seq] += 1
        else:
            for it in seq:
                if it != 0:
                    c[it] += 1
    return c, len(data)


def max_item_id(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    m = 0
    for row in data:
        for field in (row[1], row[2]):
            for it in field:
                if it != 0:
                    m = max(m, it)
    return m


def main():
    for dsname, domain in [('amazon-sport', 'sport'), ('yelp', 'yelp')]:
        base = os.path.join('dataset', dsname, domain)
        real_c, real_nseq = load_counter(os.path.join(base, 'train.pth'))
        # full item space = max id over train/val/test (ids are 1-indexed)
        n_items = max(real_c.keys())
        for f in ['val.pth', 'test.pth']:
            n_items = max(n_items, max_item_id(os.path.join(base, f)))

        def vec(counter):
            return np.array([counter.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)

        rf = vec(real_c)
        order = np.argsort(-rf, kind='stable')
        topk = int(0.2 * n_items)

        def head20(f):
            s = f.sum()
            return float(f[order[:topk]].sum() / s) if s > 0 else float('nan')

        def coverage(counter):
            return float(sum(1 for i in range(1, n_items + 1) if counter.get(i, 0) > 0) / n_items)

        print('=== %s (n_items=%d, real_seqs=%d)' % (dsname, n_items, real_nseq), flush=True)
        print('  real      gini=%.4f head20=%.4f cov=%.4f' % (gini(rf), head20(rf), coverage(real_c)), flush=True)

        prev_c = real_c
        prev_k = 0
        for k in range(1, 9):
            p = os.path.join(base, 'train_%dth.pth' % k)
            if not os.path.exists(p):
                print('  iter%d MISSING (skipped)' % k, flush=True)
                continue
            c, nseq = load_counter(p)
            f = vec(c)
            # per-round synthetic-only delta (multiset difference vs previous round)
            d = Counter()
            neg = 0
            for it in set(list(c.keys()) + list(prev_c.keys())):
                diff = c.get(it, 0) - prev_c.get(it, 0)
                if diff < 0:
                    neg += -diff
                elif diff > 0:
                    d[it] = diff
            if k != prev_k + 1:
                print('  iter%d note: previous round %d missing, delta spans %d rounds' % (k, prev_k, k - prev_k), flush=True)
            if neg > 0:
                print('  iter%d WARNING: %d negative-diff occurrences (round is not a superset of previous); SYNTH-ONLY delta is INVALID' % (k, neg), flush=True)
                print('  iter%d  seqs=%7d  COMBINED gini=%.4f head20=%.4f cov=%.4f | SYNTH-ONLY INVALID (neg=%d)' % (
                    k, nseq, gini(f), head20(f), coverage(c), neg), flush=True)
            else:
                df = vec(d)
                dsyn = sum(d.values())
                print('  iter%d  seqs=%7d  COMBINED gini=%.4f head20=%.4f cov=%.4f | SYNTH-ONLY n=%6d gini=%.4f head20=%.4f cov=%.4f' % (
                    k, nseq, gini(f), head20(f), coverage(c),
                    dsyn, gini(df), head20(df), coverage(d)), flush=True)
            prev_c = c
            prev_k = k


if __name__ == '__main__':
    main()
