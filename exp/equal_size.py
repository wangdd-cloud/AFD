"""Equal-size coverage sampling (reviewer P0-2 / item-4).

Tests whether the greedy synthetic pool's low coverage (0.266 / 0.188) is a
decoding-induced concentration effect or merely a sample-size artifact of the
small candidate pool. Null: draw the same number of item occurrences i.i.d.
from the real training item-frequency distribution, and report the coverage /
head-20% share distribution. Greedy pool is read from the saved synthetic
training file (train_RSIR_{seed}.pth, the same source aggregate.py used to
produce the paper's head-collapse numbers).
"""
import os, sys, json, glob
import numpy as np
import torch
import pandas as pd

DS = {'amazon-sport': 'sport', 'yelp': 'yelp'}
SEEDS = [0, 1, 2, 3, 4]
R = 200  # null replicates


def gini(freq):
    x = np.asarray(sorted(freq), dtype=np.float64)
    if x.sum() == 0:
        return float('nan')
    x = x / x.sum()
    n = len(x)
    idx = np.arange(1, n + 1)
    return (np.sum((2 * idx - n - 1) * x)) / n


for ds, domain in DS.items():
    folder = f'dataset/{ds}/{domain}'
    inter = pd.read_csv(f'{folder}/inter.csv')
    n_items = int(inter.item_id.nunique())
    num_users = int(inter.user_id.nunique()) + 1

    # real training item-frequency distribution (all interactions)
    real_freq = inter.item_id.value_counts().to_dict()
    items = np.arange(1, n_items + 1)
    real_cnt = np.array([real_freq.get(i, 0) for i in items], dtype=np.float64)
    real_p = real_cnt / real_cnt.sum()

    # greedy synthetic pool: concatenate items over the 5 saved seeds
    syn_items = []
    for s in SEEDS:
        hits = sorted(glob.glob(f'{folder}/train_RSIR_{s}_*.pth'))
        if len(hits) > 1:
            raise SystemExit('ambiguous: %d train files for seed %d;\n  %s' % (
                len(hits), s, '\n  '.join(hits)))
        if not hits:
            continue
        data = torch.load(hits[0], weights_only=False)
        for row in data:
            if row[0] >= num_users:            # synthetic rows
                for it in row[2]:              # target field, same as aggregate.py
                    if it != 0:
                        syn_items.append(int(it))
    syn_items = np.array(syn_items, dtype=np.int64)
    I_syn = len(syn_items)

    def metrics(cnt):
        f = np.array([np.count_nonzero(cnt == i) for i in items], dtype=np.float64)
        order = np.argsort(-real_cnt, kind='stable')
        k = int(0.2 * n_items)
        head = f[order[:k]].sum() / f.sum() if f.sum() > 0 else float('nan')
        cov = (f > 0).sum() / n_items
        return gini(f), head, cov

    g_gini, g_head, g_cov = metrics(syn_items)
    print(f'===== {ds} =====')
    print(f'greedy pool: I={I_syn}  gini={g_gini:.4f} head20={g_head:.4f} cov={g_cov:.4f}')

    # null: I_syn i.i.d. draws from the real frequency distribution
    rng = np.random.default_rng(0)
    draws = rng.choice(items, size=(R, I_syn), replace=True, p=real_p)
    covs, heads, ginis = [], [], []
    for r in range(R):
        _, h, c = metrics(draws[r])
        heads.append(h); covs.append(c); ginis.append(gini(np.bincount(draws[r], minlength=n_items + 1)[1:]))
    print(f'null (I={I_syn} iid from real freq): '
          f'gini={np.mean(ginis):.4f}+-{np.std(ginis):.4f}  '
          f'head20={np.mean(heads):.4f}+-{np.std(heads):.4f}  '
          f'cov={np.mean(covs):.4f}+-{np.std(covs):.4f}')
    print(f'gap vs null: gini {g_gini-np.mean(ginis):+.3f}  head20 {g_head-np.mean(heads):+.3f}  cov {g_cov-np.mean(covs):+.3f}')
    print()
