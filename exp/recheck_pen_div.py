"""Recompute penalty-decoding pool diversity on the suffix convention (seq[1:]),
matching the paper's greedy/temperature rows.

sweep_pen.py counted the full gen_seq (including the copied first item); this
reads the saved train_pen_{config}_{seed}.pth files and recomputes
head20/gini/coverage on the generated suffix (pt[:hl] = seq[1:]).
"""
import os, numpy as np, torch, pandas as pd

DS = {'amazon-sport': 'sport', 'yelp': 'yelp'}
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = ['T2.0', 'T1.0', 'pen0.5', 'pen1.0', 'pen1.5']


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
    train = torch.load(f'{folder}/train.pth', weights_only=False)
    real_cnt = np.zeros(n_items + 1, dtype=np.float64)
    for row in train:
        if row[0] < num_users:
            for it in row[2]:
                if it != 0:
                    real_cnt[int(it)] += 1
    real_cnt = real_cnt[1:]
    order = np.argsort(-real_cnt, kind='stable')

    def share(f):
        k = int(0.2 * n_items)
        return f[order[:k]].sum() / f.sum() if f.sum() > 0 else float('nan')

    def ms(x):
        if len(x) == 0:
            return 'nodata'
        if len(x) == 1:
            return f'{x[0]:.4f}'
        return f'{np.mean(x):.4f}+-{np.std(x, ddof=1):.4f}'

    print(f'===== {ds} (penalty sweep, suffix seq[1:]) =====')
    for cfg in CONFIGS:
        gs, hs, cs = [], [], []
        for s in SEEDS:
            fn = f'{folder}/train_pen_{cfg}_{s}.pth'
            if not os.path.exists(fn):
                continue
            data = torch.load(fn, weights_only=False)
            cc = {}
            for row in data:
                if row[0] >= num_users:
                    pt, hl = row[2], row[3]
                    for it in pt[:hl]:
                        if it != 0:
                            cc[it] = cc.get(it, 0) + 1
            f = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
            gs.append(gini(f)); hs.append(share(f)); cs.append((f > 0).sum() / n_items)
        print(f'  {cfg:<6} (n={len(gs)})  gini={ms(gs)}  head20={ms(hs)}  cov={ms(cs)}')
