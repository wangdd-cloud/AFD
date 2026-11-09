"""Recompute temperature-decoding diversity on the SAME convention as the paper's
greedy row (aggregate.py counts s[2]=seq[1:], excluding the copied first item).

run_temp5.py computed temperature diversity on the FULL generated seq (including
the copied first item), which is inconsistent with the greedy row. This reads the
saved train_tempRSIR_{seed}.pth files and recomputes on seq[1:] (and full, for
cross-check against temp5_{ds}.jsonl)."""
import os, numpy as np, torch, pandas as pd

DS = {'amazon-sport': 'sport', 'yelp': 'yelp'}
SEEDS = [0, 1, 2, 3, 4]


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
    # head ranking = target-item frequency (train.pth real rows, s[2]=tgt), same as aggregate.py
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

    g_suf, h_suf, c_suf, g_full, h_full, c_full = [], [], [], [], [], []
    for s in SEEDS:
        fn = f'{folder}/train_tempRSIR_{s}.pth'
        if not os.path.exists(fn):
            continue
        data = torch.load(fn, weights_only=False)
        cc_suf, cc_full = {}, {}
        for row in data:
            if row[0] >= num_users:
                ph, pt, hl = row[1], row[2], row[3]
                suffix = [x for x in pt[:hl] if x != 0]
                full = [ph[0]] + suffix
                for it in suffix:
                    cc_suf[it] = cc_suf.get(it, 0) + 1
                for it in full:
                    cc_full[it] = cc_full.get(it, 0) + 1
        f_suf = np.array([cc_suf.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        f_full = np.array([cc_full.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        g_suf.append(gini(f_suf)); h_suf.append(share(f_suf)); c_suf.append((f_suf > 0).sum() / n_items)
        g_full.append(gini(f_full)); h_full.append(share(f_full)); c_full.append((f_full > 0).sum() / n_items)

    def ms(x):
        return f'{np.mean(x):.4f}+-{np.std(x, ddof=1):.4f}'

    print(f'===== {ds} (temp T=2.0, {len(g_suf)} seeds) =====')
    print(f'  seq[1:]  gini={ms(g_suf)}  head20={ms(h_suf)}  cov={ms(c_suf)}')
    print(f'  full     gini={ms(g_full)}  head20={ms(h_full)}  cov={ms(c_full)}')
    print()
