"""Equal-size control for the fidelity ablation.

Answers: is the gated pool's low coverage a *selection* effect of the rank<=k
gate, or merely a *quantity* effect (fewer candidates)? Reads the saved synthetic
training files and subsamples the un-gated pool down to the gated pool's size
(both by sequence count and by item-occurrence count), then compares diversity.
"""
import os, sys, numpy as np, torch, pandas as pd

DS = {'amazon-sport': 'sport', 'yelp': 'yelp'}
R = 100


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

    # head ranking = target-item freq (train.pth real rows), same as aggregate.py
    train = torch.load(f'{folder}/train.pth', weights_only=False)
    real_cnt = np.zeros(n_items + 1, dtype=np.float64)
    for row in train:
        if row[0] < num_users:
            for it in row[2]:
                if it != 0:
                    real_cnt[int(it)] += 1
    real_cnt = real_cnt[1:]
    order = np.argsort(-real_cnt, kind='stable')

    def load_syn_items(tag):
        seqs = []
        for s in range(5):
            fn = f'{folder}/train_{tag}_{s}.pth'
            if not os.path.exists(fn):
                continue
            data = torch.load(fn, weights_only=False)
            for row in data:
                if row[0] >= num_users:
                    ph, pt, hl = row[1], row[2], row[3]
                    seqs.append([x for x in pt[:hl] if x != 0])  # seq[1:] per sequence
        return seqs  # list of per-sequence item lists

    gated = load_syn_items('greedy_fid')
    ungated = load_syn_items('greedy_nofid')

    def metrics_from_seqs(seqs):
        cc = {}
        for seq in seqs:
            for it in seq:
                cc[it] = cc.get(it, 0) + 1
        f = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        k = int(0.2 * n_items)
        head = f[order[:k]].sum() / f.sum() if f.sum() > 0 else float('nan')
        return gini(f), head, (f > 0).sum() / n_items, len(seqs), sum(len(s) for s in seqs)

    g_gini, g_head, g_cov, g_nseq, g_nitem = metrics_from_seqs(gated)
    u_gini, u_head, u_cov, u_nseq, u_nitem = metrics_from_seqs(ungated)

    print(f'===== {ds} =====')
    print(f'gated   : nseq={g_nseq} nitem={g_nitem}  gini={g_gini:.4f} head20={g_head:.4f} cov={g_cov:.4f}')
    print(f'ungated : nseq={u_nseq} nitem={u_nitem}  gini={u_gini:.4f} head20={u_head:.4f} cov={u_cov:.4f}')

    rng = np.random.default_rng(0)
    # (1) subsample un-gated to gated SEQUENCE count
    idx = np.arange(u_nseq)
    covs, heads, ginis = [], [], []
    for _ in range(R):
        sel = rng.choice(idx, size=g_nseq, replace=False)
        _, h, c, _, _ = metrics_from_seqs([ungated[i] for i in sel])
        heads.append(h); covs.append(c)
    print(f'un-gated subsampled to {g_nseq} seqs : head20={np.mean(heads):.4f}+-{np.std(heads):.4f}  cov={np.mean(covs):.4f}+-{np.std(covs):.4f}')

    # (2) subsample un-gated to gated ITEM count (sample items across the pool)
    all_items = np.array([it for s in ungated for it in s], dtype=np.int64)
    covs2, heads2 = [], []
    for _ in range(R):
        sel = rng.choice(all_items, size=g_nitem, replace=False) if g_nitem <= len(all_items) else rng.choice(all_items, size=g_nitem, replace=True)
        f = np.bincount(sel, minlength=n_items + 1)[1:].astype(np.float64)
        k = int(0.2 * n_items)
        h = f[order[:k]].sum() / f.sum() if f.sum() > 0 else float('nan')
        heads2.append(h); covs2.append((f > 0).sum() / n_items)
    print(f'un-gated subsampled to {g_nitem} items: head20={np.mean(heads2):.4f}+-{np.std(heads2):.4f}  cov={np.mean(covs2):.4f}+-{np.std(covs2):.4f}')
    print()
