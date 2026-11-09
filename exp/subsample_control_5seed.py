import os, numpy as np, torch, pandas as pd
DS = {"amazon-sport": "sport", "yelp": "yelp"}
R = 100

def gini(freq):
    x = np.asarray(sorted(freq), dtype=np.float64)
    if x.sum() == 0: return float("nan")
    x = x / x.sum(); n = len(x); idx = np.arange(1, n + 1)
    return (np.sum((2 * idx - n - 1) * x)) / n

for ds, domain in DS.items():
    folder = f"dataset/{ds}/{domain}"
    inter = pd.read_csv(f"{folder}/inter.csv")
    n_items = int(inter.item_id.nunique())
    num_users = int(inter.user_id.nunique()) + 1
    train = torch.load(f"{folder}/train.pth", weights_only=False)
    real_cnt = np.zeros(n_items + 1, dtype=np.float64)
    for row in train:
        if row[0] < num_users:
            for it in row[2]:
                if it != 0: real_cnt[int(it)] += 1
    real_cnt = real_cnt[1:]
    order = np.argsort(-real_cnt, kind="stable")

    def load_syn(tag, s):
        seqs = []
        for row in torch.load(f"{folder}/train_{tag}_{s}.pth", weights_only=False):
            if row[0] >= num_users:
                pt, hl = row[2], row[3]
                seqs.append([x for x in pt[:hl] if x != 0])
        return seqs

    def metrics(seqs):
        cc = {}
        for seq in seqs:
            for it in seq: cc[it] = cc.get(it, 0) + 1
        f = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        k = int(0.2 * n_items)
        head = f[order[:k]].sum() / f.sum() if f.sum() > 0 else float("nan")
        return gini(f), head, (f > 0).sum() / n_items, len(seqs), sum(len(s) for s in seqs)

    gcov, useq, uitem, ghead = [], [], [], []
    for s in range(5):
        gated = load_syn("greedy_fid", s)
        ungated = load_syn("greedy_nofid", s)
        g_gini, g_head, g_cov, g_nseq, g_nitem = metrics(gated)
        _, _, u_cov, u_nseq, u_nitem = metrics(ungated)
        gcov.append(g_cov); ghead.append(g_head)
        rng = np.random.default_rng(s)
        idx = np.arange(u_nseq); cs = []
        for _ in range(R):
            sel = rng.choice(idx, size=g_nseq, replace=False)
            _, _, c, _, _ = metrics([ungated[i] for i in sel]); cs.append(c)
        useq.append(np.mean(cs))
        all_items = np.array([it for seq in ungated for it in seq], dtype=np.int64)
        cs2 = []
        for _ in range(R):
            sel = rng.choice(all_items, size=g_nitem, replace=False) if g_nitem <= len(all_items) else rng.choice(all_items, size=g_nitem, replace=True)
            f = np.bincount(sel, minlength=n_items + 1)[1:].astype(np.float64)
            cs2.append((f > 0).sum() / n_items)
        uitem.append(np.mean(cs2))

    sd = lambda a: float(np.std(a, ddof=1))
    print(f"===== {ds} =====")
    print(f"gated cov   : {np.mean(gcov):.4f} +- {sd(gcov):.4f}")
    print(f"gated head20: {np.mean(ghead):.4f} +- {sd(ghead):.4f}")
    print(f"ung->seq    : {np.mean(useq):.4f} +- {sd(useq):.4f}")
    print(f"ung->item   : {np.mean(uitem):.4f} +- {sd(uitem):.4f}")
    print(f"ratio (seq) : {np.mean(useq)/np.mean(gcov):.2f}x")
