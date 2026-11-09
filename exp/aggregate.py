"""
Aggregate AFD-RSIR results across seeds: accuracy (mean+-std) + diversity/popularity metrics.

Reads exp/afd_results/results_{ds}_seed{s}.json (accuracy) and
dataset/{ds}/{domain}/train_{method}_{seed}.pth (synthetic-popularity metrics),
then prints LaTeX-ready tables and saves JSON.
"""
import json, os, glob, warnings
import numpy as np
import torch
import pandas as pd

warnings.filterwarnings('ignore')

DATASETS = {'amazon-sport': 'sport', 'yelp': 'yelp'}
SEEDS = [0, 1, 2, 3, 4]
METHODS = ['SASRec', 'RSIR', 'RSIR_random', 'RSIR_novelty', 'RSIR_MMR', 'RSIR_DPP', 'AFD_RSIR']
ACC_KEYS = ['recall@20', 'ndcg@20', 'recall@10', 'ndcg@10']


def _one(pattern):
    """Return the unique file matching `pattern`, or None if none exists.

    A per-variant signature is part of the filename, so an ambiguous match means
    several experiments were mixed into one folder; refuse to pick one silently.
    """
    hits = sorted(glob.glob(pattern))
    if len(hits) > 1:
        raise SystemExit('ambiguous: %d files match %r;\n  %s' % (
            len(hits), pattern, '\n  '.join(hits)))
    return hits[0] if hits else None


def gini(freq):
    x = np.asarray(sorted(freq), dtype=np.float64)
    if x.sum() == 0:
        return float('nan')
    x = x / x.sum()
    n = len(x)
    idx = np.arange(1, n + 1)
    return (np.sum((2 * idx - n - 1) * x)) / n


def item_occ(samples):
    cnt = {}
    for s in samples:
        for it in s[2]:
            if it != 0:
                cnt[it] = cnt.get(it, 0) + 1
    return cnt


def diversity_metrics(ds, domain, seed):
    folder = f'dataset/{ds}/{domain}'
    inter = pd.read_csv(f'{folder}/inter.csv')
    n_items = int(inter.item_id.nunique())
    num_users = int(inter.user_id.nunique()) + 1

    train = torch.load(f'{folder}/train.pth', weights_only=False)
    real = [s for s in train if s[0] < num_users]
    rc = item_occ(real)
    rf = np.array([rc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
    order = np.argsort(-rf, kind='stable')
    rank_of_item = {int(order[j]) + 1: j + 1 for j in range(n_items)}

    def share(freq, frac):
        k = int(frac * n_items)
        return freq[order[:k]].sum() / freq.sum() if freq.sum() > 0 else float('nan')

    out = {}
    for m in METHODS:
        if m == 'SASRec':
            continue
        fn = _one(f'{folder}/train_{m}_{seed}_*.pth')
        if fn is None:
            out[m] = None
            continue
        data = torch.load(fn, weights_only=False)
        syn = [s for s in data if s[0] >= num_users]
        cc = item_occ(syn)
        freq = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        occ = [i for i in cc if cc[i] > 0]
        mean_rank = np.mean([rank_of_item[i] for i in occ]) if occ else float('nan')
        out[m] = {
            'n_synthetic': len(syn),
            'gini': gini(freq),
            'head20': share(freq, 0.2),
            'tail80': 1 - share(freq, 0.2),
            'coverage': (freq > 0).sum() / n_items * 100,
            'mean_pct': (1 - mean_rank / n_items) * 100,
            'amp_ratio': share(freq, 0.2) / share(rf, 0.2) if share(rf, 0.2) > 0 else float('nan'),
        }
    # base reference values (real train)
    out['_real'] = {
        'gini': gini(rf),
        'head20': share(rf, 0.2),
        'tail80': 1 - share(rf, 0.2),
        'coverage': (rf > 0).sum() / n_items * 100,
        'n_items': n_items,
        'n_users': num_users - 1,
    }
    return out


def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def paired_t(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float('nan')
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        # Zero variance across seeds: the difference is constant, so the t-test is
        # degenerate (0/0). An all-zero difference means "no difference", never a
        # significant effect; a constant non-zero difference is undefined. Report
        # undefined rather than a spurious p=0.
        return float('nan')
    t = d.mean() / se
    from math import erf, sqrt
    # two-sided p via normal approx (5 seeds, use t-dist)
    from scipy.stats import t as tdist
    p = 2 * tdist.sf(abs(t), len(d) - 1)
    return float(p)


def main():
    all_acc = {ds: {} for ds in DATASETS}
    all_div = {ds: {} for ds in DATASETS}

    for ds, domain in DATASETS.items():
        # accuracy
        for m in METHODS:
            vals = []
            for s in SEEDS:
                fp = _one(f'exp/afd_results/results_{ds}_seed{s}_*.json')
                if fp is None:
                    vals.append(None)
                    continue
                d = json.load(open(fp))
                if m in d:
                    vals.append(d[m].get('recall@20'))
                else:
                    vals.append(None)
            all_acc[ds][m] = vals

        # diversity
        div_by_method = {m: [] for m in METHODS if m != 'SASRec'}
        real_ref = None
        for s in SEEDS:
            dm = diversity_metrics(ds, domain, s)
            if real_ref is None:
                real_ref = dm['_real']
            for m in div_by_method:
                div_by_method[m].append(dm.get(m))
        all_div[ds] = {'_real': real_ref, **div_by_method}

    # ---- print accuracy table ----
    print('\n================ ACCURACY (mean +- std over 5 seeds) ================')
    for ds in DATASETS:
        print(f'\n### {ds}')
        print(f'{"method":<14} {"recall@20":<22} {"ndcg@20":<22} {"recall@10":<22}')
        base = None
        for m in METHODS:
            r20 = all_acc[ds][m]
            mu, sd = mean_std(r20)
            # also fetch ndcg etc for completeness
            nd20, nd10 = [], []
            for s in SEEDS:
                fp = _one(f'exp/afd_results/results_{ds}_seed{s}_*.json')
                if fp is not None:
                    d = json.load(open(fp))
                    if m in d:
                        nd20.append(d[m].get('ndcg@20'))
                        nd10.append(d[m].get('recall@10'))
                    else:
                        nd20.append(None); nd10.append(None)
            m20, s20 = mean_std(nd20)
            m10, s10 = mean_std(nd10)
            line = f'{m:<14} {mu:.4f}+-{sd:.4f}      {m20:.4f}+-{s20:.4f}      {m10:.4f}+-{s10:.4f}'
            print(line)
            if m == 'SASRec':
                base = r20
        # significance: each method vs SASRec, and AFD vs RSIR
        print(f'--- paired t-test (recall@20) vs SASRec / RSIR ---')
        rsir = all_acc[ds]['RSIR']
        afd = all_acc[ds]['AFD_RSIR']
        for m in METHODS[1:]:
            p_vs_base = paired_t(all_acc[ds][m], base)
            print(f'  {m:<14} vs SASRec p={p_vs_base:.4f}')
        print(f'  {"AFD vs RSIR":<14} p={paired_t(afd, rsir):.4f}')

    # ---- print diversity table ----
    print('\n================ DIVERSITY / POPULARITY (synthetic seq, mean+-std) ================')
    div_keys = ['gini', 'head20', 'tail80', 'coverage', 'mean_pct', 'amp_ratio']
    for ds in DATASETS:
        print(f'\n### {ds}')
        real = all_div[ds]['_real']
        print(f'  REAL ref: gini={real["gini"]:.4f} head20={real["head20"]:.4f} tail80={real["tail80"]:.4f} cov={real["coverage"]:.1f}% items={real["n_items"]}')
        print(f'{"method":<14} ' + ' '.join(f'{k:>12}' for k in div_keys))
        for m in METHODS:
            if m == 'SASRec':
                continue
            recs = [r for r in all_div[ds][m] if r is not None]
            if not recs:
                print(f'{m:<14} (no data)')
                continue
            cells = []
            for k in div_keys:
                mu, sd = mean_std([r[k] for r in recs])
                cells.append(f'{mu:>8.4f}+-{sd:.3f}' if k != 'coverage' else f'{mu:>8.2f}+-{sd:.2f}')
            print(f'{m:<14} ' + '  '.join(f'{c:>14}' for c in cells))

    # ---- save JSON ----
    out = {'accuracy': {ds: {m: {'recall@20': mean_std(all_acc[ds][m])} for m in METHODS} for ds in DATASETS},
           'diversity': {ds: {m: {k: mean_std([r[k] for r in all_div[ds][m] if r is not None]) for k in div_keys}
                              for m in all_div[ds] if m != '_real'} for ds in DATASETS}}
    json.dump(out, open('exp/aggregated_results.json', 'w'), indent=2)
    print('\nSaved to exp/aggregated_results.json')


if __name__ == '__main__':
    main()
