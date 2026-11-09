"""Equal-size coverage sampling, corrected (item-4 / reviewer P0-2).

Regenerates the greedy pool with argmax decoding (temperature=None) so we do NOT
depend on the overwritten train_RSIR pth files, then tests whether its low
coverage is a decoding-induced concentration or merely a sample-size artifact.

Null: draw the same number of item occurrences i.i.d. from the real training
item-frequency distribution and report coverage / head-20% / Gini. Greedy pool
diversity is computed over generated suffixes (seq[1:]), consistent with Table 1.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from afd_pipeline import make_config, train_model, format_real_samples, item_freq_from_real
from ablation2x2 import generate, gini

DS = {'amazon-sport': 'sport', 'yelp': 'yelp'}
R = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    ds = args.dataset
    domain = DS[ds]
    # train base + regenerate greedy pool (argmax + fidelity)
    cfg = make_config(ds, '', 2, args.seed, args.device)
    setup_environment(cfg['train'])
    _, model = train_model(cfg, 'base')
    cfg = make_config(ds, '', 2, args.seed, args.device)
    setup_environment(cfg['train'])
    dsx = prepare_datasets(cfg)[0]
    real_samples = format_real_samples(dsx)
    freq = item_freq_from_real(real_samples)

    # real item-frequency distribution (train target counts, matching the main table)
    n_items = dsx.num_items - 1
    items = np.arange(1, n_items + 1)
    real_cnt = np.array([freq.get(i, 0) for i in items], dtype=np.float64)
    real_p = real_cnt / real_cnt.sum()
    order = np.argsort(-real_cnt, kind='stable')

    def share(f):
        k = int(0.2 * n_items)
        return f[order[:k]].sum() / f.sum() if f.sum() > 0 else float('nan')

    cands = generate(model, dsx, 5, 20, 0.5, min_len=3, device='cuda',
                     max_batches=0, temperature=None, fidelity=True,
                     item_freq=freq, pop_penalty=0.0)

    syn_items = np.array([it for c in cands for it in c['gen_seq'][1:] if it != 0],
                         dtype=np.int64)
    I_syn = len(syn_items)

    def metrics(cnt):
        f = np.array([np.count_nonzero(cnt == i) for i in items], dtype=np.float64)
        return gini(f), share(f), (f > 0).sum() / n_items

    g_gini, g_head, g_cov = metrics(syn_items)
    print(f'===== {ds} (seed {args.seed}) =====')
    print(f'greedy pool: n_cand={len(cands)} I={I_syn}  gini={g_gini:.4f} head20={g_head:.4f} cov={g_cov:.4f}')

    rng = np.random.default_rng(0)
    draws = rng.choice(items, size=(R, I_syn), replace=True, p=real_p)
    covs, heads, ginis = [], [], []
    for r in range(R):
        f = np.bincount(draws[r], minlength=n_items + 1)[1:].astype(np.float64)
        ginis.append(gini(f)); heads.append(share(f)); covs.append((f > 0).sum() / n_items)
    print(f'null (I={I_syn} iid from real freq): '
          f'gini={np.mean(ginis):.4f}+-{np.std(ginis):.4f}  '
          f'head20={np.mean(heads):.4f}+-{np.std(heads):.4f}  '
          f'cov={np.mean(covs):.4f}+-{np.std(covs):.4f}')
    print(f'gap vs null: gini {g_gini-np.mean(ginis):+.3f}  head20 {g_head-np.mean(heads):+.3f}  cov {g_cov-np.mean(covs):+.3f}')
    print()


if __name__ == '__main__':
    main()
