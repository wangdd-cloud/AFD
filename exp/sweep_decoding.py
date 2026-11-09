"""
Pool-level sweep over decoding configs (temperature, popularity penalty).

Trains the base model once, then generates the candidate pool under several
decoding configs and reports pool-level diversity (gini / head20 / coverage)
against the real-data reference. Retraining is NOT done here; this only maps
the decoding -> diversity landscape so we can pick a good operating point.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from afd_pipeline import (make_config, train_model, generate_candidates,
                          format_real_samples, item_freq_from_real)


def gini(freq):
    x = np.asarray(sorted(freq), dtype=np.float64)
    if x.sum() == 0:
        return float('nan')
    x = x / x.sum()
    n = len(x)
    idx = np.arange(1, n + 1)
    return (np.sum((2 * idx - n - 1) * x)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--m', type=int, default=5)
    ap.add_argument('--local_prob', type=float, default=0.5)
    args = ap.parse_args()

    # base model (already trained in afd_results_v2 for seed0; retrain cheap & safe)
    cfg = make_config(args.dataset, '', 2, args.seed, args.device)
    setup_environment(cfg['train'])
    _, model = train_model(cfg, 'base')

    cfg = make_config(args.dataset, '', 2, args.seed, args.device)
    setup_environment(cfg['train'])
    ds = prepare_datasets(cfg)[0]
    real_samples = format_real_samples(ds)
    freq = item_freq_from_real(real_samples)
    n_items = ds.num_items - 1     # real catalog size (num_items includes padding 0)

    # real reference
    rf = np.array([freq.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
    order = np.argsort(-rf, kind='stable')
    n = n_items
    idx = np.arange(1, n + 1)

    def share(f, frac):
        k = int(frac * n_items)
        return f[order[:k]].sum() / f.sum() if f.sum() > 0 else float('nan')

    real_gini = gini(rf)
    real_head = share(rf, 0.2)
    real_cov = (rf > 0).sum() / n_items
    print('REAL  gini=%.4f head20=%.4f cov=%.3f' % (real_gini, real_head, real_cov), flush=True)

    # (name, temperature, pop_penalty)
    configs = [
        ('T1.0', 1.0, 0.0),
        ('T1.2', 1.2, 0.0),
        ('T1.4', 1.4, 0.0),
        ('T1.6', 1.6, 0.0),
        ('T1.8', 1.8, 0.0),
        ('T2.0', 2.0, 0.0),
        ('pen0.5', 1.0, 0.5),
        ('pen1.0', 1.0, 1.0),
        ('pen1.5', 1.0, 1.5),
        ('T1.5+pen0.5', 1.5, 0.5),
    ]

    for name, T, pen in configs:
        cands = generate_candidates(model, ds, args.m, args.k, args.local_prob,
                                    min_len=3, device='cuda', max_batches=0,
                                    temperature=T, item_freq=freq, pop_penalty=pen)
        cc = {}
        for c in cands:
            for it in c['gen_seq'][1:]:
                if it != 0:
                    cc[it] = cc.get(it, 0) + 1
        f = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        cov = (f > 0).sum() / n_items
        print('%-14s n=%-7d gini=%.4f head20=%.4f cov=%.3f' % (
            name, len(cands), gini(f), share(f, 0.2), cov), flush=True)


if __name__ == '__main__':
    main()
