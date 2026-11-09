"""local_prob=0 control + copied-vs-new split for the greedy fidelity-gated pool.

With local_prob=0.5 RSIR copies remaining history items into the suffix, so the
reported coverage/head20 mixes copied history items with newly predicted items.
This script (a) splits the greedy+gated suffix into copied (in the user's history)
vs new (not in history) items, and (b) generates a local_prob=0 control where
every step samples the full catalog (no history copying).

Diversity is computed over the generated suffix (seq[1:]). No retraining is done.
"""
import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from afd_pipeline import make_config, train_model, format_real_samples, item_freq_from_real
from ablation2x2 import generate, gini


def dist(items, freq, n_items):
    cc = {}
    for it in items:
        if it != 0:
            cc[it] = cc.get(it, 0) + 1
    f = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
    rf = np.array([freq.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
    order = np.argsort(-rf, kind='stable')
    k = int(0.2 * n_items)
    head = f[order[:k]].sum() / f.sum() if f.sum() > 0 else float('nan')
    return {
        'n': int((f > 0).sum()),
        'gini': float(gini(f)),
        'head20': float(head),
        'coverage': float((f > 0).sum() / n_items),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    dsname = args.dataset
    domain = {'amazon-sport': 'sport', 'yelp': 'yelp'}[dsname]

    cfg = make_config(dsname, '', 2, args.seed, args.device)
    setup_environment(cfg['train'])
    _, model = train_model(cfg, 'base')

    cfg = make_config(dsname, '', 2, args.seed, args.device)
    setup_environment(cfg['train'])
    ds = prepare_datasets(cfg)[0]
    real_samples = format_real_samples(ds)
    freq = item_freq_from_real(real_samples)
    n_items = ds.num_items - 1

    # (a) local_prob=0.5 (RSIR default): split copied vs new
    cands = generate(model, ds, 5, 20, 0.5, min_len=3, device='cuda',
                     max_batches=0, temperature=None, fidelity=True,
                     item_freq=freq, pop_penalty=0.0)
    copied, new = [], []
    for c in cands:
        hist = set(c['real_seq'])
        for it in c['gen_seq'][1:]:
            if it != 0:
                (copied if it in hist else new).append(it)

    # (b) local_prob=0 control (no history copying), fidelity on
    cands0 = generate(model, ds, 5, 20, 0.0, min_len=3, device='cuda',
                      max_batches=0, temperature=None, fidelity=True,
                      item_freq=freq, pop_penalty=0.0)
    all0 = [it for c in cands0 for it in c['gen_seq'][1:] if it != 0]

    # (c) local_prob=0, fidelity off (fully relaxed)
    cands0_off = generate(model, ds, 5, 20, 0.0, min_len=3, device='cuda',
                          max_batches=0, temperature=None, fidelity=False,
                          item_freq=freq, pop_penalty=0.0)
    all0_off = [it for c in cands0_off for it in c['gen_seq'][1:] if it != 0]

    out = {
        'dataset': dsname, 'seed': args.seed,
        'lp0.5_copied': dist(copied, freq, n_items),
        'lp0.5_new': dist(new, freq, n_items),
        'lp0.0_fid_on': dist(all0, freq, n_items),
        'lp0.0_fid_off': dist(all0_off, freq, n_items),
    }
    print('RESULT %s' % json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
