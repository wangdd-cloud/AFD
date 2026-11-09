"""
5-seed temperature-decoding run. For one seed: train base, generate a T=2.0 pool,
retrain a keep-all successor, and report accuracy + pool diversity. Saves to a
distinct suffix (_tempRSIR_<seed>) so existing greedy results are not overwritten.
"""
import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from afd_pipeline import (make_config, train_model, generate_candidates,
                          format_real_samples, item_freq_from_real, format_synthetic)


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
    ap.add_argument('--seed', type=int, required=True)
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
    num_users = ds.num_users
    n_items = ds.num_items - 1      # real catalog size (num_items includes padding 0)
    max_seq_len = ds.max_seq_len

    rf = np.array([freq.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
    order = np.argsort(-rf, kind='stable')

    def share(f, frac):
        k = int(frac * n_items)
        return f[order[:k]].sum() / f.sum() if f.sum() > 0 else float('nan')

    cands = generate_candidates(model, ds, 5, 20, 0.5, min_len=3, device='cuda',
                                max_batches=0, temperature=2.0, item_freq=freq, pop_penalty=0.0)
    cc = {}
    for c in cands:
        for it in c['gen_seq'][1:]:   # generated suffix only (exclude copied first item)
            if it != 0:
                cc[it] = cc.get(it, 0) + 1
    f = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
    div = {
        'gini': float(gini(f)),
        'head20': float(share(f, 0.2)),
        'coverage': float((f > 0).sum() / n_items),
        'n_candidates': len(cands),
    }

    synth = format_synthetic([c['gen_seq'] for c in cands], num_users, max_seq_len)
    train_data = real_samples + synth
    suffix = '_tempRSIR_%d' % args.seed
    data_dir = os.path.join('dataset', dsname, domain)
    torch.save(train_data, os.path.join(data_dir, 'train%s.pth' % suffix))
    cfg2 = make_config(dsname, suffix, 2, args.seed, args.device)
    setup_environment(cfg2['train'])
    r2, _ = train_model(cfg2, 'tempRSIR_%d' % args.seed)

    out = {
        'dataset': dsname, 'seed': args.seed,
        'recall@20': float(r2['recall@20']),
        'ndcg@20': float(r2['ndcg@20']),
        'recall@10': float(r2['recall@10']),
        'ndcg@10': float(r2['ndcg@10']),
        **div,
    }
    print('RESULT %s' % json.dumps(out), flush=True)
    with open(os.path.join('exp', 'temp5_%s.jsonl' % dsname), 'a') as fh:
        fh.write(json.dumps(out) + '\n')


if __name__ == '__main__':
    main()
