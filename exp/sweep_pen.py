"""
Decoding-sweep with retraining: does the popularity logit penalty beat plain
temperature at matched diversity?

Trains the base model once, generates candidate pools under several decoding
configs (temperature vs popularity-penalty), prints pool diversity for each,
then retrains a keep-all RSIR successor per config and reports recall@20/ndcg@20.

Reference configs:
  T2.0     temperature sampling (current "fix")
  T1.0     plain softmax sampling (no temperature, no penalty)
  pen0.5/1.0/1.5   softmax(T=1) minus lambda*log(freq+1)
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from afd_pipeline import (make_config, train_model, generate_candidates,
                          format_real_samples, item_freq_from_real,
                          format_synthetic)


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
    args = ap.parse_args()

    dsname = args.dataset
    domain = {'amazon-sport': 'sport', 'yelp': 'yelp'}[dsname]

    # ---- train base once ----
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

    print('REAL gini=%.4f head20=%.4f cov=%.3f' % (
        gini(rf), share(rf, 0.2), (rf > 0).sum() / n_items), flush=True)

    # ---- generate all pools first (cheap) ----
    configs = [
        ('T2.0', 2.0, 0.0),
        ('T1.0', 1.0, 0.0),
        ('pen0.5', 1.0, 0.5),
        ('pen1.0', 1.0, 1.0),
        ('pen1.5', 1.0, 1.5),
    ]
    pools = {}
    for name, T, pen in configs:
        cands = generate_candidates(model, ds, 5, 20, 0.5, min_len=3, device='cuda',
                                    max_batches=0, temperature=T, item_freq=freq, pop_penalty=pen)
        cc = {}
        for c in cands:
            for it in c['gen_seq'][1:]:   # generated suffix only (exclude copied first item)
                if it != 0:
                    cc[it] = cc.get(it, 0) + 1
        f = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        pools[name] = (cands, f)
        print('[%s] pool n=%d gini=%.4f head20=%.4f cov=%.3f' % (
            name, len(cands), gini(f), share(f, 0.2), (f > 0).sum() / n_items), flush=True)

    # ---- retrain keep-all successor per config ----
    data_dir = os.path.join('dataset', dsname, domain)
    for name, _, _ in configs:
        cands, _ = pools[name]
        synth = format_synthetic([c['gen_seq'] for c in cands], num_users, max_seq_len)
        train_data = real_samples + synth
        suffix = '_pen_%s_%d' % (name, args.seed)
        torch.save(train_data, os.path.join(data_dir, 'train%s.pth' % suffix))
        cfg2 = make_config(dsname, suffix, 2, args.seed, args.device)
        setup_environment(cfg2['train'])
        r2, _ = train_model(cfg2, 'pen_%s' % name)
        print('[%s] recall@20=%.4f ndcg@20=%.4f recall@10=%.4f ndcg@10=%.4f' % (
            name, float(r2['recall@20']), float(r2['ndcg@20']),
            float(r2['recall@10']), float(r2['ndcg@10'])), flush=True)

    print('[ALL DONE] %s seed %d' % (dsname, args.seed), flush=True)


if __name__ == '__main__':
    main()
