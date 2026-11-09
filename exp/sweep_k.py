"""Fidelity-gate rank-threshold sweep (k-scan): does coverage recover monotonically
as the gate threshold k is relaxed, without losing successor accuracy?

Greedy decoding (temperature=None) with the RSIR rank gate at k in {20,50,100,200,None(off)}.
Per config: generate the pool, measure diversity over the generated suffix, retrain a
keep-all successor, record Recall@20/NDCG@20. Appends to a per-dataset JSONL.

k=20 reproduces ablation2x2 'greedy_fid' (collapse); k=None reproduces 'greedy_nofid'.
"""
import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from afd_pipeline import (make_config, train_model, format_real_samples,
                          item_freq_from_real, format_synthetic)
from ablation2x2 import generate, gini


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    dsname = args.dataset
    domain = {'amazon-sport': 'sport', 'yelp': 'yelp'}[dsname]
    out_file = 'exp/sweep_k_%s.jsonl' % dsname

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
        kk = int(frac * n_items)
        return f[order[:kk]].sum() / f.sum() if f.sum() > 0 else float('nan')

    def diversity(cands):
        cc = {}
        for c in cands:
            for it in c['gen_seq'][1:]:   # generated suffix only (exclude copied first item)
                if it != 0:
                    cc[it] = cc.get(it, 0) + 1
        f = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        return {'gini': float(gini(f)), 'head20': float(share(f, 0.2)),
                'coverage': float((f > 0).sum() / n_items), 'n_candidates': len(cands)}

    def retrain(cands, tag):
        synth = format_synthetic([c['gen_seq'] for c in cands], num_users, max_seq_len)
        train_data = real_samples + synth
        suffix = '_%s_%d' % (tag, args.seed)
        data_dir = os.path.join('dataset', dsname, domain)
        torch.save(train_data, os.path.join(data_dir, 'train%s.pth' % suffix))
        cfg2 = make_config(dsname, suffix, 2, args.seed, args.device)
        setup_environment(cfg2['train'])
        r, _ = train_model(cfg2, tag)
        return r

    ks = [20, 50, 100, 200, None]   # None = gate off (fully relaxed)

    with open(out_file, 'a') as fh:
        for k in ks:
            tag = 'k_off' if k is None else 'k%d' % k
            cands = generate(model, ds, 5, (k if k is not None else 20), 0.5, min_len=3,
                             device='cuda', max_batches=0, temperature=None,
                             fidelity=(k is not None), item_freq=freq, pop_penalty=0.0)
            div = diversity(cands)
            r = retrain(cands, tag)
            out = {'dataset': dsname, 'seed': args.seed, 'k': k, 'tag': tag,
                   'recall@20': float(r['recall@20']), 'ndcg@20': float(r['ndcg@20']),
                   **div}
            print('RESULT %s' % json.dumps(out), flush=True)
            fh.write(json.dumps(out) + '\n')
            fh.flush()

    print('[DONE]', dsname, 'seed', args.seed, flush=True)


if __name__ == '__main__':
    main()
