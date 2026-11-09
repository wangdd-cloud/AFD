"""Recommendation-list diversity for the 2x2 fidelity ablation (reviewer issue #10).

Training-pool diversity (Gini / head-20% / coverage of generated sequences) does
not by itself imply recommendation-output diversity. This script reruns the four
2x2 cells (greedy/temperature x fidelity gate on/off) and, in addition to pool
diversity and Recall@20, reports the retrained successor's recommendation-list
coverage@20 and tail Recall@20 (see model_cov_tail in afd_pipeline.py).
"""
import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from afd_pipeline import (make_config, train_model, format_real_samples,
                          item_freq_from_real, format_synthetic, model_cov_tail)
from ablation2x2 import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    dsname = args.dataset
    domain = {'amazon-sport': 'sport', 'yelp': 'yelp'}[dsname]
    out_file = f'exp/rec_div_{dsname}.jsonl'

    cfg = make_config(dsname, '', 2, args.seed, args.device)
    setup_environment(cfg['train'])
    _, base_model = train_model(cfg, 'base')

    cfg = make_config(dsname, '', 2, args.seed, args.device)
    setup_environment(cfg['train'])
    ds = prepare_datasets(cfg)[0]
    real_samples = format_real_samples(ds)
    freq = item_freq_from_real(real_samples)
    num_users = ds.num_users
    n_items = ds.num_items - 1
    max_seq_len = ds.max_seq_len

    rf = np.array([freq.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
    head_order = np.argsort(-rf, kind='stable')
    n_head = int(0.2 * n_items)
    tail_set = {int(head_order[j]) + 1 for j in range(n_head, n_items)}

    def retrain(cands, tag):
        synth = format_synthetic([c['gen_seq'] for c in cands], num_users, max_seq_len)
        train_data = real_samples + synth
        suffix = f'_recdiv_{tag}_{args.seed}'
        data_dir = os.path.join('dataset', dsname, domain)
        train_path = os.path.join(data_dir, f'train{suffix}.pth')
        tmp_path = train_path + f'.{os.getpid()}.tmp'
        torch.save(train_data, tmp_path)
        os.replace(tmp_path, train_path)
        cfg2 = make_config(dsname, suffix, 2, args.seed, args.device)
        setup_environment(cfg2['train'])
        r, model = train_model(cfg2, tag)
        return r, model

    configs = [('greedy_fid', None, True), ('greedy_nofid', None, False),
               ('temp2_fid', 2.0, True), ('temp2_nofid', 2.0, False)]

    with open(out_file, 'a') as fh:
        for tag, T, fid in configs:
            cands = generate(base_model, ds, 5, 20, 0.5, min_len=3, device='cuda',
                             max_batches=0, temperature=T, fidelity=fid,
                             item_freq=freq, pop_penalty=0.0)
            r, succ = retrain(cands, tag)
            cov, tail = model_cov_tail(succ, 20, tail_set)
            out = {'dataset': dsname, 'seed': args.seed, 'tag': tag,
                   'recall@20': float(r['recall@20']), 'ndcg@20': float(r['ndcg@20']),
                   'coverage@20': float(cov), 'tail_recall@20': float(tail),
                   'n_candidates': len(cands)}
            print('RESULT %s' % json.dumps(out), flush=True)
            fh.write(json.dumps(out) + '\n')
            fh.flush()

    print('[DONE]', dsname, 'seed', args.seed, flush=True)


if __name__ == '__main__':
    main()
