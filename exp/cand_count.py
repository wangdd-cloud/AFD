"""Per-user candidate-count distribution for the fidelity gate (reviewer item 14).

For one dataset+seed: train the base SASRec, generate the greedy+gated pool
(m=5 trajectories per user, rank gate k=20, local_prob=0.5, min_len=3, global
dedup keeping the first occurrence), then report how many candidates each user
retains AFTER the gate + min_len + global dedup. The selector budget B=4 binds
exactly for users who retain more than 4 distinct candidates.
"""
import os, sys, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from afd_pipeline import (make_config, train_model, format_real_samples,
                          item_freq_from_real)
from ablation2x2 import generate


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
    _, base_model = train_model(cfg, 'base')

    cfg = make_config(dsname, '', 2, args.seed, args.device)
    setup_environment(cfg['train'])
    ds = prepare_datasets(cfg)[0]
    real_samples = format_real_samples(ds)
    freq = item_freq_from_real(real_samples)
    num_users = ds.num_users  # includes padding index 0

    cands = generate(base_model, ds, 5, 20, 0.5, min_len=3, device='cuda',
                     max_batches=0, temperature=None, fidelity=True,
                     item_freq=freq, pop_penalty=0.0)

    per_user = Counter(c['user_id'] for c in cands)
    hist = Counter(per_user.values())
    n_real_users = num_users - 1           # exclude padding
    n_with = len(per_user)
    n_zero = n_real_users - n_with
    gt_B = sum(v for k, v in hist.items() if k > 4)

    print('DATASET %s seed %d' % (dsname, args.seed), flush=True)
    print('  real users            = %d' % n_real_users, flush=True)
    print('  accepted candidates (post-dedup) = %d' % len(cands), flush=True)
    print('  users with >=1 candidate = %d' % n_with, flush=True)
    print('  users with 0 candidates  = %d' % n_zero, flush=True)
    print('  per-user candidate histogram (n_candidates : n_users):', flush=True)
    for k in sorted(hist):
        print('    %d : %d (%.1f%% of users with candidates)' % (k, hist[k], 100.0 * hist[k] / n_with), flush=True)
    print('  users with >4 candidates (B=4 budget binds) = %d / %d = %.1f%%' % (gt_B, n_with, 100.0 * gt_B / n_with), flush=True)


if __name__ == '__main__':
    main()
