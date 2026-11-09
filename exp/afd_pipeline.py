"""
AFD-RSIR experiment pipeline.

Runs the recursive self-improvement comparison on the official RSIR codebase:
  base SASRec -> RSIR generation (fidelity gate) -> diversity-aware selection -> retrain.

Methods: SASRec, RSIR (keep-all), RSIR+random, RSIR+novelty, RSIR+MMR, RSIR+DPP, AFD-RSIR.

Usage (single process per dataset/seed/device):
  python afd_pipeline.py --dataset amazon-sport --seed 0 --device 0 --outdir afd_results [--quick]
"""
import os, sys, json, time, argparse, math, pickle, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from utils import *          # noqa: E402  (get_default_parser, load_config, seed_everything, set_device, prepare_datasets, prepare_model, get_logger)
import wandb                 # noqa: E402

os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'


# --------------------------------------------------------------------------- config
def make_config(dataset, trainfile, num_layer, seed, device, quick=False, epochs=None, synthetic_weight=1.0, real_num_users=None):
    config = {
        'model': 'SASRec', 'dataset': dataset, 'trainfile': trainfile,
        'output_trainfile': '_1th', 'device': str(device), 'eval_only': True,
        'theta_r_path': '', 'neg_num': 256, 'num_layer': num_layer,
        'loss_fn_name': 'sampled_softmax', 'mode': 'recommendation',
    }
    config = load_config(config)
    config['train']['seed'] = seed
    config['train']['device'] = int(device)
    config['num_layer'] = num_layer
    config['synthetic_weight'] = synthetic_weight
    if real_num_users is not None:
        config['real_num_users'] = real_num_users
    if quick:
        config['train']['epochs'] = 2
        config['train']['early_stop_patience'] = 2
    if epochs is not None:
        config['train']['epochs'] = epochs
    return config


# --------------------------------------------------------------------------- training
def train_model(config, tag):
    dataset_list = prepare_datasets(config)
    model = prepare_model(config, dataset_list)
    ts = time.strftime('%Y%m%d-%H%M%S-%f')
    log_path = f"{config['model']['model']}/{config['data']['dataset']}/{tag}-{ts}.log"
    get_logger(log_path)
    wandb.init(project='AFD', name=f"{tag}-{config['data']['dataset']}", config=config, mode='disabled')
    model.fit()
    result = model.evaluate()
    wandb.finish()
    return result, model


# --------------------------------------------------------------------------- generation (with user mapping)
@torch.no_grad()
def generate_candidates(model, train_dataset, m, k, local_prob, min_len, device, max_batches=0, temperature=2.0, item_freq=None, pop_penalty=0.0, fidelity=True):
    """Generate fidelity-passing trajectories, preserving the originating user.

    Decoding strategy is explicit: ``temperature=None`` is greedy argmax (which
    concentrates the pool), a float ``temperature`` is temperature-scaled sampling.
    ``fidelity=True`` applies the RSIR rank gate (truncate when no remaining history
    item ranks <= k); ``fidelity=False`` disables it. When ``pop_penalty>0`` a
    popularity logit penalty (``-lambda * log(freq+1)``, normalized) is subtracted
    from the decoding scores only (the RSIR fidelity predicate still uses raw scores).

    Returns list of dicts: {'user_id', 'hist_len', 'real_seq', 'gen_seq'}.
    """
    model.eval()
    max_seq_len = model.max_seq_len
    num_items = model.num_items
    batch_size = train_dataset.config['train']['batch_size']
    loader = train_dataset.get_loader(batch_size=batch_size, shuffle=False)

    pop_bias = torch.zeros(num_items, device=device)
    if item_freq is not None and pop_penalty > 0:
        for it, c in item_freq.items():
            if 0 < it < num_items:
                pop_bias[it] = math.log(c + 1)
        mx = pop_bias.max()
        if mx > 0:
            pop_bias = pop_bias / mx   # normalize to [0, 1]

    candidates = []
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        batch = {kk: vv.to(device) for kk, vv in batch.items()}
        original_seqs_tensor = batch['in_item_id']
        original_seqs_list = [s[s != 0].tolist() for s in original_seqs_tensor]
        user_ids = batch['user_id'].tolist()
        seq_lens = batch['seqlen'].tolist()

        B = original_seqs_tensor.size(0)
        eff = B * m
        generated = torch.zeros(eff, max_seq_len, dtype=torch.long, device=device)
        generated[:, 0] = original_seqs_tensor[:, 0].repeat_interleave(m)
        cur_len = torch.ones(eff, dtype=torch.long, device=device)
        is_active = torch.ones(eff, dtype=torch.bool, device=device)
        orig_expanded = [s for s in original_seqs_list for _ in range(m)]

        for step in range(1, max_seq_len):
            if not is_active.any():
                break
            active = torch.where(is_active)[0]
            in_seqs = generated[active]
            in_lens = cur_len[active]
            fake_batch = {'in_item_id': in_seqs, 'seqlen': in_lens}
            q_all = model.forward(fake_batch, need_pooling=False)          # [nA, L, D]
            pred = q_all[torch.arange(len(active)), in_lens - 1]
            all_scores = pred @ model.item_embedding.weight.T               # [nA, num_items]

            next_items = torch.zeros(len(active), dtype=torch.long, device=device)
            deact = []
            for i, fb in enumerate(active.tolist()):
                orig = orig_expanded[fb]
                gen = generated[fb, :cur_len[fb]].tolist()
                remaining = set(orig) - set(gen)
                if not remaining:
                    deact.append(i); continue
                scores_i = all_scores[i]
                if fidelity:
                    rem = torch.LongTensor(list(remaining)).to(device)
                    ranks = (scores_i > scores_i[rem].unsqueeze(1)).sum(1) + 1
                    if not (ranks <= k).any():
                        deact.append(i); continue
                masked = scores_i.clone()
                masked[torch.LongTensor(gen).to(device)] = -torch.inf
                if pop_penalty > 0:
                    masked = masked - pop_penalty * pop_bias   # popularity logit penalty (decoding only)
                if random.random() < local_prob:
                    pool = torch.LongTensor(list(set(orig))).to(device)
                else:
                    pool = torch.arange(1, num_items, device=device)
                cand_scores = masked[pool]
                if not torch.isfinite(cand_scores).any():
                    deact.append(i); continue
                if temperature is None:
                    next_items[i] = pool[torch.argmax(cand_scores)]          # greedy
                else:
                    probs = torch.softmax(cand_scores / temperature, dim=0)
                    next_items[i] = pool[torch.multinomial(probs, 1).item()]

            if deact:
                keep_mask = torch.ones(len(active), dtype=torch.bool, device=device)
                keep_mask[torch.LongTensor(deact).to(device)] = False
                upd = active[keep_mask]
                is_active[active[~keep_mask]] = False
                valid_next = next_items[keep_mask]
            else:
                upd = active
                valid_next = next_items
            generated[upd, cur_len[upd]] = valid_next
            cur_len[upd] += 1

        for i in range(eff):
            L = cur_len[i].item()
            if L >= min_len:
                orig_u = i // m
                candidates.append({
                    'user_id': user_ids[orig_u],
                    'hist_len': seq_lens[orig_u],
                    'real_seq': original_seqs_list[orig_u],
                    'gen_seq': generated[i, :L].tolist(),
                })

    # dedup by generated sequence, keep first occurrence
    seen = {}
    for c in candidates:
        key = tuple(c['gen_seq'])
        seen.setdefault(key, c)
    return list(seen.values())


# --------------------------------------------------------------------------- formatting helpers
def format_real_samples(train_dataset):
    max_seq_len = train_dataset.max_seq_len
    fuid = train_dataset.fuid
    fiid = train_dataset.fiid
    loader = train_dataset.get_loader(batch_size=train_dataset.config['train']['batch_size'], shuffle=False)
    out = []
    for batch in loader:
        B = batch[fuid].size(0)
        for i in range(B):
            uid = batch[fuid][i].item()
            hist = batch['in_' + fiid][i].tolist()
            tgt = batch[fiid][i].tolist()
            hlen = batch['seqlen'][i].item()
            if hlen <= 0:
                continue
            lab = [1] * hlen + [0] * (max_seq_len - hlen)
            dom = [1] * max_seq_len
            out.append([uid, hist, tgt, hlen, lab, dom])
    return out


def format_synthetic(seqs, num_users, max_seq_len):
    out = []
    nid = num_users
    for seq in seqs:
        hl = len(seq) - 1
        if hl < 1:
            continue
        hist, tgt = seq[:-1], seq[1:]
        ph = hist + [0] * (max_seq_len - hl)
        pt = tgt + [0] * (max_seq_len - hl)
        lab = [1] * hl + [0] * (max_seq_len - hl)
        dom = [1] * max_seq_len
        out.append([nid, ph, pt, hl, lab, dom])
        nid += 1
    return out


# --------------------------------------------------------------------------- selection
def l2norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def build_item_emb(model):
    E = model.item_embedding.weight.detach().cpu().numpy()          # [num_items, D]
    return np.stack([l2norm(E[i]) for i in range(E.shape[0])], axis=0)


def item_freq_from_real(real_samples):
    freq = {}
    for s in real_samples:
        for it in s[2]:
            if it != 0:
                freq[it] = freq.get(it, 0) + 1
    return freq


def seq_rep(seq, E):
    ids = [i for i in seq if 0 < i < E.shape[0]]
    if not ids:
        return np.zeros(E.shape[1], dtype=np.float64)
    return l2norm(E[ids].mean(axis=0))


def cos(a, b):
    return float(np.dot(a, b))


def select_per_user(seqs, q_u, E, freq, selector, B, alpha, beta, mmr_lambda, n_items):
    """Return indices of selected sequences for one user."""
    reps = [seq_rep(s, E) for s in seqs]
    aff = [max(0.0, cos(r, q_u)) for r in reps]

    # tail weights
    w = np.zeros(n_items + 1, dtype=np.float64)
    for it in range(1, n_items + 1):
        w[it] = 1.0 / math.sqrt(freq.get(it, 0) + 1.0)
    wmax = w.max() if w.max() > 0 else 1.0

    def tail_norm(seq, selected_seqs):
        covered = set()
        for s in selected_seqs:
            covered.update(x for x in s if x != 0)
        gain = 0.0
        for x in seq:
            if x != 0 and x not in covered:
                gain += w[x]
        return gain / wmax

    def maxsim(rep, sel_reps):
        return max([cos(rep, r) for r in sel_reps], default=0.0)

    sel_idx = []
    sel_reps = []
    sel_seqs = []
    remaining = list(range(len(seqs)))

    while remaining and len(sel_idx) < B:
        if selector == 'dpp':
            # greedy MAP inference: argmax det(L_{A u {c}})
            def dpp_score(i):
                sub = sel_idx + [i]
                kk = len(sub)
                L = np.empty((kk, kk))
                q = [max(0.05, aff[j]) for j in sub]
                for a in range(kk):
                    for b in range(kk):
                        L[a, b] = q[a] * q[b] * cos(reps[sub[a]], reps[sub[b]])
                sign, val = np.linalg.slogdet(L)
                return val if sign > 0 else -1e9
            best = max(remaining, key=dpp_score)
        elif selector == 'mmr':
            best = max(remaining, key=lambda i: mmr_lambda * aff[i] - (1 - mmr_lambda) * maxsim(reps[i], sel_reps))
        elif selector == 'novelty':
            best = max(remaining, key=lambda i: 1.0 - maxsim(reps[i], sel_reps))
        elif selector == 'afd':
            def score(i):
                nu = 1.0 - maxsim(reps[i], sel_reps)
                tn = tail_norm(seqs[i], sel_seqs)
                return alpha * nu + beta * tn + (1 - alpha - beta) * aff[i]
            best = max(remaining, key=score)
        else:  # random
            best = random.choice(remaining)

        sel_idx.append(best)
        sel_reps.append(reps[best])
        sel_seqs.append(seqs[best])
        remaining.remove(best)

    return sel_idx


def select_all(candidates, E, freq, selector, B, alpha, beta, mmr_lambda, n_items, head_set=None, alpha_min=None, alpha_max=None):
    by_user = {}
    for c in candidates:
        by_user.setdefault(c['user_id'], []).append(c)
    selected_seqs = []
    for uid, cs in by_user.items():
        seqs = [c['gen_seq'] for c in cs]
        q_u = seq_rep(cs[0]['real_seq'], E)
        a_u = alpha
        if head_set is not None and alpha_min is not None and alpha_max is not None:
            # adaptive alpha_u: diversity pressure proportional to the user's candidate
            # head-concentration (concentrated users get more diversity pressure).
            items = [it for c in cs for it in c['gen_seq'] if it != 0]
            conc = sum(1 for it in items if it in head_set) / len(items) if items else 0.5
            a_u = alpha_min + (alpha_max - alpha_min) * conc
        idx = select_per_user(seqs, q_u, E, freq, selector, B, a_u, beta, mmr_lambda, n_items)
        selected_seqs.extend(seqs[i] for i in idx)
    return selected_seqs


# --------------------------------------------------------------------------- model-level diversity metrics
@torch.no_grad()
def model_cov_tail(model, K, tail_set):
    """Recommendation-list coverage@K and tail Recall@K of a trained model on the test set."""
    model.eval()
    test_data = model.dataset_list[-1]
    loader = test_data.get_loader()
    num_items = model.num_items
    recommended = set()
    tail_hits = 0
    tail_total = 0
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        _, topk_items = model.topk(batch, K, batch['user_hist'])          # [bs, K]
        topk_np = topk_items.cpu().numpy()
        recommended.update(int(x) for x in topk_np.flatten() if 1 <= x < num_items)
        label = batch[model.fiid].view(-1, 1)                              # [bs, 1]
        is_tail = torch.tensor([int(l.item()) in tail_set for l in label.view(-1)],
                               device=model.device).view(-1)
        hit = (label == topk_items).any(dim=1)                             # [bs]
        tail_hits += int((hit & is_tail).sum())
        tail_total += int(is_tail.sum())
    coverage = len(recommended) / (num_items - 1)
    tail_recall = tail_hits / tail_total if tail_total > 0 else 0.0
    return coverage, tail_recall


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--outdir', default='afd_results')
    ap.add_argument('--num_layer', type=int, default=2)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--k', type=int, default=20)          # fidelity rank threshold
    ap.add_argument('--m', type=int, default=5)            # trajectories per user
    ap.add_argument('--local_prob', type=float, default=0.5)
    ap.add_argument('--B', type=int, default=4)            # per-user selection budget
    ap.add_argument('--alpha', type=float, default=0.3)    # novelty weight
    ap.add_argument('--beta', type=float, default=0.4)     # tail-coverage weight (diversity-first)
    ap.add_argument('--mmr_lambda', type=float, default=0.7)
    ap.add_argument('--max_batches', type=int, default=0)   # 0 = all; limits generation batches for smoke tests
    ap.add_argument('--temperature', type=float, default=2.0)
    ap.add_argument('--greedy', action='store_true', help='greedy argmax decoding (temperature=None)')
    ap.add_argument('--pop_penalty', type=float, default=0.0)   # popularity logit penalty strength (0 = off; matches the greedy baseline)
    ap.add_argument('--adaptive', action='store_true')          # per-user adaptive alpha_u
    ap.add_argument('--alpha_min', type=float, default=0.1)
    ap.add_argument('--alpha_max', type=float, default=0.5)
    ap.add_argument('--synthetic_weight', type=float, default=1.0)  # weight on synthetic samples (1.0 = off)
    ap.add_argument('--methods', default='', help='comma-separated subset of methods (empty = all)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # variant signature: any change to a generation/selection/weighting parameter
    # produces a distinct results/pool/intermediate file, so stale outputs are never reused
    sig_parts = [
        'greedy' if args.greedy else f'T{args.temperature}',
        f'k{args.k}', f'm{args.m}', f'lp{args.local_prob}', f'pp{args.pop_penalty}', f'sw{args.synthetic_weight}',
        f'nl{args.num_layer}', f'mb{args.max_batches}',
    ]
    if args.quick:
        sig_parts.append('quick')
    if args.adaptive:
        sig_parts.append(f'ada{args.alpha_min}-{args.alpha_max}')
    sig_parts.append(f'B{args.B}_a{args.alpha}_b{args.beta}_mmr{args.mmr_lambda}')
    sig = '_'.join(sig_parts)
    results_file = os.path.join(args.outdir, f"results_{args.dataset}_seed{args.seed}_{sig}.json")
    results = {} if not os.path.exists(results_file) else json.load(open(results_file))

    dataset = args.dataset
    device = args.device
    domain = {'amazon-sport': 'sport', 'yelp': 'yelp'}[dataset]

    def save_results():
        tmp = results_file + f'.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, results_file)

    pool_path = os.path.join(args.outdir, f"pool_{dataset}_seed{args.seed}_{sig}.pkl")

    # ---- Phase 1: base SASRec + candidate pool ----
    if not os.path.exists(pool_path):
        cfg = make_config(dataset, '', args.num_layer, args.seed, device, quick=args.quick)
        setup_environment(cfg['train'])
        r, model = train_model(cfg, 'base')
        results['SASRec'] = {k: float(v) for k, v in r.items()}
        print('[base]', json.dumps(results['SASRec']), flush=True)

        # ---- Generate candidate pool ----
        cfg = make_config(dataset, '', args.num_layer, args.seed, device, quick=args.quick)
        setup_environment(cfg['train'])
        ds = prepare_datasets(cfg)[0]
        real_samples = format_real_samples(ds)
        E = build_item_emb(model)
        freq = item_freq_from_real(real_samples)
        rf0 = np.array([freq.get(i, 0) for i in range(1, ds.num_items)], dtype=np.float64)
        _ho = np.argsort(-rf0, kind='stable')
        _nh = int(0.2 * (ds.num_items - 1))
        _ts0 = {int(_ho[j]) + 1 for j in range(_nh, ds.num_items - 1)}
        cov0, tail0 = model_cov_tail(model, 20, _ts0)
        results['SASRec']['_coverage@20'] = cov0
        results['SASRec']['_tail_recall@20'] = tail0
        cands = generate_candidates(model, ds, args.m, args.k, args.local_prob, min_len=3, device='cuda', max_batches=args.max_batches, temperature=(None if args.greedy else args.temperature), item_freq=freq, pop_penalty=args.pop_penalty)
        acceptance = len(cands) / (args.m * ds.num_users) if ds.num_users > 0 else 0.0
        pool = {
            'num_users': ds.num_users, 'num_items': ds.num_items, 'max_seq_len': ds.max_seq_len,
            'real_samples': real_samples, 'candidates': cands,
            'E': E.tolist(), 'freq': {str(k): v for k, v in freq.items()},
            'acceptance': acceptance,
            'base_results': results['SASRec'],
        }
        tmp = pool_path + f'.{os.getpid()}.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(pool, f)
        os.replace(tmp, pool_path)
        save_results()
        print(f'[pool] {len(cands)} candidates', flush=True)

    # ---- Phase 2: selectors ----
    pool = pickle.load(open(pool_path, 'rb'))
    if 'SASRec' not in results and 'base_results' in pool:
        results['SASRec'] = pool['base_results']
        save_results()
    real_samples = pool['real_samples']
    num_users = pool['num_users']
    max_seq_len = pool['max_seq_len']
    n_items = pool['num_items'] - 1     # real catalog size (num_items includes padding 0)
    E = np.array(pool['E'])
    freq = {int(k): v for k, v in pool['freq'].items()}
    candidates = pool['candidates']

    # head = top 20% by real-data target frequency (desc, stable); tail = complement
    rf_full = np.array([freq.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
    head_order = np.argsort(-rf_full, kind='stable')
    n_head = int(0.2 * n_items)
    head_set = {int(head_order[j]) + 1 for j in range(n_head)}
    tail_set = {int(head_order[j]) + 1 for j in range(n_head, n_items)}

    ALL_METHODS = ['RSIR', 'RSIR_random', 'RSIR_novelty', 'RSIR_MMR', 'RSIR_DPP', 'AFD_RSIR']
    methods = list(ALL_METHODS)
    if args.methods:
        methods = [m for m in methods if m in args.methods.split(',')]
    selector_map = {'RSIR': 'all', 'RSIR_random': 'random', 'RSIR_novelty': 'novelty',
                    'RSIR_MMR': 'mmr', 'RSIR_DPP': 'dpp', 'AFD_RSIR': 'afd'}

    for method in methods:
        if method in results:
            print(f'[skip] {method} already done', flush=True)
            continue
        sel = selector_map[method]
        # Deterministic per-method RNG keyed on the fixed method table, so the result
        # does not depend on the --methods subset nor on whether the pool was cached.
        random.seed(args.seed * 1000 + ALL_METHODS.index(method))
        if sel == 'all':
            selected = [c['gen_seq'] for c in candidates]
        else:
            hs = head_set if (args.adaptive and sel == 'afd') else None
            amin = args.alpha_min if args.adaptive else None
            amax = args.alpha_max if args.adaptive else None
            selected = select_all(candidates, E, freq, sel, args.B, args.alpha, args.beta, args.mmr_lambda, n_items,
                                  head_set=hs, alpha_min=amin, alpha_max=amax)

        synth = format_synthetic(selected, num_users, max_seq_len)
        train_data = real_samples + synth
        suffix = f"_{method}_{args.seed}_{sig}"
        data_dir = os.path.join('dataset', dataset, domain)
        train_path = os.path.join(data_dir, f"train{suffix}.pth")
        tmp_path = train_path + f'.{os.getpid()}.tmp'
        torch.save(train_data, tmp_path)
        os.replace(tmp_path, train_path)

        cfg = make_config(dataset, suffix, args.num_layer, args.seed, device, quick=args.quick,
                          synthetic_weight=args.synthetic_weight, real_num_users=num_users)
        setup_environment(cfg['train'])
        r, model = train_model(cfg, method.lower())
        cov, tail_recall = model_cov_tail(model, 20, tail_set)
        results[method] = {k: float(v) for k, v in r.items()}
        results[method]['_n_synthetic'] = len(synth)
        results[method]['_coverage@20'] = cov
        results[method]['_tail_recall@20'] = tail_recall
        save_results()
        print(f'[{method}]', json.dumps(results[method]), flush=True)

    print('[DONE]', args.dataset, 'seed', args.seed, flush=True)


if __name__ == '__main__':
    main()
