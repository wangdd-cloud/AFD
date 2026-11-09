"""2x2 fidelity ablation + temperature-sweep Pareto (item-4 / reviewer P0-4, P1).

For one dataset+seed: train the base SASRec once, then for each decoding config
(mode x fidelity) generate a candidate pool, measure pool diversity, retrain a
keep-all successor on it, and record accuracy.

Cells (mode = greedy argmax vs temperature sampling; fidelity = RSIR rank-gate
on/off):
  greedy  + fidelity   -> reproduces paper's "RSIR"  (head20 ~0.85)
  greedy  + no fidelity
  temp2.0 + fidelity   -> reproduces paper's "T=2.0" (head20 ~0.58)
  temp2.0 + no fidelity

Plus a temperature sweep (fidelity on) T in {1.0,1.2,1.4,1.6,1.8,2.0} to build the
accuracy-vs-diversity Pareto curve. Results append to a per-dataset JSONL.
"""
import os, sys, argparse, json, math, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from afd_pipeline import (make_config, train_model, format_real_samples,
                          item_freq_from_real, format_synthetic)


def gini(freq):
    x = np.asarray(sorted(freq), dtype=np.float64)
    if x.sum() == 0:
        return float('nan')
    x = x / x.sum()
    n = len(x)
    idx = np.arange(1, n + 1)
    return (np.sum((2 * idx - n - 1) * x)) / n


@torch.no_grad()
def generate(model, train_dataset, m, k, local_prob, min_len, device,
             max_batches=0, temperature=None, fidelity=True, item_freq=None,
             pop_penalty=0.0):
    """temperature=None => greedy argmax; fidelity=False => skip the rank gate."""
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
            pop_bias = pop_bias / mx

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
            q_all = model.forward(fake_batch, need_pooling=False)
            pred = q_all[torch.arange(len(active)), in_lens - 1]
            all_scores = pred @ model.item_embedding.weight.T

            next_items = torch.zeros(len(active), dtype=torch.long, device=device)
            deact = []
            for i, fb in enumerate(active.tolist()):
                orig = orig_expanded[fb]
                gen = generated[fb, :cur_len[fb]].tolist()
                remaining = set(orig) - set(gen)
                if not remaining:
                    deact.append(i); continue
                scores_i = all_scores[i]
                rem = torch.LongTensor(list(remaining)).to(device)
                if fidelity:
                    ranks = (scores_i > scores_i[rem].unsqueeze(1)).sum(1) + 1
                    if not (ranks <= k).any():
                        deact.append(i); continue
                masked = scores_i.clone()
                masked[torch.LongTensor(gen).to(device)] = -torch.inf
                if pop_penalty > 0:
                    masked = masked - pop_penalty * pop_bias
                if random.random() < local_prob:
                    pool = torch.LongTensor(list(set(orig))).to(device)
                else:
                    pool = torch.arange(1, num_items, device=device)
                cand_scores = masked[pool]
                if not torch.isfinite(cand_scores).any():
                    deact.append(i); continue
                if temperature is None:
                    next_items[i] = pool[torch.argmax(cand_scores)]
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

    seen = {}
    for c in candidates:
        seen.setdefault(tuple(c['gen_seq']), c)
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    dsname = args.dataset
    domain = {'amazon-sport': 'sport', 'yelp': 'yelp'}[dsname]
    out_file = f'exp/ablation2x2_{dsname}.jsonl'

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
        cc_full = {}
        for c in cands:
            seq = c['gen_seq']
            for it in seq[1:]:              # generated suffix only (exclude copied first item)
                if it != 0:
                    cc[it] = cc.get(it, 0) + 1
            for it in seq:                   # full seq, for cross-check with run_temp5
                if it != 0:
                    cc_full[it] = cc_full.get(it, 0) + 1
        f = np.array([cc.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        ff = np.array([cc_full.get(i, 0) for i in range(1, n_items + 1)], dtype=np.float64)
        return {'gini': float(gini(f)), 'head20': float(share(f, 0.2)),
                'coverage': float((f > 0).sum() / n_items), 'n_candidates': len(cands),
                'gini_full': float(gini(ff)), 'head20_full': float(share(ff, 0.2)),
                'coverage_full': float((ff > 0).sum() / n_items)}

    def retrain(cands, tag):
        synth = format_synthetic([c['gen_seq'] for c in cands], num_users, max_seq_len)
        train_data = real_samples + synth
        suffix = f'_{tag}_{args.seed}'
        data_dir = os.path.join('dataset', dsname, domain)
        train_path = os.path.join(data_dir, f'train{suffix}.pth')
        tmp_path = train_path + f'.{os.getpid()}.tmp'
        torch.save(train_data, tmp_path)
        os.replace(tmp_path, train_path)
        cfg2 = make_config(dsname, suffix, 2, args.seed, args.device)
        setup_environment(cfg2['train'])
        r, _ = train_model(cfg2, tag)
        return r

    configs = [
        # 2x2 fidelity ablation (greedy = argmax, temperature=None)
        ('greedy_fid',   None, True),    # reproduce paper's RSIR (expect gini ~0.94)
        ('greedy_nofid', None, False),
        ('temp2_fid',    2.0,  True),    # reproduce paper's T=2.0 (expect gini ~0.59)
        ('temp2_nofid',  2.0,  False),
        # temperature sweep (fidelity on) -> Pareto curve
        ('T1.0', 1.0, True),
        ('T1.2', 1.2, True),
        ('T1.4', 1.4, True),
        ('T1.6', 1.6, True),
        ('T1.8', 1.8, True),
    ]

    with open(out_file, 'a') as fh:
        for tag, T, fid in configs:
            cands = generate(model, ds, 5, 20, 0.5, min_len=3, device='cuda',
                             max_batches=0, temperature=T, fidelity=fid,
                             item_freq=freq, pop_penalty=0.0)
            div = diversity(cands)
            r = retrain(cands, tag)
            out = {'dataset': dsname, 'seed': args.seed, 'tag': tag,
                   'temperature': T, 'fidelity': fid,
                   'recall@20': float(r['recall@20']), 'ndcg@20': float(r['ndcg@20']),
                   **div}
            print('RESULT %s' % json.dumps(out), flush=True)
            fh.write(json.dumps(out) + '\n')
            fh.flush()

    print('[DONE]', dsname, 'seed', args.seed, flush=True)


if __name__ == '__main__':
    main()
