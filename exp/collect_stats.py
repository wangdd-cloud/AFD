"""Collect mean+-std + ndcg@20 for the ESWA paper tables, from the server's existing
results (afd_results JSONs, temp5 jsonl, aggregated_results.json). Also dump the
state of the abl_* / sweep_* artifacts to see what new experiments already exist."""
import json, os, glob
import numpy as np

DS = {'amazon-sport': 'sport', 'yelp': 'yelp'}
SEEDS = [0, 1, 2, 3, 4]

def ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return float('nan'), float('nan')
    if len(v) == 1:
        return float(v[0]), 0.0
    return float(np.mean(v)), float(np.std(v, ddof=1))

for ds, dom in DS.items():
    print('===== %s =====' % ds)
    # base + greedy accuracy (recall@20 + ndcg@20) per seed
    acc = {'SASRec': {'r20': [], 'n20': []}, 'RSIR': {'r20': [], 'n20': []}}
    for s in SEEDS:
        hits = sorted(glob.glob('exp/afd_results/results_%s_seed%d_*.json' % (ds, s)))
        if len(hits) > 1:
            raise SystemExit('ambiguous: %d results files for %s seed %d;\n  %s' % (
                len(hits), ds, s, '\n  '.join(hits)))
        if not hits:
            continue
        d = json.load(open(hits[0]))
        for m in ['SASRec', 'RSIR']:
            if m in d:
                acc[m]['r20'].append(d[m].get('recall@20'))
                acc[m]['n20'].append(d[m].get('ndcg@20'))
    for m in ['SASRec', 'RSIR']:
        rmu, rsd = ms(acc[m]['r20']); nmu, nsd = ms(acc[m]['n20'])
        print('%-6s recall@20 %.5f+-%.5f | ndcg@20 %.5f+-%.5f' % (m, rmu, rsd, nmu, nsd))

    # temperature (temp5 jsonl)
    tmp = {'r20': [], 'n20': [], 'gini': [], 'head20': [], 'cov': []}
    for line in open('exp/temp5_%s.jsonl' % ds):
        o = json.loads(line)
        tmp['r20'].append(o['recall@20']); tmp['n20'].append(o['ndcg@20'])
        tmp['gini'].append(o['gini']); tmp['head20'].append(o['head20']); tmp['cov'].append(o['coverage'])
    print('TEMP5 recall@20 %.5f+-%.5f | ndcg@20 %.5f+-%.5f | gini %.4f+-%.4f | head20 %.4f+-%.4f | cov %.4f+-%.4f' % (
        *ms(tmp['r20']), *ms(tmp['n20']), *ms(tmp['gini']), *ms(tmp['head20']), *ms(tmp['cov'])))

    # diversity mean+-std from aggregated_results.json (greedy + selectors)
    agg = json.load(open('exp/aggregated_results.json'))
    div = agg['diversity'][ds]
    for m in ['RSIR', 'RSIR_random', 'RSIR_novelty', 'RSIR_MMR', 'RSIR_DPP']:
        if m not in div:
            print('%-14s (missing)' % m); continue
        mm = div[m]
        print('%-14s gini %.4f+-%.4f | head20 %.4f+-%.4f | cov %.2f+-%.2f' % (
            m, mm['gini'][0], mm['gini'][1], mm['head20'][0], mm['head20'][1],
            mm['coverage'][0], mm['coverage'][1]))

print('\n===== item-4 artifacts =====')
print('--- run_abl.sh ---')
print(open('exp/run_abl.sh').read() if os.path.exists('exp/run_abl.sh') else '(no run_abl.sh)')
for f in ['exp/sweep_sport.log', 'exp/sweep_yelp.log', 'exp/sweep_pen_sport.log', 'exp/sweep_pen_yelp.log']:
    if os.path.exists(f):
        print('\n--- %s (tail) ---' % f)
        lines = open(f).read().strip().splitlines()
        print('\n'.join(lines[-12:]))
