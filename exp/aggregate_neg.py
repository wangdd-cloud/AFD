"""Aggregate the 5-seed negative-result experiments:
1) penalty sweep (sweep_pen.py logs) -> head20 + recall@20 per decoding config
2) adaptive / down-weighting (afd_pipeline.py JSONs) -> AFD_RSIR recall@20 per variant
All mean +/- sample std (ddof=1), matching the paper convention.
"""
import os, json, re, glob
import numpy as np

DS = ['amazon-sport', 'yelp']
SEEDS = list(range(5))

def mean_std(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), a.std(ddof=1)

PEN_CFGS = ['T2.0', 'T1.0', 'pen0.5', 'pen1.0', 'pen1.5']
print('=== PENALTY (sweep_pen) 5-seed ===', flush=True)
for ds in DS:
    head = {c: [] for c in PEN_CFGS}
    rec = {c: [] for c in PEN_CFGS}
    ndcg = {c: [] for c in PEN_CFGS}
    for s in SEEDS:
        log = 'exp/pen_%s_s%d.log' % (ds, s)
        if not os.path.exists(log):
            print('  MISSING %s' % log, flush=True)
            continue
        txt = open(log).read()
        for c in PEN_CFGS:
            m = re.search(r'\[%s\] pool.*?head20=([0-9.]+)' % re.escape(c), txt)
            if m: head[c].append(float(m.group(1)))
            m = re.search(r'\[%s\] recall@20=([0-9.]+) ndcg@20=([0-9.]+)' % re.escape(c), txt)
            if m:
                rec[c].append(float(m.group(1)))
                ndcg[c].append(float(m.group(2)))
    print('--- %s ---' % ds, flush=True)
    for c in PEN_CFGS:
        hm, hs = mean_std(head[c]); rm, rs = mean_std(rec[c]); nm, ns = mean_std(ndcg[c])
        print('  %-7s head20=%.4f+-%.4f  R@20=%.4f+-%.4f  N@20=%.4f+-%.4f  (n=%d)' % (
            c, hm, hs, rm, rs, nm, ns, len(rec[c])), flush=True)

print('', flush=True)
print('=== ADAPTIVE / DOWN-WEIGHTING (afd_pipeline) 5-seed ===', flush=True)
TAGS = ['A', 'B', 'C', 'D']
for ds in DS:
    print('--- %s ---' % ds, flush=True)
    for tag in TAGS:
        rec = []; ndcg = []; cov = []
        missing = False
        for s in SEEDS:
            hits = sorted(glob.glob('exp/abl_%s_%s/results_%s_seed%d_*.json' % (ds, tag, ds, s)))
            if len(hits) > 1:
                raise SystemExit('ambiguous: %d results files for %s %s seed %d;\n  %s' % (
                    len(hits), ds, tag, s, '\n  '.join(hits)))
            if not hits:
                print('    %s: MISSING seed %d' % (tag, s), flush=True)
                missing = True; break
            j = json.load(open(hits[0]))
            if 'AFD_RSIR' not in j:
                print('    %s: no AFD_RSIR in seed %d' % (tag, s), flush=True)
                missing = True; break
            rec.append(j['AFD_RSIR']['recall@20'])
            ndcg.append(j['AFD_RSIR']['ndcg@20'])
            if '_coverage@20' in j['AFD_RSIR']:
                cov.append(j['AFD_RSIR']['_coverage@20'])
        if not missing:
            rm, rs = mean_std(rec); nm, ns = mean_std(ndcg)
            line = '  %s: AFD_RSIR R@20=%.4f+-%.4f  N@20=%.4f+-%.4f' % (tag, rm, rs, nm, ns)
            if cov:
                cm, cs = mean_std(cov)
                line += '  cov=%.4f+-%.4f' % (cm, cs)
            print(line, flush=True)
