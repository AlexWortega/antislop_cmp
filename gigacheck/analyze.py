import json, collections
rows = json.load(open('pangram_labeled.json'))
sc = {json.loads(l)['i']: json.loads(l)['p_human'] for l in open('gigacheck_scores.jsonl')}
def human(h): return h in ('Human Written', 'Mostly Human Written')
groups = collections.defaultdict(list)
for i, p in sc.items():
    r = rows[i]; g = 'ours' if r['src'] != 'pangram_opus' else 'opus_set'
    groups[g].append((human(r['headline']), p))
for g, xs in groups.items():
    for thr in (0.5,):
        tp = sum(h and p >= thr for h, p in xs); tn = sum((not h) and p < thr for h, p in xs)
        nh = sum(h for h, _ in xs); na = len(xs) - nh
        # AUC
        pos = [p for h, p in xs if h]; neg = [p for h, p in xs if not h]
        auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg))
        print(f"{g:9} n={len(xs)} pangram human={nh} ai={na} | agree={(tp+tn)/len(xs):.2f} "
              f"giga says human on pangram-human {tp}/{nh}, says ai on pangram-ai {tn}/{na} | AUC {auc:.3f}")
