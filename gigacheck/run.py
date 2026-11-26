"""Score Pangram-labelled texts with GigaCheck-Classifier-Multi locally (MPS) and compare with real Pangram."""
import json, sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from transformers import AutoModel
OUT = Path('gigacheck_scores.jsonl')
done = {json.loads(l)['i'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
rows = json.load(open('pangram_labeled.json'))
# our model's outputs first: they matter most
order = sorted(range(len(rows)), key=lambda i: rows[i]['src'] == 'pangram_opus')
m = AutoModel.from_pretrained('model', trust_remote_code=True, torch_dtype=torch.bfloat16); m.to('mps'); m.eval()
t0 = time.time(); n = 0
with OUT.open('a') as f, torch.no_grad():
    for i in order:
        if i in done: continue
        o = m([rows[i]['text'].replace('\n', ' ')])
        p_h = float(o.classification_head_probs[0][1])
        f.write(json.dumps({'i': i, 'p_human': p_h}) + '\n'); f.flush(); n += 1
        if n % 50 == 0: print(n, f'{(time.time()-t0)/n:.2f}s/text', flush=True)
print('done', n)
