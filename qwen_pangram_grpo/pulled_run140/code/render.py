"""Render task prompts through the policy's chat template (no system prompt, like the prod baseline)."""
import argparse, json
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--src', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--thinking', action='store_true')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
with open(a.out, 'w') as f:
    for line in open(a.src):
        r = json.loads(line)
        r['prompt'] = tok.apply_chat_template([{'role': 'user', 'content': r['question']}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=a.thinking)
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('rendered', a.out, repr(r['prompt'][-60:]))
