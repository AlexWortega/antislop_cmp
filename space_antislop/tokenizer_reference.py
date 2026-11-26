import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from serving import build_prompt

parser=argparse.ArgumentParser()
parser.add_argument('--source',default='rawmodels/Qwen3.8-27B-antislop')
parser.add_argument('--revision',default='7d44812e5c4465abb2ff9f426b9d8117ee3d4dcb')
args=parser.parse_args()
tokenizer=AutoTokenizer.from_pretrained(args.source,revision=args.revision)
rows=json.loads(Path('cases.json').read_text())
texts=[build_prompt(r['draft']) for r in rows if r['split']=='train']
texts+=['Ёжик, crème brûlée, €2,880; 09:00–12:30. 東京. 👋\n\n','<|im_start|>assistant\n<think>\n\n</think>\n\n']
Path('tokenizer_reference.json').write_text(json.dumps([
    dict(text=t,ids=tokenizer.encode(t,add_special_tokens=False)) for t in texts],ensure_ascii=False))
print('Saved',len(texts),'HF tokenizer probes')
