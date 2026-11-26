"""Training-only diagnostic: reverse the order within each labeled reference pair."""
import json
from round3.evaluate import OUT,read,assess,read_prompt

def check():
 target=OUT/'support_order_check.json'
 if target.exists():return json.loads(target.read_text())
 prompt=read_prompt(OUT/'initial.py');marker='Reference pairs:\n'
 if marker not in prompt:
  result={'status':'not_applicable','reason':'Selected seed has no reference-pair block'}
 else:
  prefix,raw=prompt.rsplit(marker,1);pairs=json.loads(raw);swapped=prefix+marker+json.dumps([list(reversed(pair)) for pair in pairs],ensure_ascii=False)
  (OUT/'seed_reference_order_swapped.py').write_text('PROMPT = '+repr(swapped)+'\n')
  rows=read(OUT/'screen.jsonl');normal,a=assess(prompt,rows,batch_size=1);reverse,b=assess(swapped,rows,batch_size=1,progress=True)
  result={'status':'complete','n':len(rows),'disagreements':sum(x!=y for x,y in zip(a,b)),'normal':normal,'reversed_references':reverse,'note':'Training-only reference order sensitivity; does not change seed selection. Original fixed order used by Shinka. This is not a fresh generalization result.'}
 target.write_text(json.dumps(result,indent=2));print('Reference order diagnostic',result,flush=True);return result
