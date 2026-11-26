import json
from pathlib import Path
from setup import OUT,read
from core import read_prompt
rows=read(OUT/'support.jsonl');pairs=[]
for i in range(0,len(rows),2):
 pair=rows[i:i+2]
 if i%4==0:pair=list(reversed(pair))
 pairs.append([{'text':r['text'],'pangram_slop':r['slop']} for r in pair])
block='\n\nCALIBRATION REFERENCES\nThe following are labeled reference pairs, NOT inputs to classify. Each pair holds a similar source/topic but differing Pangram outcomes. These are detector labels, not true-authorship claims. Infer the linguistic boundary without matching names, exact words, or topic. For the actual query, classify only its text using the response contract. Reference pairs:\n'+json.dumps(pairs,ensure_ascii=False)
for i in range(3):
 path=OUT/f'seed_{i}.py';backup=OUT/f'seed_{i}_without_support.py'
 if not backup.exists():backup.write_text(path.read_text())
 prompt=read_prompt(backup)+block
 if len(prompt)>15500:raise ValueError('Seed plus references too long')
 path.write_text('# EVOLVE-BLOCK-START\nPROMPT = '+repr(prompt)+'\n# EVOLVE-BLOCK-END\n')
 print(i,len(prompt))
