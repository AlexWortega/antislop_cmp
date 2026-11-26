import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SCAFFOLDS={
 'plain':'Answer the task directly in natural English. Keep all supplied facts accurate.',
 'concise':'You are a concise assistant. Give a useful answer without filler or generic introductions. Preserve the facts.',
 'editor':'You are a careful editor. Write clear, natural prose that satisfies the task. Do not invent details.',
 'check_then_answer':'Check the facts and requested format privately, then give only the final answer. Avoid unnecessary framing.'}
TRAIN=[
 ('library','Write a 45–80 word notice: the Oak Street library closes at 4 pm this Friday for electrical work. It reopens Saturday at 10 am. Online renewals remain available. Do not invent contact details.'),
 ('refund','Write a 50–90 word customer reply: order K17 arrived with a cracked mug. A replacement was dispatched today, delivery takes 3–5 business days, and no return is required. Be helpful without promotional language.'),
 ('battery','In 60–100 words explain why a phone battery may drain faster in cold weather. Distinguish temporary performance effects from permanent damage. Avoid unsupported numerical claims.'),
 ('meeting','Draft a 45–80 word email: move the project review from Tuesday 2 pm to Wednesday 11 am because the supplier report arrives Tuesday evening. The meeting link stays the same. Ask colleagues to flag conflicts.'),
 ('recipe','Explain in 60–100 words why bread can be dense if it has not risen long enough. Give two practical checks without promising a fixed proofing time.'),
 ('project','Summarize in 50–80 words: a pilot involved 24 volunteers over two weeks; 18 completed every session; six missed at least one; there was no control group. Explain what cannot be concluded about effectiveness.'),
 ('rent','Write a 40–75 word tenant notice: water will be off Monday from 9 am to noon for valve repairs in Building B only. Residents should fill drinking-water containers beforehand. Do not add guarantees.'),
 ('budget','Explain in 45–80 words: a club has $240, spends $75 on supplies and $45 on room hire, then receives $30 in donations. State the remaining balance and show the calculation briefly.')]
VAL=[
 ('museum','Write a 45–80 word visitor notice: the east entrance of the Riverside museum is closed Thursday for repairs. Use the north entrance. Opening hours and ticket prices are unchanged. Do not invent accessibility details.'),
 ('shipping','Write a 50–90 word customer reply: parcel M42 was sent to the wrong sorting hub. Tracking was corrected today; the carrier estimates two extra business days. No refund or delivery guarantee has been approved.'),
 ('condensation','Explain in 60–100 words why water droplets form on the outside of a cold glass in a humid room. Make clear where the water comes from.'),
 ('survey','Summarize in 50–80 words: 80 of 200 invited residents answered a park survey; 56 respondents wanted more benches. State the share among respondents and explain why it need not represent all residents.')]
def rows(topics):
 out=[]
 for topic,question in topics:
  for scaffold,system in SCAFFOLDS.items():
   out.append({'data_source':'opus_antislop_pilot','prompt':[{'role':'system','content':system},{'role':'user','content':question}],'ability':'writing','reward_model':{'style':'rule','ground_truth':''},'extra_info':{'question':question,'topic':topic,'scaffold':scaffold,'id':topic+'__'+scaffold}})
 return out
if __name__=='__main__':
 import pandas as pd
 train,val=rows(TRAIN),rows(VAL)
 assert not {r['extra_info']['topic'] for r in train}&{r['extra_info']['topic'] for r in val}
 (ROOT/'data').mkdir(exist_ok=True)
 for name,data in [('train',train),('val',val)]:
  pd.DataFrame(data).to_parquet(ROOT/'data'/f'{name}.parquet',index=False)
  (ROOT/'data'/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
 print({'train':len(train),'val':len(val),'scaffolds':list(SCAFFOLDS),'heldout_topics':[x[0] for x in VAL]})
