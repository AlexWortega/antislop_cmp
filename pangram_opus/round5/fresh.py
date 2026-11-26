import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT
from round3 import fresh as f
f.OUT=ROOT/'round5';f.PREFIX='round5';f.prior=lambda:f.read(f.OUT/'prior.jsonl')
f.WANTED=['enron','yelp','brown_news','brown_learned','brown_hobbies','inaugural']
f.TOPICS={
'lt25':[
('business','Write one brief notice saying a fictional botanical garden gift shop stops accepting cash five minutes before closing.'),
('review','Write a one-sentence mixed review of a fictional insulated lunch bag: keeps food cool, zipper catches on the lining.')],
'25-49':[
('technical','Explain briefly why a metal spoon left in hot soup becomes warm at the handle.'),
('academic','Write a brief academic-style statement distinguishing a legal boundary on a map from the way residents describe their neighborhood.')],
'50-99':[
('business','Write a practical notice for a fictional neighborhood repair cafe: bring one broken item, take a numbered ticket, and collect it before volunteers leave. Avoid promotional language.'),
('review','Write a mixed first-person review of a fictional rechargeable reading light with a secure clip, uneven brightness, and a charging port hidden under a stiff cover.')],
'100-199':[
('technical','Explain how a simple rain gauge measures rainfall, why its opening and measuring tube may have different diameters, and two common sources of inaccurate readings. Use plain accurate prose.'),
('academic','Discuss how old school attendance registers can help historians study daily life; mention a limitation without inventing citations.')]
}
def label_with_budget_check():
 rows=f.read(f.OUT/'fresh_unlabeled.jsonl');ledger=json.loads((ROOT/'cache/budget.json').read_text())
 total=ledger['spent']['pangram']+sum(v['amount'] for v in ledger['pending'].values() if v['provider']=='pangram')
 estimate=.055*1.15*sum(max(1,math.ceil(len(r['text'].split())/100)) for r in rows)
 check={'remaining_conservative':ledger['limits']['pangram']-total,'estimated_label_cost_with_extra_margin':estimate,'n':len(rows),'note':'Estimate only; the cumulative reservation guard still enforces the actual cap.'}
 (f.OUT/'teacher_budget_check.json').write_text(json.dumps(check,indent=2))
 if estimate>check['remaining_conservative']:raise RuntimeError('Insufficient estimated teacher budget for the complete fresh test; no requests submitted.')
 f.label()
if __name__=='__main__':{'build':f.build,'label':label_with_budget_check}[sys.argv[1]]()
