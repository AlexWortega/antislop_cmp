"""Independent multi-model/scaffold stress suite, never used by evolution."""
import argparse
import json
import os
import random
import re
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from core import ROOT, request, digest, teacher, teacher_label, predict, read_prompt

MODELS = ['openai/gpt-5.5', 'anthropic/claude-sonnet-5.5', 'google/gemini-3.5-flash',
          'deepseek/deepseek-v4.1-flash', 'qwen/qwen3.6-plus', 'anthropic/claude-opus-5.5']
TOPICS = {
 'technical': 'Explain to a home cook why a cast-iron pan can stay hot longer than a thin aluminum pan even though aluminum conducts heat better. Include a concrete kitchen example and one limitation of the comparison.',
 'notice': 'Write a notice from the fictional Northbank community garden: the north gate will be closed on Saturday 18 April from 9 am to 2 pm to repair a leaking water pipe. Use the east entrance. The seed swap moves to Sunday at 11 am. Volunteers should bring gloves; tools are supplied. A resident named Mina coordinates volunteers.',
 'review': 'Write a fictional first-person review of a second-hand mechanical keyboard bought for 38 dollars: the space bar rattles, one key occasionally repeats, but it is comfortable for long emails. The seller answered a question about the switches a week late. The reviewer kept it rather than shipping it back.',
 'fiction': 'Write a short fictional scene: a night-shift baker discovers that someone has been returning a library book to the bakery every Thursday. Tonight a note is tucked inside. End with a small concrete action rather than an explanation of the theme.'}
STYLES = {
 'plain': 'Write the requested English text in a normal helpful style.',
 'conversational': 'Write in a conversational voice with contractions, varied rhythm, a brief aside, and a distinct point of view. Avoid sounding like an assistant introducing an answer.',
 'antislop': 'Write direct, specific English. Avoid generic introductions, summaries, motivational flourishes, balanced lists of three, corporate jargon, stock transitions, and inflated adjectives. Use concrete nouns and varied sentence lengths. Preserve the supplied facts; do not add fake evidence.'}
OUT = ROOT/'robustness'

def complete(model, messages, tag):
    path = OUT/'responses'/(digest([model,messages,tag])+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): result=json.loads(path.read_text())
    else:
        body={'model':model,'messages':messages,'max_tokens':4096,
              'reasoning':({'enabled':False} if model.startswith('qwen/') else {'effort':'minimal','exclude':True})}
        try:
            result=request('https://openrouter.ai/api/v1/chat/completions',os.environ['OPENROUTER_API_KEY'],body)
        except urllib.error.HTTPError as error:
            detail=error.read().decode().replace(os.environ['OPENROUTER_API_KEY'],'[REDACTED]')
            raise RuntimeError(model+' '+tag+' '+str(error.code)+' '+detail[:600]) from None
        path.write_text(json.dumps(result,ensure_ascii=False))
    choice=result['choices'][0]
    text=choice['message'].get('content') or ''
    if choice.get('finish_reason') != 'stop' or not text.strip():
        retry_path=path.with_name(path.stem+'.retry.json')
        if retry_path.exists(): result=json.loads(retry_path.read_text())
        else:
            body={'model':model,'messages':messages,'max_tokens':8192,'reasoning':{'enabled':False}}
            result=request('https://openrouter.ai/api/v1/chat/completions',os.environ['OPENROUTER_API_KEY'],body)
            result['_technical_retry_policy']='Incomplete first response; reasoning disabled, max_tokens=8192'
            retry_path.write_text(json.dumps(result,ensure_ascii=False))
        choice=result['choices'][0];text=choice['message'].get('content') or ''
        if choice.get('finish_reason') != 'stop' or not text.strip():
            raise ValueError('Incomplete retry for '+model+' '+tag)
    return text.strip()

def generate_pair(job):
    model, topic = job
    final=OUT/'generated'/(digest([model,topic])+'.json')
    final.parent.mkdir(parents=True,exist_ok=True)
    if final.exists(): return json.loads(final.read_text())
    task=TOPICS[topic]+' Write 150–210 words. Return only the text, without a preface.'
    rows=[]
    for scaffold, system in STYLES.items():
        text=complete(model,[{'role':'system','content':system},{'role':'user','content':task}],topic+'/'+scaffold)
        rows.append({'model':model,'scaffold':scaffold,'topic':topic,'text':text,'origin':'fresh'})
    draft=rows[0]['text']
    critique=complete(model,[{'role':'system','content':'You are an exacting prose editor. Identify generic wording, artificial transitions, monotonous rhythm, and any unsupported details. Give concise actionable criticism, not a rewrite.'},
                             {'role':'user','content':task+'\n\nDRAFT:\n'+draft}],topic+'/critique')
    text=complete(model,[{'role':'system','content':'Revise the draft using the editorial critique. Preserve all supplied facts and the requested genre. Use a specific, natural voice without adding typos, fake memories, or fake evidence. Return only the revised text, 150–210 words.'},
                        {'role':'user','content':task+'\n\nDRAFT:\n'+draft+'\n\nCRITIQUE:\n'+critique}],topic+'/rewrite')
    rows.append({'model':model,'scaffold':'draft_critique_rewrite','topic':topic,'text':text,'origin':'fresh'})
    for row in rows:
        row['id']=digest([model,row['scaffold'],topic])
        row['words']=len(row['text'].split())
        if row['words'] < 50 or row['words'] > 400:
            raise ValueError('Generation length outside predeclared bounds: '+row['id'])
    final.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print('Generated',model,topic,flush=True)
    return rows

def legacy_rows():
    repo=ROOT.parent
    rows=[]
    data=[json.loads(l) for l in (repo/'cmp/out/gen.jsonl').read_text().splitlines()]
    rng=random.Random(20261031)
    # Legacy systems have provenance in cmp/run.py; exact revisions unavailable.
    for system,model,scaffold in [('H','legacy/humanizer','rewrite'),('QR','legacy/qwen-antislop','rewrite'),('QD','legacy/qwen-antislop','direct')]:
        subset=[r for r in data if r['sys']==system and 80<=len(r['text'].split())<=400]
        rng.shuffle(subset)
        for r in subset[:8]:
            rows.append({'id':f'legacy-{system}-{r["tid"]}','model':model,'scaffold':scaffold,'topic':str(r['tid']),
                         'text':r['text'],'origin':'cmp/out/gen.jsonl','checkpoint':'unverified legacy revision'})
    data=json.loads((repo/'cmp_rawmodels/default_generations.json').read_text())
    subset=[r for r in data['generations'] if r['language']=='en' and 80<=len(r['text'].split())<=400]
    for r in subset:
        rows.append({'id':'rawmodels-'+r['id'],'model':'rawmodels.ai/antislop','scaffold':'default_reasoning',
                     'topic':r['domain'],'text':r['text'],'origin':'cmp_rawmodels/default_generations.json','checkpoint':'not exposed by API'})
    return rows

def build():
    OUT.mkdir(exist_ok=True)
    protocol={'models':MODELS,'topics':TOPICS,'styles':STYLES,'extra_scaffold':'same-model draft → critique → rewrite',
              'selection':'Full fresh factorial; deterministic legacy sample, selected before seeing Pangram/Opus labels',
              'usage':'Independent stress test, never optimization feedback','single_text_inference':True, 'requested_words':'150-210', 'accepted_fresh_words':'50-400; retain complete short answers, do not select by detector scores', 'technical_retries':'If output is empty/truncated, retry with reasoning disabled and max_tokens=8192'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ThreadPoolExecutor(max_workers=10) as pool:
        groups=list(pool.map(generate_pair,[(m,t) for m in MODELS for t in TOPICS]))
    rows=[r for group in groups for r in group]+legacy_rows()
    seen=set()
    train=[json.loads(l) for l in (ROOT/'data/train.jsonl').read_text().splitlines()]
    train_hashes={digest(' '.join(r['text'].split())) for r in train}
    cleaned=[]
    for r in rows:
        h=digest(' '.join(r['text'].split()))
        if h in train_hashes: raise ValueError('Training text leaked into robustness suite')
        if h in seen: continue
        seen.add(h);r['hash']=h;cleaned.append(r)
    (OUT/'unlabeled.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cleaned))
    print('Stress suite:',len(cleaned),flush=True)

def label():
    rows=[json.loads(l) for l in (OUT/'unlabeled.jsonl').read_text().splitlines()]
    def work(row):
        result=teacher(row['text'])
        return {**row,'slop':teacher_label(result),'teacher_version':result['version']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        labeled=[]
        for i,row in enumerate(pool.map(work,rows)):
            labeled.append(row);print('Stress label',i+1,'/',len(rows),flush=True)
    (OUT/'labeled.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in labeled))

def evaluate():
    rows=[json.loads(l) for l in (OUT/'labeled.jsonl').read_text().splitlines()]
    prompt=read_prompt(ROOT/'best_candidate.py')
    with ThreadPoolExecutor(max_workers=3) as pool:
        predictions=[]
        for i,value in enumerate(pool.map(lambda r:predict(prompt,r['text']),rows)):
            predictions.append(value)
            if (i+1)%16==0: print('Stress classified',i+1,'/',len(rows),flush=True)
    records=[{**r,'opus_slop':pred,'agree':r['slop']==pred} for r,pred in zip(rows,predictions)]
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    def summary(group):
        tp=sum(r['slop'] and r['opus_slop'] for r in group);tn=sum(not r['slop'] and not r['opus_slop'] for r in group)
        fp=sum(not r['slop'] and r['opus_slop'] for r in group);fn=sum(r['slop'] and not r['opus_slop'] for r in group)
        return {'balanced_accuracy':((tn/(tn+fp))+(tp/(tp+fn)))/2 if tn+fp and tp+fn else None,'n':len(group),'agreement':sum(r['agree'] for r in group)/len(group),'tp':tp,'tn':tn,'fp':fp,'fn':fn,
                'human_recall':tn/(tn+fp) if tn+fp else None,'slop_recall':tp/(tp+fn) if tp+fn else None}
    report={'prompt_hash':digest(prompt),'overall':summary(records),'by_model':{},'by_scaffold':{},'by_model_scaffold':{}}
    for field,target in [('model','by_model'),('scaffold','by_scaffold')]:
        for key in sorted({r[field] for r in records}):report[target][key]=summary([r for r in records if r[field]==key])
    for model,scaffold in sorted({(r['model'],r['scaffold']) for r in records}):
        report['by_model_scaffold'][model+' | '+scaffold]=summary([r for r in records if r['model']==model and r['scaffold']==scaffold])
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

def repeatability():
    rows=[json.loads(l) for l in (OUT/'predictions.jsonl').read_text().splitlines()]
    prompt=read_prompt(ROOT/'best_candidate.py')
    selected=[min([r for r in rows if r['model']==model],key=lambda r:r['hash'])
              for model in sorted({r['model'] for r in rows})]
    selected.extend(r for r in rows if not r['slop'])
    holdout=json.loads((ROOT/'holdout_results.json').read_text())['optimized']['predictions']
    holdout_texts={r['hash']:r for r in [json.loads(l) for l in (ROOT/'data/test.jsonl').read_text().splitlines()]}
    hard=sorted([r for r in holdout if r['teacher_slop'] != r['opus_slop']],key=lambda r:r['hash'])[:4]
    selected.extend({**holdout_texts[r['hash']], 'opus_slop':r['opus_slop'], 'model':'heldout_boundary_case'} for r in hard)
    selected=list({r['hash']:r for r in selected}.values())
    jobs=[(row,rep) for row in selected for rep in [1,2]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        values=list(pool.map(lambda item:predict(prompt,item[0]['text'],cache_tag='repeat-'+str(item[1])),jobs))
    report=[]
    for i,row in enumerate(selected):
        labels=[row['opus_slop']]+values[2*i:2*i+2]
        report.append({'hash':row['hash'],'model':row['model'],'labels':labels,'all_agree':len(set(labels))==1})
    result={'selection':'Lowest hash per model, all Pangram-negative stress cases, and four fixed heldout errors. Deliberately enriched for difficult cases; not a population stability estimate',
            'n_texts':len(report),'calls_per_text':3,'stable_texts':sum(r['all_agree'] for r in report),'details':report}
    (OUT/'repeatability.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','label','evaluate','repeatability']);a=p.parse_args()
    {'build':build,'label':label,'evaluate':evaluate,'repeatability':repeatability}[a.action]()
