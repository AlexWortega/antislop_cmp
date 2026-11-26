"""Score collected files outside agent containers with the shared API budget."""
import argparse
import copy
import importlib.util
import json
import re
from pathlib import Path
import reward_server as budget

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('unslop_reward', ROOT/'dataset_sources/unslop/reward.py')
unslop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unslop)
cfg = json.loads((ROOT/'dataset_sources/unslop/data/reward_config.json').read_text())
sources = {x['task']:x for x in map(json.loads,(ROOT/'selected_source_rows.jsonl').read_text().splitlines())}

def native_judge(config, assignment, rubric, answer):
    model = config['judge']
    body = {'model':model,'max_tokens':config['judge_max_tokens'],'reasoning':config['reasoning'],
            'provider':{'require_parameters':True,**config['provider']},
            'response_format':{'type':'json_schema','json_schema':{'name':'single_continuation_score','strict':True,'schema':unslop.single_score_schema(rubric)}},
            'messages':[{'role':'system','content':config['judge_system']},
                        {'role':'user','content':unslop.score_prompt(assignment,rubric,answer)}]}
    if model not in config['omit_temperature_models']:body['temperature']=0
    raw = budget.call(body)
    result = budget.content(raw)
    return unslop.score_value(result,rubric),result,{'model':raw.get('model'),'usage':raw.get('usage')}
unslop.judge = native_judge

def anti_slop(answer):
    body = copy.deepcopy(budget.TEMPLATE)
    body['messages'][-1]['content']=json.dumps([{'id':0,'text':answer}],ensure_ascii=False)
    item = budget.content(budget.call(body))['results'][0]
    assert item['id']==0 and type(item['slop']) is bool
    return float(not item['slop'])

def normalized(s):
    return re.sub(r'\s+',' ',s.casefold().replace('−','-').replace('μ','µ')).strip()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--groups',type=Path,required=True);ap.add_argument('--trial-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    groups=[json.loads(s) for s in a.groups.read_text().splitlines()]
    metrics=[]
    for group in groups:
        source=sources[group['task']];row=source['row']
        for trajectory in group['trajectories']:
            grade=trajectory['grade_outcome'];trial=trajectory['trial_id']
            record={'trial_id':trial,'task':group['task'],'split':group['split'],'agent':group['agent'],'source':source['source']}
            if not grade.get('valid'):
                record.update(valid=False,reason='verifier_unavailable');metrics.append(record);continue
            if grade['reward']==0:
                result={'score':0.,'native_reward':0.,'anti_slop':0.,'file_success':False}
            else:
                files=list(a.trial_root.glob(f'**/{trial}/verifier/answer.txt'))
                if len(files)!=1:raise RuntimeError(f'Expected one captured answer for {trial}, found {files}')
                answer=files[0].read_text()
                if source['source']=='unslop-harbor':
                    assert row['domain']=='tweets'
                    native=unslop.grade({**row,'assignment':row['prompt']},cfg,answer)
                    anti=anti_slop(answer)
                    result={'score':native['reward']*(.7+.3*anti),'native_reward':native['reward'],'anti_slop':anti,'q':native['q'],'length_factor':native['length_factor'],'file_success':True}
                else:
                    text=normalized(re.sub(r'https?://\S+','',answer))
                    correct=any(re.search(r'(?<!\w)'+re.escape(normalized(alias))+r'(?!\w)',text) for alias in [row['answer'],*row['aliases']])
                    judged=budget.judge(row['question']+'\nCorrect reference answer: '+row['answer']+'\nSupporting evidence: '+json.dumps(row['evidence']),answer)
                    result={'score':float(correct)*judged['score'],'native_reward':float(correct),'anti_slop':judged['anti_slop'],'quality':judged['quality'],'file_success':True}
                record['answer']=answer
            trajectory['outcome']={'valid':True,'reward':result['score'],'source':'dataset_plus_calibrated_opus'}
            record.update(valid=True,**result);metrics.append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='answer'}),flush=True)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'groups.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in groups))
    (a.out/'metrics.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in metrics))

if __name__=='__main__':main()
