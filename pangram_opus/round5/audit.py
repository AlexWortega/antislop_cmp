"""Offline completion audit of exported artifacts and independently recomputed scores."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,CONTRACT,digest,read_prompt,teacher_label
from round3.runtime import body_for
from round3.goal_check import check
OUT=ROOT/'round5'
def read(name):return [json.loads(l) for l in (OUT/name).read_text().splitlines()]
if __name__=='__main__':
 rows=read('test.jsonl');prior=read('prior.jsonl');train=read('train.jsonl');reports=json.loads((OUT/'test_results.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text());checks={}
 def require(name,condition):
  checks[name]=bool(condition)
  if not condition:raise AssertionError(name)
 require('active_round',manifest['active_round']=='round5')
 prompt=read_prompt(ROOT/'best_candidate.py');require('all_prompt_exports_match',prompt==(ROOT/'best_prompt.txt').read_text()==(OUT/'best_prompt.txt').read_text()==read_prompt(OUT/'best_candidate.py'))
 require('system_contract_matches',(ROOT/'inference_system.txt').read_text()==prompt+'\n\n'+CONTRACT)
 require('selected_hash',digest(prompt)==manifest['selected_prompt_hash']==reports['selected']['prompt_hash']==json.loads((OUT/'selection.json').read_text())['selected']['prompt_hash'])
 mode=json.loads((OUT/'runtime_selection.json').read_text())['mode'];config=json.loads((ROOT/'inference_config.json').read_text());require('runtime',config=={'mode':mode,'prompt_hash':digest(prompt)})
 template=json.loads((ROOT/'request_template.json').read_text());require('request_template',template==body_for(prompt,'REPLACE WITH THE ENGLISH TEXT TO CLASSIFY',mode))
 require('strict_schema',template['response_format']['type']=='json_schema' and template['response_format']['json_schema']['strict'] is True and template['provider']['require_parameters'] is True)
 expected_template=json.loads((OUT/'previous_request_template.json').read_text());expected_template['messages']=template['messages'];expected_template.pop('reasoning',None)
 if mode in ['medium','high']:expected_template['reasoning']={'effort':mode,'exclude':True}
 require('only_declared_prompt_and_reasoning_changed',expected_template==template)
 require('expected_model',template['model']=='anthropic/claude-opus-5.5')
 require('teacher_versions',all(str(r['teacher_version']).startswith('4.') for r in rows))
 require('test_unique',len({r['hash'] for r in rows})==len(rows))
 require('test_hash_integrity',all(r['hash']==digest(r['text']) for r in rows))
 require('source_groups_disjoint',not ({r['split_group'] for r in rows}&{r.get('split_group',r['group']) for r in prior}))
 require('train_texts_excluded',not ({r['hash'] for r in rows}&{r['hash'] for r in train}))
 dev=read('dev.jsonl');support=read('support.jsonl')
 oldrefs=[json.loads(l) for name in ['support.jsonl','extended_support.jsonl'] for l in (ROOT/'round3'/name).read_text().splitlines()]
 reference_groups={r['split_group'] for r in support+oldrefs}
 require('reference_groups_excluded_from_scoring',not (reference_groups&{r['split_group'] for r in train+dev}))
 require('calibration_train_dev_groups_disjoint',not ({r['split_group'] for r in train}&{r['split_group'] for r in dev}))
 require('goal_thresholds_unchanged',json.loads((OUT/'goal_criteria.json').read_text())==json.loads((ROOT/'round3/goal_criteria.json').read_text()))
 selection=json.loads((OUT/'selection.json').read_text())
 if selection.get('calibration_stage')=='expanded':
  expanded_train=read('expanded/train.jsonl');expanded_dev=read('expanded/dev.jsonl')
  require('expanded_train_dev_groups_disjoint',not ({r['split_group'] for r in expanded_train}&{r['split_group'] for r in expanded_dev}))
  fixed_reference_groups={r['split_group'] for r in oldrefs}
  require('expanded_fixed_reference_groups_excluded',not (fixed_reference_groups&{r['split_group'] for r in expanded_train+expanded_dev}))
  require('former_dev_promoted_to_training',{r['hash'] for r in dev}<={r['hash'] for r in expanded_train})
  require('expanded_labels_match_raw',all(r['slop']==teacher_label(json.loads((ROOT/'cache'/('pangram4_'+digest(r['text'])+'.json')).read_text())) for r in expanded_train+expanded_dev))
  require('expanded_selection_gate',json.loads((OUT/'expanded/selection.json').read_text())['fresh_label_gate'] is True)
 require('test_normalized_disjoint',not ({' '.join(r['text'].lower().split()) for r in rows}&{' '.join(r['text'].lower().split()) for r in prior}))
 def grams(text):
  words=re.findall(r'\w+',text.lower());return {tuple(words[i:i+5]) for i in range(len(words)-4)}
 oldgrams=[grams(r['text']) for r in prior];newgrams=[grams(r['text']) for r in rows]
 require('near_duplicates_rechecked',not any(len(a)>=8 and len(b)>=8 and len(a&b)/min(len(a),len(b))>=.8 for a in newgrams for b in oldgrams))
 for name,r in reports.items():
  pred=r['predictions'];require(name+'_prediction_alignment',len(pred)==len(rows) and all(a['hash']==b['hash'] and a['teacher']==b['slop'] for a,b in zip(pred,rows)))
  n=len(rows);positive=sum(r['slop'] for r in rows);negative=n-positive
  counts={key:0 for key in ['tp','tn','fp','fn','unavailable','unavailable_positive','unavailable_negative']}
  for example,answer in zip(rows,pred):
   v=answer['prediction'];label=example['slop']
   if v is None:counts['unavailable']+=1;counts['unavailable_positive' if label else 'unavailable_negative']+=1
   else:
    require(name+'_boolean_predictions',type(v) is bool)
    counts['tp' if label and v else 'fn' if label else 'fp' if v else 'tn']+=1
  expected={'n':n,**counts,'agreement':(counts['tp']+counts['tn'])/n,'human_recall':counts['tn']/negative,'slop_recall':counts['tp']/positive,'balanced_accuracy':(counts['tn']/negative+counts['tp']/positive)/2}
  require(name+'_metrics_recomputed',r['overall']==expected)
 require('teacher_labels_match_cached_raw',all(r['slop']==teacher_label(json.loads((ROOT/'cache'/('pangram4_'+digest(r['text'])+'.json')).read_text())) for r in rows))
 for name,item in manifest['dataset_files'].items():require(name+'_manifest_integrity',item['sha256']==hashlib.sha256((OUT/(name+'.jsonl')).read_bytes()).hexdigest() and item['rows']==len(read(name+'.jsonl')))
 require('six_generator_models',len({r['model'] for r in rows if r['origin']=='ai_direct'})==6)
 generated=[json.loads(p.read_text()) for p in (OUT/'fresh_generation').glob('*.json')]
 require('generator_model_identity',len(generated)==72 and all(r['response'].get('model')==r['request']['model']==r['row']['model'] for r in generated))
 require('requested_scenarios',{'human','ai_direct','ai_rewrite','human_format_only','mixed'}<={r['origin'] for r in rows})
 require('short_length_coverage',{'lt25','25-49','50-99','100-199'}<={r['token_bucket'] for r in rows})
 b=json.loads((ROOT/'cache/budget.json').read_text());costs={k:v+sum(x['amount'] for x in b['pending'].values() if x['provider']==k) for k,v in b['spent'].items()};require('cumulative_budgets',all(costs[k]<=100 and b['limits'][k]==100 for k in costs))
 # Exercise the public CLI on existing cached predictions. This makes no new paid requests.
 selected=reports['selected']['predictions'];samples=[]
 for label in [False,True]:
  for bucket in ['lt25','25-49','50-99','100-199']:
   pair=next(((r,p) for r,p in zip(rows,selected) if r['slop']==label and r['token_bucket']==bucket and p['prediction'] is not None),None)
   if pair:samples.append(pair)
 for r,p in samples:
  key=digest(['strict-runtime-v1',mode,prompt,r['text'],None]);require('cli_cache_present_'+r['hash'],(ROOT/'cache'/('opus_runtime_'+key+'.json')).exists())
  run=subprocess.run([sys.executable,str(ROOT/'classify.py')],input=r['text'],text=True,capture_output=True)
  require('cli_matches_'+r['hash'],run.returncode==0 and json.loads(run.stdout)=={'slop':p['prediction']})
 assessment=check(reports['selected'],json.loads((OUT/'goal_criteria.json').read_text()));require('goal_assessment_consistent',assessment==json.loads((OUT/'goal_assessment.json').read_text()))
 result={'artifact_checks_passed':all(checks.values()),'checks':checks,'cli_cases':len(samples),'goal_met':assessment['met'],'failed_quality_checks':assessment['failed'],'budget_totals':costs,'note':'Artifact correctness is separate from quality. Operational point estimates do not prove universal equivalence to Pangram.'}
 (OUT/'completion_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
