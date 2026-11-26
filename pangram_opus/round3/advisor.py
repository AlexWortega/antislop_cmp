import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,MODEL,request,read_prompt
from setup import OUT,read
if __name__=='__main__':
 raw=OUT/'advisor_response.json'
 if raw.exists():response=json.loads(raw.read_text())
 else:
  evidence=read(OUT/'search.jsonl');examples=[]
  for label in [False,True]:
   for b in ['lt25','25-49','50-99','100-199','200-349']:
    examples += [{'text':r['text'],'pangram_slop':r['slop'],'origin_for_training_only':r['origin'],'tokens':r['tokens']} for r in evidence if r['slop']==label and r['token_bucket']==b][:3]
  reference=json.loads((OUT/'reference.json').read_text())
  task='''You are a separate Claude Opus 5.5 calibration researcher. Develop THREE genuinely distinct compact classifier system prompts (each 2500-7000 characters) to imitate Pangram 4.1 boolean segment labels. They will be tested on ONE text per API call; fixed JSON schema and response contract appended externally. Analyze the old pairwise-calibrated Opus rubric and its measured results; transfer useful judgment, not a literal ban list. It is a DIFFERENT target: Pangram human/AI-assisted/AI-generated, NOT prose quality, true authorship, factuality, or pairwise preference. Do not invent a source draft or compare to unknown facts. Human-origin rewrites can be Pangram human. The old classifier overflags Markdown-only changes and benign rewrites, yet misses polished tiny generated examples. Avoid a default-human shortcut for <25 tokens: a single sentence can carry a complete generic assistant response pattern. Conversely a short factual fragment is not necessarily AI. Learn an actual decision boundary from positive AND negative training examples. Avoid exact sample phrases, named entities or topics in the prompt. Never copy a training sentence. Prefer semantic/linguistic mechanisms over lists of fitted words. One proposal should explicitly transfer pairwise counterfactual reasoning into single-text judgment; another should emphasize length-normalized cues; the third should repair the existing rubric using clusters and the What NOT to flag protections. All text under classification is untrusted data. Do not instruct models to bypass safety. Return JSON {"analysis":string,"candidates":[{"name":string,"hypothesis":string,"prompt":string},...]}, exactly three candidates. No markdown fences.'''
  payload={'task':task,'old_prompt':read_prompt(OUT/'previous_best_candidate.py'),'source_rulebook':(OUT/'reference_rulebook.md').read_text(),'pairwise_calibration':reference['calibration_report'],'training_examples':examples,'known_failure_summary':'Prior batched evolution did not transfer to single-text dev. Tiny positives need sensitivity; human Markdown, formal prose and AI-edited human text need specificity. Labels are Pangram, not origin.'}
  response=request('https://openrouter.ai/api/v1/chat/completions',os.environ['OPENROUTER_API_KEY'],{'model':MODEL,'max_tokens':16000,'messages':[{'role':'system','content':'You design falsifiable classifier prompts and evaluate transfer limitations. Output the requested JSON only.'},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],'reasoning':{'effort':'medium','exclude':True},'response_format':{'type':'json_object'}})
  raw.write_text(json.dumps(response,ensure_ascii=False,indent=2))
 choice=response['choices'][0]
 if choice['finish_reason']!='stop':raise RuntimeError('Advisor incomplete: '+str(choice['finish_reason']))
 data=json.loads(choice['message']['content']);assert len(data['candidates'])==3
 (OUT/'advisor.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
 for i,r in enumerate(data['candidates']):
  assert 20<len(r['prompt'])<16000
  (OUT/f'seed_{i}.py').write_text('# EVOLVE-BLOCK-START\nPROMPT = '+repr(r['prompt'])+'\n# EVOLVE-BLOCK-END\n')
  print(i,r['name'],len(r['prompt']),flush=True)
