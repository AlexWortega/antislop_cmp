"""Real verl AgentLoop: Harbor runs locally; every model call uses verl's LLMServerClient."""
import asyncio,copy,json,os,time,uuid
from pathlib import Path
import httpx
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from verl.experimental.agent_loop.agent_loop import AgentLoopBase,AgentLoopOutput

ROOT=Path(os.environ.get('ANTISLOP_NATIVE_ROOT','/workspace/native'));SESSIONS={};SERVER=None;COUNTS={}
ATTEMPT=uuid.uuid4().hex[:12]
KEY=(ROOT/'serve.key').read_text().strip()
APP=FastAPI()

@APP.get('/v1/models')
@APP.get('/session/{uid}/trial/{trial}/v1/models')
async def models(uid:str='',trial:str=''):return {'data':[{'id':'antislop-scaffold','object':'model','max_model_len':65536}]}

@APP.post('/session/{uid}/trial/{trial}/v1/chat/completions')
async def completion(uid:str,trial:str,request:Request):
 if request.headers.get('authorization')!='Bearer '+KEY:raise HTTPException(401)
 if uid not in SESSIONS:raise HTTPException(404)
 body=await request.json();message,finish,usage=await SESSIONS[uid].complete(body)
 ident='chatcmpl-'+uuid.uuid4().hex
 if not body.get('stream'):
  return {'id':ident,'object':'chat.completion','created':int(time.time()),'model':'antislop-scaffold','choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':usage}
 async def chunks():
  delta=copy.deepcopy(message)
  for i,call in enumerate(delta.get('tool_calls',[])):call['index']=i
  for d,reason in [(delta,None),({},finish)]:
   data={'id':ident,'object':'chat.completion.chunk','created':int(time.time()),'model':'antislop-scaffold','choices':[{'index':0,'delta':d,'finish_reason':reason}]}
   yield 'data: '+json.dumps(data)+'\n\n'
  yield 'data: [DONE]\n\n'
 return StreamingResponse(chunks(),media_type='text/event-stream')

class HarborAgentLoop(AgentLoopBase):
 async def complete(self,body):
  async with self.call_lock:
   messages=body['messages']
   if messages and messages[0].get('role')=='system' and str(messages[0].get('content','')).startswith('You name chat sessions.'):
    prompt=await self.ct_build_initial_tokens(messages)
    params=dict(self.sampling_params);params['max_tokens']=min(128,body.get('max_tokens') or 64)
    clock=time.time();output=await self.server_manager.generate(request_id=self.uid+'-title',prompt_ids=prompt,sampling_params=params);self.timed(clock);self.versions(output)
    with (ROOT/'audit'/f'{self.uid}-auxiliary.jsonl').open('a') as f:f.write(json.dumps({'kind':'session_title','excluded_from_training':True,'prompt_ids':prompt,'response_ids':output.token_ids})+'\n')
    return {'role':'assistant','content':self.tokenizer.decode(output.token_ids,skip_special_tokens=True)},'stop',{'prompt_tokens':len(prompt),'completion_tokens':len(output.token_ids),'total_tokens':len(prompt)+len(output.token_ids)}
   signature=json.dumps(body,sort_keys=True,ensure_ascii=False)
   if signature==getattr(self,'last_signature',None):return self.last_result
   messages=body['messages'];tools=body.get('tools')
   message_key=json.dumps(messages,sort_keys=True,ensure_ascii=False)
   snapshots=getattr(self,'snapshots',{})
   rewound=False
   if self.turns and message_key in snapshots:
    prior=snapshots[message_key]
    self.tokens=list(prior['tokens']);self.mask=list(prior['mask']);self.logps=list(prior['logps']);rewound=True
   if self.turns==0:
    self.tokens=await self.ct_build_initial_tokens(messages,tools=tools)
    self.initial=list(self.tokens)
    if len(self.tokens)>=self.rollout_config.prompt_length:raise ValueError('Initial prompt at truncation boundary')
   elif not rewound:
    # The scaffold may canonicalize assistant dictionaries; use its current
    # prefix up to the last generated assistant, never re-tokenize model output.
    prefix=messages[:self.previous_count+1]
    if len(prefix)!=self.previous_count+1 or prefix[-1]['role']!='assistant':
     (ROOT/'audit'/f'{self.uid}-context-error.json').write_text(json.dumps({'previous_count':self.previous_count,'turns':self.turns,'messages':messages},ensure_ascii=False))
     raise ValueError('Scaffold rewrote trajectory structure: '+str({'previous_count':self.previous_count,'roles':[m['role'] for m in messages]}))
    merged,self.mask,self.logps=await self.ct_merge_context_msg(prefix,messages,self.tokens,self.mask,self.logps,tools=tools)
    self.tokens=merged.token_ids
   snapshots[message_key]={'tokens':list(self.tokens),'mask':list(self.mask),'logps':list(self.logps)}
   self.snapshots=snapshots
   remaining=self.rollout_config.response_length-len(self.mask)
   if remaining<=0:raise ValueError('Trajectory exceeded response budget')
   params=dict(self.sampling_params);params['max_tokens']=min(2048,remaining)
   clock=time.time();output=await self.server_manager.generate(request_id=self.uid,prompt_ids=self.tokens,sampling_params=params);gen_s=self.timed(clock);self.versions(output)
   if not output.token_ids or output.log_probs is None:raise ValueError('Missing exact generated tokens/log probabilities')
   prompt=list(self.tokens)
   merged,self.mask,self.logps=await self.ct_merge_assistant_token(self.tokens,output.token_ids,self.mask,self.logps,assistant_logprobs=output.log_probs)
   self.tokens=merged.token_ids
   text=self.tokenizer.decode(output.token_ids,skip_special_tokens=False)
   from vllm.tool_parsers import ToolParserManager
   from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
   parser=ToolParserManager.get_tool_parser('qwen3_xml')(self.tokenizer)
   parsed=parser.extract_tool_calls(text,request=ChatCompletionRequest(model='antislop-scaffold',messages=messages,tools=tools,tool_choice='auto' if tools else 'none'))
   if not tools or body.get('tool_choice')=='none':
    from types import SimpleNamespace
    parsed=SimpleNamespace(tools_called=False,content=None)
   content=parsed.content
   if not parsed.tools_called:
    content=self.tokenizer.decode(output.token_ids,skip_special_tokens=True)
   message={'role':'assistant','content':content}
   if parsed.tools_called:message['tool_calls']=[c.model_dump(exclude_none=True) for c in parsed.tool_calls]
   self.previous_count=len(messages);self.turns+=1
   audit={'uid':self.uid,'turn':self.turns,'prompt_ids':prompt,'response_ids':output.token_ids,'old_logprobs':output.log_probs,'rewound_uncommitted_response':rewound,'gen_s':round(gen_s,3)}
   with (ROOT/'audit'/f'{self.uid}.jsonl').open('a') as f:f.write(json.dumps(audit)+'\n')
   result=message,('tool_calls' if parsed.tools_called else 'stop'),{'prompt_tokens':len(prompt),'completion_tokens':len(output.token_ids),'total_tokens':len(prompt)+len(output.token_ids)}
   self.last_signature=signature;self.last_result=result
   return result

 def split_time(self,started):
  # verl's slots: generate_sequences = model time; tool_calls = agent and container time between model calls;
  # compute_score = after the last model call until the reward came back (agent exit, verifier, copy, judges).
  end=time.time();last=self.last_llm_end or started
  return {'generate_sequences':self.llm_s,'tool_calls':max(0.,last-started-self.llm_s),'compute_score':end-last}

 def versions(self,output):
  # Weight versions the tokens came from; the async trainer needs both on every trajectory (staleness stats).
  extra=getattr(output,'extra_fields',None) or {}
  lo,hi=extra.get('min_global_steps'),extra.get('max_global_steps')
  if lo is not None and self.min_steps is None:self.min_steps=lo
  if hi is not None:self.max_steps=hi

 def timed(self,clock):
  # Time inside the model, for splitting a trajectory into model / agent between turns / tail after the last turn.
  now=time.time();self.llm_s+=now-clock;self.last_llm_end=now;return now-clock

 async def run(self,sampling_params,**kwargs):
  global SERVER
  if SERVER is None:
   SERVER=uvicorn.Server(uvicorn.Config(APP,host='127.0.0.1',port=9001,log_level='warning'))
   asyncio.create_task(SERVER.serve())
   while not SERVER.started:await asyncio.sleep(.05)
  self.uid=uuid.uuid4().hex;self.call_lock=asyncio.Lock();self.tokens=[];self.initial=[];self.mask=[];self.logps=[];self.turns=0
  self.llm_s=0.;self.last_llm_end=None;self.min_steps=None;self.max_steps=None
  self.sampling_params=sampling_params;SESSIONS[self.uid]=self
  meta=kwargs['extra_info'];meta=meta.item() if hasattr(meta,'item') else meta
  group=meta['group'];counter=COUNTS.get(group,0);COUNTS[group]=counter+1
  size=int((ROOT/'group_size').read_text()) if (ROOT/'group_size').exists() else 4
  # Validation samples val_kwargs.n per prompt, which differs from the training group size.
  if meta['split']!='train':size=int((ROOT/'val_group_size').read_text()) if (ROOT/'val_group_size').exists() else 4
  group_cycle=f'{group}-cycle{counter//size}-attempt{ATTEMPT}'
  started=time.time()
  try:
   async with httpx.AsyncClient(timeout=1800) as client:
    result=await client.post('http://127.0.0.1:18975/run',headers={'Authorization':'Bearer '+KEY},json={'uid':self.uid,'group':group_cycle,'task':meta['task'],'agent':meta['agent'],'split':meta['split'],'policy':meta['policy'],'size':size})
    if not result.is_success:raise RuntimeError('Harbor callback failed: '+result.text[:500])
    result=result.json()
   if result.get('infra_failed'):
    # The trial broke for infrastructure reasons: keep the sample for batch shape, train on none of it.
    if not self.initial or not self.mask:
     self.initial=await self.ct_build_initial_tokens(list(kwargs['raw_prompt']))
     self.tokens=self.initial+[self.tokenizer.eos_token_id];self.mask=[0];self.logps=[0.]
    self.mask=[0]*len(self.mask)
   if not self.initial or not self.mask:raise ValueError('No model trajectory captured')
   # A long tool result merged after the last turn can push the response past response_length, which
   # breaks batch collation. Truncate as verl's own agent loops do; the overflow is masked tool output.
   limit=self.rollout_config.response_length
   if len(self.mask)>limit:
    with (ROOT/'audit'/f'{self.uid}-truncated.json').open('w') as f:f.write(json.dumps({'response_tokens':len(self.mask),'limit':limit}))
    self.tokens=self.tokens[:len(self.initial)+limit];self.mask=self.mask[:limit];self.logps=self.logps[:limit]
   if len(self.tokens)-len(self.initial)!=len(self.mask):raise ValueError('Trajectory mask alignment failed')
   if self.tokens[:len(self.initial)]!=self.initial:raise ValueError('Prompt changed during trajectory')
   return AgentLoopOutput(prompt_ids=self.initial,response_ids=self.tokens[len(self.initial):],response_mask=self.mask,response_logprobs=self.logps,reward_score=result['reward'],num_turns=self.turns*2,metrics=self.split_time(started),extra_fields={'harbor_uid':self.uid,'group':group_cycle,'reward_components':result['metrics'],
    'min_global_steps':self.min_steps if self.min_steps is not None else (self.max_steps or 0),
    'max_global_steps':self.max_steps if self.max_steps is not None else (self.min_steps or 0)})
  finally:SESSIONS.pop(self.uid,None)
