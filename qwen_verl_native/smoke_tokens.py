"""No model generation: exercise the real Qwen tokenizer, XML parser and verl token masks."""
import asyncio,json
from types import SimpleNamespace
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.utils.dataset.rl_dataset import RLHFDataset
from native_agent import HarborAgentLoop,ROOT

async def main():
 cfg=OmegaConf.load(ROOT/'output/manual-preflight.yaml')
 tokenizer=AutoTokenizer.from_pretrained('/workspace/antislop_model')
 class Server:
  async def generate(self,**kwargs):
   ids=tokenizer.encode('<tool_call>\n<function=terminal>\n<parameter=command>pwd</parameter>\n</function>\n</tool_call><|im_end|>',add_special_tokens=False)
   return SimpleNamespace(token_ids=ids,log_probs=[-1.]*len(ids))
 loop=HarborAgentLoop(trainer_config=DictConfigWrap(config=cfg),server_manager=Server(),tokenizer=tokenizer,processor=None,dataset_cls=RLHFDataset,data_config=DictConfigWrap(config=cfg.data),hf_model_type='qwen3_5')
 loop.uid='smoke';loop.call_lock=asyncio.Lock();loop.tokens=[];loop.initial=[];loop.mask=[];loop.logps=[];loop.turns=0;loop.sampling_params={}
 tools=[{'type':'function','function':{'name':'terminal','description':'Run a command','parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}]
 messages=[{'role':'user','content':'Run pwd.'}]
 first,finish,_=await loop.complete({'messages':messages,'tools':tools})
 assert finish=='tool_calls' and first['tool_calls'][0]['function']['name']=='terminal',first
 messages=messages+[first,{'role':'tool','tool_call_id':first['tool_calls'][0]['id'],'content':'/app'}]
 second,_,_=await loop.complete({'messages':messages,'tools':tools})
 assert loop.tokens[:len(loop.initial)]==loop.initial
 assert len(loop.tokens)-len(loop.initial)==len(loop.mask)==len(loop.logps)
 assert 0 in loop.mask and 1 in loop.mask
 assert all(p==0 for p,m in zip(loop.logps,loop.mask) if not m)
 # A scaffold can retry the same history with tools disabled. Discard the
 # uncommitted branch from the loss, while retaining every call in the audit.
 retry={'messages':[{'role':'user','content':'Run pwd.'}],'tools':tools,'tool_choice':'none'}
 _,finish,_=await loop.complete(retry)
 assert finish=='stop' and loop.turns==3
 assert 0 not in loop.mask
 before_count=loop.turns
 await loop.complete(retry)
 assert loop.turns==before_count
 preserved=(list(loop.tokens),list(loop.mask),list(loop.logps),loop.turns)
 await loop.complete({'messages':[{'role':'system','content':'You name chat sessions. Return a short title.'},{'role':'user','content':'Run pwd.'}]})
 assert preserved==(loop.tokens,loop.mask,loop.logps,loop.turns)
 print(json.dumps({'status':'passed','turns':loop.turns,'generated_tokens':sum(loop.mask),'tool_tokens_masked':len(loop.mask)-sum(loop.mask)}))
asyncio.run(main())
