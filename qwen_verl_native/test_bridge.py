import asyncio,importlib.util,json,os,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
os.environ['ANTISLOP_NATIVE_ROOT']=str(ROOT)
stub=types.ModuleType('verl.experimental.agent_loop.agent_loop');stub.AgentLoopBase=object;stub.AgentLoopOutput=dict
with patch.dict(sys.modules,{'verl.experimental.agent_loop.agent_loop':stub}):
 spec=importlib.util.spec_from_file_location('bridge_under_test',ROOT/'native_agent.py');bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
from fastapi.testclient import TestClient

class FakeSession:
 async def complete(self,body):return ({'role':'assistant','content':None,'tool_calls':[{'id':'call_1','type':'function','function':{'name':'terminal','arguments':'{"command":"pwd"}'}}]},'tool_calls',{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15})

class BridgeTests(unittest.TestCase):
 def setUp(self):self.client=TestClient(bridge.APP);bridge.SESSIONS['uid']=FakeSession()
 def test_rejects_unauthenticated_generation(self):
  self.assertEqual(self.client.post('/session/uid/trial/t/v1/chat/completions',json={}).status_code,401)
 def test_stream_preserves_tool_calls(self):
  response=self.client.post('/session/uid/trial/t/v1/chat/completions',json={'stream':True},headers={'Authorization':'Bearer '+bridge.KEY})
  self.assertEqual(response.status_code,200)
  lines=[line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
  self.assertEqual(lines[-1],'[DONE]')
  call=json.loads(lines[0])['choices'][0]['delta']['tool_calls'][0]
  self.assertEqual(call['index'],0);self.assertEqual(json.loads(call['function']['arguments']),{'command':'pwd'})
  self.assertEqual(json.loads(lines[1])['choices'][0]['finish_reason'],'tool_calls')
 def test_nested_models_route_for_scaffolds(self):
  self.assertEqual(self.client.get('/session/uid/trial/t/v1/models').json()['data'][0]['max_model_len'],65536)
 def test_unknown_session_is_rejected(self):
  self.assertEqual(self.client.post('/session/absent/trial/t/v1/chat/completions',json={},headers={'Authorization':'Bearer '+bridge.KEY}).status_code,404)
 def test_training_rows_are_four_groups_of_four(self):
  import pandas as pd
  data=pd.read_parquet(ROOT/'train.parquet')
  for start in range(0,len(data),4):
   group=data.iloc[start:start+4];self.assertEqual(len(group),4)
   self.assertEqual(len({x['group'] for x in group.extra_info}),4)
   self.assertTrue(all(x['split']=='train' for x in group.extra_info))

if __name__=='__main__':unittest.main()
