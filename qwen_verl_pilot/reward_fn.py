import requests

def compute_score(data_source,solution_str,ground_truth,extra_info=None,**kwargs):
 extra_info=extra_info or {}
 r=requests.post('http://127.0.0.1:18971',json={'question':extra_info['question'],'answer':solution_str},timeout=300)
 if not r.ok:raise RuntimeError('Reward service HTTP '+str(r.status_code)+': '+r.text[:500])
 result=r.json()
 if 'score' not in result:raise RuntimeError('Reward service unavailable')
 return result
