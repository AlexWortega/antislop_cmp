import json
from pathlib import Path
import requests

root = Path(__file__).resolve().parent
headers = {'Authorization': 'Bearer '+Path('/tmp/antislop-pilot-serve.key').read_text().strip()}
body = {'model':'antislop', 'messages':[{'role':'user','content':'Reply with the single word ready.'}],
        'max_tokens':32, 'temperature':1, 'top_p':1, 'logprobs':True,
        'return_token_ids':True, 'return_tokens_as_token_ids':True}
response = requests.post('http://127.0.0.1:18972/trial/api-smoke/v1/chat/completions',
                         headers=headers, json=body, timeout=120)
response.raise_for_status()
data = response.json()
(root/'scaffold_api_probe.json').write_text(json.dumps(data,indent=2))
choice = data['choices'][0]
print(json.dumps({'finish':choice['finish_reason'], 'message':choice['message'],
                  'response_ids':choice.get('token_ids'), 'logprobs':choice.get('logprobs'),
                  'prompt_token_count':len(data.get('prompt_token_ids',[]))}))
