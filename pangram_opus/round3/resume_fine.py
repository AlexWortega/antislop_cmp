import json,time
from pipeline import OUT,stage
for _ in range(1200):
 try:ready='bounded support/rubric' in json.loads((OUT/'selection.json').read_text())['criterion']
 except (FileNotFoundError,json.JSONDecodeError):ready=False
 if ready:break
 time.sleep(3)
else:raise TimeoutError('Static candidate selection')
stage('fine',['pangram_opus/round3/fine_tune.py'])
stage('test_final',['pangram_opus/round3/finish.py','test'])
stage('publish',['pangram_opus/round3/publish.py'])
(OUT/'pipeline_status.json').write_text(json.dumps({'stage':'complete','status':'complete','finished':time.time()},indent=2))
