import json,time
from pipeline import OUT,stage
stage('evolve',['pangram_opus/round3/run_evo.py'])
stage('validate',['pangram_opus/round3/finish.py','validate'])
stage('label',['pangram_opus/round3/fresh.py','label'])
stage('test',['pangram_opus/round3/finish.py','test'])
stage('publish',['pangram_opus/round3/publish.py'])
(OUT/'pipeline_status.json').write_text(json.dumps({'stage':'complete','status':'complete','finished':time.time()},indent=2))
