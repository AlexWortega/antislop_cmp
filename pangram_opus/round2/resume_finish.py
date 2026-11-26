import json
import time
from pipeline import OUT,stage
stage('validate',['pangram_opus/round2/finish_round.py','validate'])
stage('test',['pangram_opus/round2/finish_round.py','test'])
stage('summarize',['pangram_opus/round2/summarize_round.py'])
stage('publish',['pangram_opus/round2/publish_round.py'])
(OUT/'pipeline_status.json').write_text(json.dumps({'stage':'complete','status':'complete','finished':time.time()},indent=2))
