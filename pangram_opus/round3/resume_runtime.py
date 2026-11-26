import json,time
from pipeline import OUT,stage,waitfile
(OUT/'pipeline_status.json').write_text(json.dumps({'stage':'runtime_selection','status':'running','started':time.time()},indent=2))
waitfile(OUT/'runtime_selection.json',3600)
stage('test_runtime',['pangram_opus/round3/finish.py','test'])
stage('publish',['pangram_opus/round3/publish.py'])
(OUT/'pipeline_status.json').write_text(json.dumps({'stage':'complete','status':'complete','finished':time.time()},indent=2))
