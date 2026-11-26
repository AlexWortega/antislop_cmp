import json,time
from pipeline import OUT,stage,waitfile
waitfile(OUT/'runtime_selection.json',3600)
stage('anchors',['pangram_opus/round3/extend_anchors.py'])
stage('test_final',['pangram_opus/round3/finish.py','test'])
stage('publish',['pangram_opus/round3/publish.py'])
(OUT/'pipeline_status.json').write_text(json.dumps({'stage':'complete','status':'complete','finished':time.time()},indent=2))
