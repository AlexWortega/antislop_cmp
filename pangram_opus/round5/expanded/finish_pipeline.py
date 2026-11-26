import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core import ROOT
from round5.pipeline import stage
ROUND=ROOT/'round5'
if __name__=='__main__':
 stage('adopt_expanded',['pangram_opus/round5/expanded/adopt.py'])
 stage('label',['pangram_opus/round5/fresh.py','label'])
 stage('test',['pangram_opus/round5/finish.py'])
 stage('publish',['pangram_opus/round5/publish.py'])
 stage('audit',['pangram_opus/round5/audit.py'])
 (ROUND/'pipeline_status.json').write_text(json.dumps({'stage':'complete','status':'complete','finished':time.time()},indent=2))
