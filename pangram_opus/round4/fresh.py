import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT
from round3 import fresh as f
f.OUT=ROOT/'round4';f.PREFIX='round4';f.prior=lambda:f.read(f.OUT/'prior.jsonl')
f.TOPICS={
'lt25':[
('business','Write one brief notice saying a fictional swimming pool equipment counter stops lending kickboards ten minutes before closing.'),
('review','Write a one-sentence mixed review of a fictional folding umbrella: compact when packed, difficult to fold back neatly.')],
'25-49':[
('technical','Explain briefly why condensation appears on the outside of a glass of iced water on a humid day.'),
('academic','Write a brief academic-style statement distinguishing a recorded interview from the memory of the event being described.')],
'50-99':[
('business','Write a practical notice for a fictional community tool library: reserve tools online, bring identification, and report damage before returning them. Avoid a marketing tone.'),
('review','Write a mixed first-person review of a fictional kitchen timer with a clear display, an annoying beep, and buttons difficult to press with wet hands.')],
'100-199':[
('technical','Explain how a siphon transfers water between two containers, why the outlet must be lower than the source surface, and how air entering the tube can interrupt flow. Use plain accurate prose.'),
('academic','Discuss how old household shopping lists can help historians study daily consumption; mention a limitation without inventing citations.')]
}
if __name__=='__main__':{'build':f.build,'label':f.label}[sys.argv[1]]()
