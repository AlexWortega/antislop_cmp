"""Side-by-side chat with the base model and the step-50 LoRA. Runs on the laptop, talks to vLLM through the ssh tunnel.

    python3 compare_ui.py   ->  http://127.0.0.1:7860
"""
import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = json.loads((HERE / 'play_state.json').read_text())['serve_key']
API = 'http://127.0.0.1:8000/v1/chat/completions'
MODELS = ('base', 'step50', 'step50x4')

PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Antislop compare</title><style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1db;--acc:#2f6f5e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9992;--card:#1f1f1d;--line:#33332f;--acc:#6fbfa8}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1300px;margin:0 auto;padding:16px}
textarea,input,select{width:100%;box-sizing:border-box;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px;font:inherit}
label{font-size:13px;color:var(--mut)} .row{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:8px 0}
.row>div{flex:1;min-width:120px} button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:9px 18px;font:inherit;cursor:pointer}
button.alt{background:transparent;color:var(--acc);border:1px solid var(--acc)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px} @media(max-width:800px){.cols{grid-template-columns:1fr}}
.out{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;white-space:pre-wrap;min-height:120px}
.out h3{margin:0 0 6px;font-size:14px} .meta{color:var(--mut);font-size:12px;margin-bottom:8px}
</style></head><body><main>
<h2 style="margin:4px 0">base vs <select id="right" style="width:auto;font-size:18px"><option value="step50">step50</option><option value="step50x4">step50x4 (LoRA ×4)</option></select></h2>
<label>System prompt (необязательно)</label><textarea id="sys" rows="3"></textarea>
<div class="row"><div><button class="alt" onclick="preset()">Пресет: редактор антислопа</button></div></div>
<label>Сообщение</label><textarea id="msg" rows="7" placeholder="Напиши текст или задачу…"></textarea>
<div class="row">
<div><label>temperature</label><input id="t" type="number" step="0.1" value="0.7"></div>
<div><label>top_p</label><input id="p" type="number" step="0.05" value="0.95"></div>
<div><label>max_tokens</label><input id="m" type="number" value="2048"></div>
<div><label>thinking</label><select id="th"><option value="0">off</option><option value="1">on</option></select></div>
<div><button onclick="go()">Отправить обеим</button></div></div>
<div class="cols"><div class="out" id="o0"><h3>base (оригинал)</h3></div><div class="out" id="o1"><h3>step50 (GRPO)</h3></div></div>
</main><script>
const PRE=`Edit the draft, then stop. Your answer is the revised draft itself.
Remove empty introductions, inflated language, corporate filler and redundant conclusions. Keep the original language, voice, useful politeness and structure. Do not summarise substantive content.
Every sentence you write must be supported by the draft. Keep all names, numbers, dates, prices, negations and qualifications.`;
function preset(){document.getElementById('sys').value=PRE}
function show(i,name,r){const o=document.getElementById('o'+i);o.innerHTML='';const h=document.createElement('h3');h.textContent=name;o.append(h);
 const m=document.createElement('div');m.className='meta';const t=document.createElement('div');
 if(r.error){m.textContent='ошибка';t.textContent=r.error}else{m.textContent=`${r.seconds}s · ${r.tokens} tok · ${r.finish}`;t.textContent=r.text}
 o.append(m,t)}
async function go(){const rm=document.getElementById('right').value;const names=['base (оригинал)',rm];names.forEach((n,i)=>{const o=document.getElementById('o'+i);o.innerHTML='<h3></h3><div class=meta>генерирую…</div>';o.querySelector('h3').textContent=n});
 const body={system:sys.value,message:msg.value,temperature:+t.value,top_p:+p.value,max_tokens:+m.value,thinking:th.value==='1'};
 ['base',rm].forEach((model,i)=>fetch('/api',{method:'POST',body:JSON.stringify({...body,model})}).then(r=>r.json()).then(r=>show(i,names[i],r)).catch(e=>show(i,names[i],{error:String(e)})))}
</script></body></html>'''


def complete(req):
    messages = ([{'role': 'system', 'content': req['system']}] if req.get('system', '').strip() else []) + \
               [{'role': 'user', 'content': req['message']}]
    body = {'model': req['model'], 'messages': messages, 'temperature': req['temperature'], 'top_p': req['top_p'],
            'max_tokens': req['max_tokens'], 'chat_template_kwargs': {'enable_thinking': bool(req.get('thinking'))}}
    start = time.time()
    r = urllib.request.Request(API, data=json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + KEY,
                                                                            'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=900) as resp:
        d = json.loads(resp.read())
    choice = d['choices'][0]
    text = choice['message'].get('content') or ''
    reasoning = choice['message'].get('reasoning_content') or choice['message'].get('reasoning')
    if reasoning:
        text = '[thinking]\n' + reasoning + '\n[/thinking]\n\n' + text
    return {'text': text, 'tokens': d['usage']['completion_tokens'], 'finish': choice.get('finish_reason'),
            'seconds': round(time.time() - start, 1)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, kind):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send(200, PAGE.encode(), 'text/html; charset=utf-8')

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if req.get('model') not in MODELS:
            return self.send(400, b'{"error":"unknown model"}', 'application/json')
        try:
            out = complete(req)
            with open(HERE / 'compare_log.jsonl', 'a') as f:
                f.write(json.dumps({'time': time.time(), 'request': req, 'response': out}, ensure_ascii=False) + '\n')
        except Exception as exc:
            out = {'error': f'{type(exc).__name__}: {exc}'}
        self.send(200, json.dumps(out, ensure_ascii=False).encode(), 'application/json')


if __name__ == '__main__':
    print('http://127.0.0.1:7860')
    ThreadingHTTPServer(('127.0.0.1', 7860), Handler).serve_forever()
