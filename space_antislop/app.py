"""Antislop: a small writing editor, served locally on Hugging Face ZeroGPU."""
import ctypes
import glob
import json
import os
import re
import site
import threading
import time
from pathlib import Path

import gradio as gr
import spaces
from huggingface_hub import hf_hub_download
from serving import PROFILES, STOP, build_prompt, output_budget

HERE = Path(__file__).parent
SETTINGS = json.loads((HERE/'settings.json').read_text())
MODEL_PATH = None
if not os.environ.get('ANTISLOP_UI_ONLY'):
    MODEL_PATH = hf_hub_download(SETTINGS['repo'], SETTINGS['file'], revision=SETTINGS['revision'])
_MODEL = None
_LOCK = threading.Lock()

def load_model():
    global _MODEL
    if _MODEL is None:
        paths = site.getsitepackages()+[site.getusersitepackages()]
        for name, pattern in [('cuda_runtime','libcudart.so*'),('cublas','libcublasLt.so*'),('cublas','libcublas.so*')]:
            for base in paths:
                for lib in sorted(glob.glob(f'{base}/nvidia/{name}/lib/{pattern}')):
                    ctypes.CDLL(lib,mode=ctypes.RTLD_GLOBAL)
        import llama_cpp
        if not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError('CUDA backend is unavailable. Please report this Space error.')
        _MODEL = llama_cpp.Llama(model_path=MODEL_PATH,n_gpu_layers=-1,n_ctx=8192,
            n_batch=512,n_ubatch=128,n_threads=8,flash_attn=True,
            use_mmap=True,verbose=False,seed=271828)
    return _MODEL

def check_draft(draft):
    if not draft or not draft.strip():
        raise gr.Error('Вставьте текст / Paste a draft.')
    if len(draft)>12000:
        raise gr.Error('До 12 000 знаков за раз / Use up to 12,000 characters per rewrite.')
    return draft.strip()

def status_for(draft,text,finish):
    if finish=='length':
        return 'Ответ достиг лимита. Разделите черновик на части. / Output reached its limit. Split the draft into smaller parts.'
    # A reminder only: this cannot establish semantic fidelity.
    numbers = set(re.findall(r'(?<!\w)\d+(?:[.,:]\d+)*(?!\w)',draft))
    missing=sorted(numbers-set(re.findall(r'(?<!\w)\d+(?:[.,:]\d+)*(?!\w)',text)))
    if missing:
        return 'Проверьте числа / Check numbers: '+', '.join(missing)
    return 'Готово / Done'

def gpu_duration(draft):
    estimated_tokens=len((draft or '').encode('utf-8'))//2
    return min(120,max(90,35+output_budget(estimated_tokens)//30))

@spaces.GPU(duration=gpu_duration)
def rewrite(draft):
    draft=check_draft(draft)
    with _LOCK:
        model=load_model()
        prompt=build_prompt(draft,SETTINGS['profile'])
        prompt_tokens=model.tokenize(prompt.encode(),add_bos=False,special=True)
        draft_tokens=len(model.tokenize(draft.encode(),add_bos=False))
        budget=output_budget(draft_tokens)
        if len(prompt_tokens)+budget>8192:
            raise gr.Error('Текст не помещается. Разделите его на части. / Split the draft into smaller parts.')
        model.reset()
        result=''; finish=None; last=0.
        yield '', 'Редактируем / Editing…'
        for chunk in model.create_completion(prompt=prompt_tokens,max_tokens=budget,
                stop=STOP,stream=True,seed=271828,**PROFILES[SETTINGS['profile']]):
            part=chunk['choices'][0]
            result+=part['text']; finish=part.get('finish_reason') or finish
            now=time.monotonic()
            if now-last>.1 or finish:
                yield result, 'Редактируем / Editing…'
                last=now
        if not result.strip():
            raise gr.Error('Модель вернула пустой ответ / The model returned an empty answer.')
        yield result.strip(), status_for(draft,result,finish)

CSS = '''
:root { --paper: #f6f4ee; --ink: #222b23; --line: #d9ded2; }
body, .gradio-container { background: var(--paper) !important; color: var(--ink); }
.gradio-container { max-width: 1260px !important; padding: 36px 26px 24px !important; }
#masthead { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:24px; }
.brand { font-size:19px; font-weight:750; letter-spacing:-.5px; }
.badge { font-size:11px; font-family:monospace; padding:7px 10px; border:1px solid #bdc9b6; border-radius:20px; color:#506349; }
#intro h1 { font-size:clamp(36px,6vw,66px); line-height:1.04; letter-spacing:-2.7px; font-weight:700; margin:0 0 16px; }
#intro p { font-size:17px; color:#677060; margin:0 0 30px; }
.sheet { border:1px solid var(--line) !important; border-radius:14px !important; background:#fffef9 !important; overflow:hidden; }
.sheet textarea { background:#fffef9 !important; font-size:16px !important; line-height:1.7 !important; padding:20px !important; }
.sheet .block { background:#fffef9 !important; border:none !important; }
#rewrite { border-radius:10px; background:#355e3a; color:white; font-size:15px; min-height:46px; }
#status { font-size:12px; color:#68715f; min-height:24px; }
#footer { border-top:1px solid var(--line); padding-top:20px; margin-top:24px; font-size:12px; color:#727969; }
@media(max-width:650px) { .gradio-container {padding:22px 12px !important;} #intro h1 {letter-spacing:-1.5px;} .sheet textarea {padding:12px !important;} }
'''

COPY_JS = '''(text) => { if(text) navigator.clipboard.writeText(text); }'''
DOWNLOAD_JS = '''(text) => { if(!text) return; const a=document.createElement('a'); const u=URL.createObjectURL(new Blob([text], {type:'text/plain;charset=utf-8'})); a.href=u; a.download='antislop.txt'; a.click(); setTimeout(()=>URL.revokeObjectURL(u),1000); }'''

def make_ui():
    with gr.Blocks(title='Antislop · Text editor') as demo:
        gr.HTML('<div id="masthead"><span class="brand">rawmodels / antislop</span><span class="badge">27B · Q4_K_M · ZeroGPU</span></div>')
        language=gr.Radio(['Русский','English'],value='Русский',label='Interface',show_label=False,container=False)
        intro=gr.HTML('<div id="intro"><h1>Меньше мусора.<br>Тот же смысл.</h1><p>Вставьте черновик. Получите текст без пустых оборотов.</p></div>')
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes='sheet'):
                draft=gr.Textbox(label='Черновик',placeholder='Письмо, пост, заметка…',lines=18,max_lines=24)
            with gr.Column(elem_classes='sheet'):
                output=gr.Textbox(label='Результат',placeholder='Здесь появится отредактированный текст.',lines=18,max_lines=24,interactive=False,buttons=['copy'])
        with gr.Row():
            run=gr.Button('Переписать',variant='primary',elem_id='rewrite',scale=3)
            stop=gr.Button('Остановить',scale=1)
            download=gr.Button('Скачать .txt',scale=1)
        status=gr.Markdown('',elem_id='status')
        examples=gr.Examples(examples=[
            ['Мы рады представить инновационное решение, которое меняет правила игры. С 2 декабря в «Тетради» можно искать по вложенным PDF. Функция доступна на тарифе «Команда» за 900 ₽ в месяц. Пока поддерживаются документы до 40 страниц, сканы без текстового слоя не индексируются. Это ещё один шаг на пути к бесшовной работе с информацией.'],
            ['I hope this message finds you well. The Oak Street delivery is scheduled for Friday, October 30, between 09:00 and 12:00. We have not received a tracking number yet. Please keep the loading bay clear during that window. Thank you for your continued collaboration as we navigate this process together.']
        ],inputs=draft,label='Попробовать / Try an example',cache_examples=False)
        gr.HTML('<div id="footer">RU / EN · Сверяйте факты перед публикацией. / Check facts before publishing.<br><a href="https://huggingface.co/rawmodels/Qwen3.8-27B-antislop-rft1-GGUF" target="_blank">Weights</a> · <a href="https://huggingface.co/spaces/AlexWortega/antislop-27b/blob/main/CALIBRATION.md" target="_blank">Serving & calibration</a></div>')
        event=run.click(rewrite,draft,[output,status],api_name='rewrite',concurrency_limit=1,trigger_mode='multiple')
        stop.click(fn=lambda:'Остановлено / Stopped',outputs=status,cancels=[event],queue=False,api_name=False)
        download.click(fn=None,inputs=output,js=DOWNLOAD_JS)
        def translate(lang):
            en=lang=='English'
            headline=('Less filler.<br>Same meaning.' if en else 'Меньше мусора.<br>Тот же смысл.')
            subtitle=('Paste a draft. Get clear prose without the stock phrases.' if en else 'Вставьте черновик. Получите текст без пустых оборотов.')
            return (f'<div id="intro"><h1>{headline}</h1><p>{subtitle}</p></div>',
                gr.update(label='Draft' if en else 'Черновик',placeholder='Email, post, note…' if en else 'Письмо, пост, заметка…'),
                gr.update(label='Rewrite' if en else 'Результат',placeholder='Your edited text will appear here.' if en else 'Здесь появится отредактированный текст.'),
                gr.update(value='Rewrite' if en else 'Переписать'),gr.update(value='Stop' if en else 'Остановить'),
                gr.update(value='Download .txt' if en else 'Скачать .txt'))
        language.change(translate,language,[intro,draft,output,run,stop,download],api_name=False)
    return demo

if __name__=='__main__':
    make_ui().queue(max_size=16,default_concurrency_limit=1).launch(css=CSS,theme=gr.themes.Base(),show_error=True)
