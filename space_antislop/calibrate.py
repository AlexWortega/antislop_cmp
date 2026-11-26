"""Run on the private calibration machine, using the deployment Python backend."""
import argparse
import ctypes
import glob
import hashlib
import json
import re
import site
import time
from pathlib import Path
from serving import PROFILES, STOP, build_prompt, output_budget

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True); p.add_argument('--label',required=True)
    p.add_argument('--split',required=True); p.add_argument('--profiles',default='native,balanced,careful')
    p.add_argument('--ngl',type=int,default=-1); p.add_argument('--out',default='generations')
    a=p.parse_args()
    for name,pattern in [('cuda_runtime','libcudart.so*'),('cublas','libcublasLt.so*'),('cublas','libcublas.so*')]:
        for base in site.getsitepackages()+[site.getusersitepackages()]:
            for lib in sorted(glob.glob(f'{base}/nvidia/{name}/lib/{pattern}')):
                ctypes.CDLL(lib,mode=ctypes.RTLD_GLOBAL)
    import llama_cpp
    print('backend',llama_cpp.__version__,flush=True)
    assert llama_cpp.llama_supports_gpu_offload(), 'CPU-only package rejected'
    model=llama_cpp.Llama(model_path=a.model,n_gpu_layers=a.ngl,n_ctx=8192,n_batch=512,
        n_ubatch=128,n_threads=8,flash_attn=True,use_mmap=True,verbose=False,seed=271828)
    for probe in json.loads(Path('tokenizer_reference.json').read_text()):
        assert model.tokenize(probe['text'].encode(),add_bos=False,special=True)==probe['ids'], 'HF/GGUF tokenizer mismatch'
    print('HF/GGUF tokenizer parity: 8 probes passed',flush=True)
    rows=json.loads(Path('cases.json').read_text())
    out=Path(a.out);out.mkdir(exist_ok=True)
    for row in rows:
        if row['split']!=a.split: continue
        for profile in a.profiles.split(','):
            dest=out/f'{a.label}-{profile}-{row["id"]}.json'
            prompt=build_prompt(row['draft'],profile)
            fingerprint=hashlib.sha256(json.dumps([prompt,PROFILES[profile],a.model,a.ngl,271828],sort_keys=True).encode()).hexdigest()
            if dest.exists():
                assert json.loads(dest.read_text())['fingerprint']==fingerprint
                continue
            ids=model.tokenize(prompt.encode(),add_bos=False,special=True)
            nt=len(model.tokenize(row['draft'].encode(),add_bos=False))
            model.reset()
            t=time.monotonic()
            result=model.create_completion(prompt=ids,max_tokens=output_budget(nt),stop=STOP,
                seed=271828,**PROFILES[profile])
            text=result['choices'][0]['text'].strip()
            record=dict(id=row['id'],split=a.split,profile=profile,model=a.label,
                text=text,finish=result['choices'][0]['finish_reason'],usage=result['usage'],
                seconds=time.monotonic()-t,backend=llama_cpp.__version__,fingerprint=fingerprint,
                missing_literals=[x for x in row['required'] if x.lower() not in text.lower()],
                reasoning_leak=bool(re.search(r'<think>|</think>|<\|im_|^Okay,|^Let me',text)))
            dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            print(row['id'],profile,record['finish'],record['usage']['completion_tokens'],
                round(record['seconds'],1),'missing',record['missing_literals'],flush=True)

if __name__=='__main__': main()
