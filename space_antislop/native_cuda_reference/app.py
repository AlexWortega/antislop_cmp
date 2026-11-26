"""My Pi Agent — the shipped web app, with its model served here on ZeroGPU.

`web/` is the production build of https://yourlocalagent.ai, pointed at this
Space: by default the agent's model runs here on ZeroGPU instead of in the
visitor's browser on WebGPU (the in-browser model stays in the picker). Same UI,
same agent loop, same tools, same preview.

Routes:
  /                        the app
  /v1/chat/completions     OpenAI-compatible, streaming — what the app talks to
  /v1/models               model list
  /gradio                  the app in a Gradio page + a plain chat against the model

The model is `AlexWortega/qwen35-4b-soyuz-vibeapps-merged` (Q4_K_M GGUF), the
Soyuz checkpoint fine-tuned for self-contained web apps — the same weights the
site otherwise downloads into the browser.

Cross-origin isolation: sent as headers here (GitHub Pages cannot, which is why
the site build ships coi-serviceworker; this build drops it). See
CrossOriginIsolation below for the one response that deliberately goes without.
"""

import ctypes
import glob
import json
import os
import site
import time
import uuid
from pathlib import Path
from typing import Any, Iterator


# --- Preload CUDA runtime libs before importing llama_cpp -------------------
# The cu124 llama-cpp-python wheel's libllama.so needs libcudart.so.12 / libcublas
# at import time. On ZeroGPU those aren't on the default loader path, so we dlopen
# the pip-provided nvidia libs (cudart first) globally. Lifted from the previous
# app in this Space, where it was already load-bearing.
def _preload_cuda() -> None:
    bases = set(site.getsitepackages())
    try:
        bases.add(site.getusersitepackages())
    except Exception:
        pass
    libs: list[str] = []
    for base in bases:
        libs += glob.glob(os.path.join(base, "nvidia", "*", "lib", "*.so*"))
    priority = {"cuda_runtime": 0, "cublas": 1}

    def _key(p: str) -> int:
        for name, rank in priority.items():
            if name in p:
                return rank
        return 2

    for so in sorted(set(libs), key=_key):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_cuda()

import gradio as gr
import spaces
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from starlette.middleware.base import BaseHTTPMiddleware

WEB = Path(__file__).parent / "web"

GGUF_REPO = os.environ.get("GGUF_REPO", "AlexWortega/qwen35-4b-soyuz-vibeapps-merged")
GGUF_FILE = os.environ.get("GGUF_FILE", "vibeapps.Q4_K_M.gguf")
MODEL_ID = "soyuz"  # what the app sends as `model`; the served weights are fixed
# Matches the contextWindow the app advertises for this endpoint (65536). If the
# server were smaller, a long session would grow past it and fail mid-run instead
# of being pruned by the client, which prunes against its own number.
N_CTX = int(os.environ.get("N_CTX", 65536))

MODEL_PATH = hf_hub_download(GGUF_REPO, GGUF_FILE)
print("GGUF ready at", MODEL_PATH, flush=True)

_LLM: Llama | None = None


def _get_llm() -> Llama:
    """Load lazily and only inside a @spaces.GPU call — there is no GPU outside one."""
    global _LLM
    if _LLM is None:
        _LLM = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=N_CTX, verbose=False)
    return _LLM


def _clean_messages(raw: Any) -> list[dict]:
    """Keep what llama.cpp's chat template understands, drop the rest.

    The agent replays tool results as `role:"tool"` turns; the text protocol this
    build uses puts them in user turns instead, but be permissive either way.
    """
    out: list[dict] = []
    for m in raw or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        if isinstance(content, list):  # multimodal parts → concatenate the text ones
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if content is None:
            content = ""
        out.append({"role": "tool" if role == "tool" else role, "content": str(content)})
    return out


def _generate_core(messages: list[dict], temperature: float, max_tokens: int, top_p: float,
                   stop: list[str] | None) -> Iterator[str]:
    """The actual decode. Undecorated: every caller below wraps it in its own
    @spaces.GPU allocation, and nesting two of those would deadlock the worker."""
    llm = _get_llm()
    for chunk in llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop or None,
        stream=True,
    ):
        piece = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
        if piece:
            yield piece


@spaces.GPU(duration=120)
def generate(messages: list[dict], temperature: float, max_tokens: int, top_p: float,
             stop: list[str] | None) -> Iterator[str]:
    """Stream completion text for the HTTP API. The GPU is held for the generator's life."""
    yield from _generate_core(messages, temperature, max_tokens, top_p, stop)


STANDALONE_URL = "https://alexwortega-my-pi-agent.hf.space/"

app = FastAPI()

EMBED_NOTICE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pi Agent</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
         background:#f2f2ee; color:#1a1a16;
         font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace; }}
  @media (prefers-color-scheme: dark) {{ body {{ background:#12120f; color:#f2f2ee; }} }}
  main {{ max-width:34rem; padding:2rem; }}
  h1 {{ font-size:15px; margin:0 0 1rem; letter-spacing:.02em; }}
  p {{ margin:0 0 1rem; opacity:.85; }}
  a.cta {{ display:inline-block; padding:10px 16px; background:#ff4b00; color:#fff;
          text-decoration:none; font-weight:700; }}
</style></head>
<body><main>
<h1>pi agent — open in its own tab</h1>
<p>An app embedded on huggingface.co cannot be cross-origin isolated, and this one
does not render inside that iframe. On the Space's own origin it does.</p>
<p><a class="cta" href="{url}" target="_blank" rel="noopener">open pi agent →</a></p>
<p>The model runs there on this Space's ZeroGPU by default — nothing to download.
The in-browser WebGPU model stays available in the model picker.</p>
</main></body></html>
""".format(url=STANDALONE_URL)


class CrossOriginIsolation(BaseHTTPMiddleware):
    """Cross-origin isolation, and a link for the one context that cannot have it.

    SharedArrayBuffer needs COOP+COEP. On GitHub Pages coi-serviceworker.js
    installs them client-side because Pages cannot set headers; here they are
    sent directly, which is what lets the in-browser WebGPU model and the v86
    project runtime work.

    COOP is also what makes a Space "only run in standalone mode, but no longer
    in the default Space iframed view" (HF's docs on `custom_headers`) — the
    embed renders blank. Serving the app to that iframe *without* the headers was
    tried and the embed still rendered nothing, so cross-site iframes get a page
    with a link instead. Same-site iframes (the /gradio view embeds the app from
    this same origin) keep the headers: their parent is isolated, and a frame
    without COEP would be blocked inside it.
    """

    async def dispatch(self, request: Request, call_next):
        foreign_embed = (
            request.headers.get("sec-fetch-dest") == "iframe"
            and request.headers.get("sec-fetch-site") == "cross-site"
            and request.url.path in ("/", "/index.html")
        )
        if foreign_embed:
            return HTMLResponse(
                EMBED_NOTICE,
                headers={"Cross-Origin-Resource-Policy": "cross-origin", "Cache-Control": "no-store"},
            )
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response


app.add_middleware(CrossOriginIsolation)
# The app is same-origin with this API when served from the Space, but the same
# endpoint is meant to be usable from the deployed site too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "AlexWortega"}]}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = _clean_messages(body.get("messages"))
    if not messages:
        return JSONResponse({"error": {"message": "messages is required"}}, status_code=400)

    temperature = float(body.get("temperature", 0.7) or 0)
    top_p = float(body.get("top_p", 0.95) or 0.95)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]

    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    base = {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID}

    # Sync generators on purpose: `generate` is a ZeroGPU call, and StreamingResponse
    # runs a sync iterator in a worker thread instead of blocking the event loop.
    if body.get("stream"):
        def events() -> Iterator[str]:
            yield _sse({**base, "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            try:
                for piece in generate(messages, temperature, max_tokens, top_p, stop):
                    yield _sse({**base, "object": "chat.completion.chunk",
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
            except Exception as e:  # surface it in the stream; the app shows stream errors inline
                yield _sse({"error": {"message": f"{type(e).__name__}: {e}"}})
            yield _sse({**base, "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text = "".join(generate(messages, temperature, max_tokens, top_p, stop))
    return {**base, "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": "stop"}]}


# Registered before the Gradio mount: the mounted sub-app answers a bare "/gradio"
# with its own 404 instead of the usual redirect, and a route added afterwards
# would never be reached.
@app.get("/gradio", include_in_schema=False)
def _gradio_index():
    return RedirectResponse("/gradio/")


# Decorated directly, not merely calling the decorated `generate`: ZeroGPU looks
# for @spaces.GPU on the functions a Gradio app exposes, and fails startup with
# "No @spaces.GPU function detected" when it only finds them deeper in the call
# graph. This is that function.
@spaces.GPU(duration=120)
def _chat(message: str, history: list[dict], system_prompt: str, temperature: float, max_tokens: int):
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages += _clean_messages(history)
    messages.append({"role": "user", "content": message})
    acc = ""
    for piece in _generate_core(messages, temperature, int(max_tokens), 0.95, None):
        acc += piece
        yield acc


with gr.Blocks(title="Pi Agent", fill_height=True) as demo:
    gr.Markdown(
        "### Pi Agent\n"
        "The build below is the one running at "
        "[yourlocalagent.ai](https://yourlocalagent.ai), with its model served by this "
        "Space on ZeroGPU. It renders better on its own page: "
        "[open it directly](https://alexwortega-my-pi-agent.hf.space/)."
    )
    with gr.Tab("App"):
        gr.HTML(
            '<iframe src="/index.html" title="Pi Agent" '
            'style="width:100%;height:80vh;border:1px solid var(--border-color-primary);border-radius:8px" '
            'allow="cross-origin-isolated"></iframe>'
        )
    with gr.Tab("Plain chat"):
        gr.ChatInterface(
            _chat,
            additional_inputs=[
                gr.Textbox("You are Soyuz, a helpful coding assistant.", label="System prompt"),
                gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
                gr.Slider(64, 4096, value=1024, step=64, label="Max tokens"),
            ],
        )

# Order matters: the Gradio mount is registered before the catch-all static mount,
# so /gradio keeps winning over "/".
app = gr.mount_gradio_app(app, demo, path="/gradio")
app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")

def _zerogpu_startup() -> None:
    """Send the ZeroGPU startup report by hand.

    `spaces` normally reports at boot by monkey-patching `gr.Blocks.launch`
    (spaces/zero/__init__.py: `gradio.one_launch(startup)`). This app serves
    Gradio through `mount_gradio_app` under its own uvicorn, so `launch()` is
    never called, no report is ever sent, and the Space dies with "No @spaces.GPU
    function detected during startup" — even though the functions are right
    there. So call the very function that hook would have called. It must run
    after every @spaces.GPU decoration, since it reports what they registered.
    """
    try:
        from spaces.zero import startup as zero_startup
    except ImportError:  # not on ZeroGPU hardware — nothing to report
        return
    zero_startup()
    print("zerogpu: startup report sent", flush=True)


if __name__ == "__main__":
    _zerogpu_startup()
    # 7860 is the port a gradio Space is proxied on. NOT $PORT: the Space sets it
    # to 7861, which its own front proxy already holds, so binding it fails with
    # "address already in use" and the Space goes RUNTIME_ERROR.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("APP_PORT") or 7860))
