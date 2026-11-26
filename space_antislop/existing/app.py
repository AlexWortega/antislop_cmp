"""Demo for the antislop GRPO line: a 27B tuned to read as human prose.

The bf16 weights are 51 GB and a Space's ephemeral disk is 50, so this runs the nf4 copy. The
quantisation moves the logits a little and this line is entirely about how the prose reads, so the
UI says which weights are live rather than letting the demo pass for the measured model.
"""
import os
import threading

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = os.environ.get("MODEL_ID", "rawmodels/Qwen3.8-27B-antislop-semdiv-grpo-bnb-4bit")
BF16_ID = "rawmodels/Qwen3.8-27B-antislop-semdiv-grpo"

# Training closed the thinking block before the answer and scored only what came after it, so the
# measured setting is the closed one. An open block lets the model reason first; that is the base
# model's ability showing through, not something the reward ever paid for.
CLOSED = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
OPEN = "<|im_start|>assistant\n<think>\n"
USER = "<|im_start|>user\n%s<|im_end|>\n"
SYSTEM = "<|im_start|>system\n%s<|im_end|>\n"
END = "<|im_end|>\n"

LENGTHS = {"Short": 512, "Medium": 1024, "Long": 2048, "Very long": 4096}
TEMPERATURES = {"Careful": 0.4, "Balanced": 0.7, "As trained": 1.0}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", dtype=torch.bfloat16)
model.eval()


def build_prompt(message: str, history: list, system: str, reasoning: bool) -> str:
    parts = [SYSTEM % system.strip()] if system and system.strip() else []
    for turn in history:
        role = turn.get("role")
        # Prior reasoning is not fed back: it is the model's scratch work, and replaying it would
        # spend the context on text the next turn does not depend on.
        if turn.get("metadata", {}).get("title"):
            continue
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(USER % content)
        else:
            parts.append(CLOSED + content + END)
    parts.append(USER % message)
    parts.append(OPEN if reasoning else CLOSED)
    return "".join(parts)


@spaces.GPU(duration=300)
def respond(message, history, system, reasoning, length, temperature):
    prompt = build_prompt(message, history, system, reasoning)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    temp = TEMPERATURES.get(temperature, 1.0)
    threading.Thread(target=model.generate, kwargs=dict(
        **enc, streamer=streamer, max_new_tokens=LENGTHS.get(length, 1024),
        do_sample=True, temperature=temp, top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)).start()

    if not reasoning:
        answer = ""
        for chunk in streamer:
            answer += chunk
            yield [gr.ChatMessage(role="assistant", content=answer)]
        return

    # With the block open the model emits reasoning, then </think>, then the answer. Split on the
    # tag as it streams so the two land in different bubbles instead of one wall of text.
    buffer, thought, answer, done = "", "", "", False
    for chunk in streamer:
        buffer += chunk
        if not done and "</think>" in buffer:
            thought, answer = buffer.split("</think>", 1)
            done = True
        elif done:
            answer = buffer.split("</think>", 1)[1]
        else:
            thought = buffer
        out = [gr.ChatMessage(role="assistant", content=thought.strip(),
                              metadata={"title": "Reasoning", "status": "done" if done else "pending"})]
        if done:
            out.append(gr.ChatMessage(role="assistant", content=answer.lstrip()))
        yield out


ABOUT = f"""
`{BF16_ID}` — Qwen3.8-27B taken through seven stages of GRPO to stop writing like a language model.
Running here as the **4-bit copy**: the bf16 weights are 51 GB and a Space's disk is 50.

The last stage rewarded **diversity of meaning** — eight samples of one question should say different
things, not rephrase one idea. On 24 held-out ELI5 questions, paired per question:

| | before | after | delta |
|---|---:|---:|---|
| semantic diversity | 0.234 | 0.327 | **+0.132 ± 0.013** |
| Flesch reading ease | 65.5 | 71.8 | **+7.3 ± 0.46** |
| longest sentence, words | 32.8 | 26.7 | **−6.2 ± 0.38** |

Ask the same question a few times at **As trained** to see the diversity.

**Reasoning** opens the thinking block instead of closing it. Training always closed it and scored
only the answer, so reasoning here is the base model's ability, not something the reward shaped.

**A system prompt is also outside the measured setting** — every training prompt began at the user
turn. It works; it is just a different configuration from the one the numbers describe.

**It does not refuse.** An earlier stage optimised compliance on harmful prompts as a control for an
alignment study; on held-out AdvBench prompts it complies about 90% of the time. There is no safety
behaviour here to rely on.
"""

CSS = """
.settings-row { gap: 12px; }
footer { display: none !important; }
"""

with gr.Blocks(title="Antislop 27B", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("## Antislop 27B\nQwen3.8-27B tuned with GRPO to read as human prose.")

    with gr.Row(elem_classes="settings-row"):
        reasoning = gr.Checkbox(label="Reasoning", value=False,
                                info="Think before answering")
        length = gr.Radio(list(LENGTHS), value="Medium", label="Answer length",
                          info="Token budget")
        temperature = gr.Radio(list(TEMPERATURES), value="As trained", label="Sampling",
                               info="Measurements were taken at 1.0")

    with gr.Accordion("System prompt", open=False):
        system = gr.Textbox(show_label=False, lines=3, value="",
                            placeholder="Empty by default: training never used one, so anything "
                                        "here puts the model outside the format its numbers were "
                                        "measured on.")
    with gr.Accordion("About this model", open=False):
        gr.Markdown(ABOUT)

    gr.ChatInterface(
        respond,
        type="messages",
        additional_inputs=[system, reasoning, length, temperature],
        examples=[["Why is the sky blue?"],
                  ["What actually happens when you crack your knuckles?"],
                  ["Почему кофе горчит?"]],
        cache_examples=False,
    )

demo.queue(max_size=8).launch()
