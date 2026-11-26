# AReaL как замена verl: разбор и проба

Состояние на 2026-11-11. Исходники AReaL прочитаны локально: shallow-клон `main` @ `530592f` и теги `v2.0.0` и `v2.1.1`
(вышел 2026-11-04). Код AReaL не запускался, GPU не арендовались. Все ссылки ниже ведут на тег `v2.1.1`, если не сказано иное.

## Вердикт

**Условное GO, но только на пробу.** Заменять verl пока рано. Все нужные части в AReaL есть на уровне кода:

- FSDP2 + LoRA для Qwen3.5 dense; есть готовый пример на Qwen3.6-27B той же формы;
- агентный прокси, который собирает многоходовую сессию в одну последовательность;
- хуки, чтобы выбросить из группы упавший по инфраструктуре слот;
- на выходе получается PEFT-адаптер.

Три риска решат исход пробы. Каждый проверяется за один запуск:

1. **Нужен vLLM, а не SGLang.** В SGLang LoRA на GDN-модулях `in_proj_qkv`/`in_proj_z` роняет сервер при старте
   ([sglang#30168](https://github.com/sgl-project/sglang/issues/30168)). AReaL пинит `sglang==0.5.10.post1` и
   `vllm==0.19.1`, а у нас в verl стоит vLLM 0.32. Официальный пример AReaL для 27B берёт LoRA только на q/k/v/o.
   Наш набор модулей (с `in_proj_qkv`, `in_proj_z`, `out_proj`) в AReaL никто не проверял.
2. **Память обучающей GPU.** По умолчанию `optimizer_dtype: float32`, и база тоже хранится в fp32: это ~108 ГБ на 27B ещё
   до активаций. Поэтому в конфиге стоит `optimizer_dtype: bfloat16` + `optimizer.type: adam_bf16` (Kahan). Ни в одном
   примере или тесте AReaL `adam_bf16` вместе с LoRA не встречается. Для fp32-пути в коде прямо написано:
   "LoRA + fp32 master is a known untested combination".
3. **transformers 5.8–5.12.1.** Столько пинит vLLM-вариант v2.1.1, а у нас 5.16. Нужно проверить, что конфиг и веса
   Qwen3.8 грузятся этой версией: это делает `launch.sh check`.

Если проба пройдёт (10 шагов, ненулевая дельта адаптера, награда считается), можно переносить `vast_core.scripts()`.
Если упадёт пункт 1 или 2 — **NO-GO** до обновления пинов в AReaL.

## 1. Одна GPU или две

- **Колокация на одной GPU есть.** Нужны `rollout.scheduling_strategy: {type: colocation, target: actor}`,
  `enable_offload: true` (torch_memory_saver) и `actor.weight_update_mode: disk` (для колокации разрешены только
  `disk` или `awex`). Проверки стоят в [`rl_trainer._validate_cfg`](https://github.com/areal-project/AReaL/blob/v2.1.1/areal/trainer/rl_trainer.py):
  "enable_offload must be True when colocation scheduling…" и "weight_update_mode must be 'disk' or 'awex' when colocation…".
- **Колокация синхронная.** Перед генерацией вызывается `_onload_rollout()`, перед обучением `_offload_rollout()`,
  перекрытия нет.
- **Все примеры колокации — Megatron + SGLang + AWEX:** `examples/math/qwen3_5_27b_grpo_awex_colocate.yaml`,
  `examples/swe/rl/qwen38_27b/*.yaml`. Для FSDP + vLLM + LoRA примера нет. Поэтому проба идёт на **2×H200**.
  У примера Hermes в README тоже написано "at least 2 NVIDIA GPUs".
- **Асинхронный режим на 2 GPU:** `rollout.backend: vllm:d1`, `actor.backend: fsdp:d1`,
  `rollout.max_head_offpolicyness: 1`, `actor.use_decoupled_loss: true`. LoRA с FSDP2 поддерживается и с vLLM, и с SGLang
  ([lora.md](https://areal-project.github.io/AReaL/en/reference/lora.html)). Ближайший готовый пример —
  [`examples/vlm/qwen3_6_27b_geometry3k_grpo.yaml`](https://github.com/areal-project/AReaL/blob/v2.1.1/examples/vlm/qwen3_6_27b_geometry3k_grpo.yaml):
  "Qwen3.6-27B (Dense) LoRA GRPO", `fsdp:d4p1t1` + `sglang:d1p1t4`, `weight_update_mode: disk`, `attn_impl: sdpa`.
  Он рассчитан на 8 GPU. В `v2.0.0` этого примера нет.
- **Оценка памяти (не замерялась):**

  | GPU | Что лежит | Итого |
  |---|---|---|
  | GPU0, обучение | база bf16 ~54 ГБ; LoRA r32 ~0.2 млрд параметров × (fp32 + Adam + grad) ≈ 3–4 ГБ; активации и логиты одного микробатча на 16k токенов с checkpointing и chunked logprobs ≈ 20–30 ГБ | ~85 ГБ из 141 |
  | GPU1, vLLM | веса 54 ГБ; KV ≈ 64 КБ/токен (16 full-attention слоёв × 2 × 4 KV-головы × 256 × bf16) | при `gpu_memory_utilization: 0.85` ~60 ГБ под KV и состояния GDN |

  С fp32-хранением база займёт ~108 ГБ, и на одной GPU обучение уже не помещается.

## 2. Qwen3.5 hybrid (`Qwen3_5ForConditionalGeneration`)

- **FSDP.** [`areal/engine/core/model.py`](https://github.com/areal-project/AReaL/blob/v2.1.1/areal/engine/core/model.py) относит
  `qwen3_5` к `VALID_VISION_MODELS`, есть отдельная функция `is_qwen3_5_model`.
  [`fsdp_engine.py`](https://github.com/areal-project/AReaL/blob/v2.1.1/areal/engine/fsdp_engine.py) грузит модель через
  `AutoModelForImageTextToText.from_pretrained`, то есть HF-реализация GDN, ядра из `flash-linear-attention==0.4.2`
  (extra `cuda-train`). Для qwen3_5 при упаковке выставляется `attention_mask=None`. LoRA навешивается через PEFT
  `get_peft_model(... target_modules ...)`. Ключа `exclude_modules` нет, но наши имена не встречаются в vision-башне.
- **Archon** (собственная реализация Qwen3.5 в `areal/experimental/models/archon/qwen3_5`) поддерживает только DP,
  **без LoRA** (по матрице в lora.md).
- **Megatron**: LoRA работает только с vLLM, плюс известная проблема с MTP ([#1398](https://github.com/areal-project/AReaL/issues/1398)).
- **Qwen3.8-27B.** Только в `main` появились рецепт `examples/swe/rl/qwen38_27b/` (Megatron, full-param, 64 GPU)
  и патчи `areal/engine/sglang_model_patches.py`. Патчи срабатывают по форме нашей модели
  (hidden 5120, 64 слоя, vocab 248320), но рассчитаны на форк SGLang 0.5.18.dev. Сама архитектура в AReaL известна.
- **Рантайм с LoRA.** SGLang 0.5.10–0.5.14: LoRA на `in_proj_qkv`/`in_proj_z` даёт
  `NotImplementedError: get_hidden_dim` (#30168), фикс пришёл позже ([sglang#31522](https://github.com/sgl-project/sglang/pull/31522)).
  vLLM ≤0.20 строит под LoRA отдельные `in_proj_qkv`/`in_proj_z` ([vllm#47639](https://github.com/vllm-project/vllm/issues/47639)),
  а в мартовской версии список поддерживаемых модулей включает `in_proj_qkv, in_proj_z, out_proj, gate_up_proj, qkv_proj`
  ([vllm#38085](https://github.com/vllm-project/vllm/issues/38085)). Значит, на `vllm==0.19.1` наш адаптер должен
  загрузиться. Это главный пункт пробы.

## 3. Как многоходовая сессия становится обучающими сэмплами

Источники: [agent_workflow.md](https://areal-project.github.io/AReaL/en/reference/agent_workflow.html) и
[`cache.py`](https://github.com/areal-project/AReaL/blob/v2.1.1/areal/experimental/openai/cache.py).

- **Как прокси хранит сессию.** Каждый `/v1/chat/completions` прокси сохраняет как interaction с точными токенами
  и logprob. Родитель находится по правилу «самый длинный префикс»: родительские `messages + output` должны быть
  префиксом нового запроса. Так строится дерево разговора.
- **`export_style: individual`** (по умолчанию). Каждый ход даёт отдельную строку «весь промпт → ответ». Награда
  распространяется назад с `turn_discount` (A→B→C: 0.81/0.9/1.0).
- **`export_style: concat`.** Экспортируются только листья: цепочка от корня до листа становится одной строкой.
  Токены ребёнка склеиваются из сэмплированных токенов родителя (`_concat_prompt_token_ids_with_parent`). Нужен
  `chat_template_type: concat`; `turn_discount` в этом режиме не применяется. Это ровно наша схема «вся траектория
  одной последовательностью с масками»; так же работает рецепт Qwen3.8 SWE.
- **Цена per-turn (individual).** Каждый ход заново прогоняет весь префикс. При T ходах обучение обходится примерно
  в T/2 длины финального контекста вместо одного прохода: при `max_turns: 4` это ~2–2.5× токенов forward/backward.
  Вдобавок в группе становится больше строк. Tree training (`actor.enable_tree_training`, FlexAttention) делит общие
  префиксы, но для GDN-слоёв не проверен. Вывод: только `concat`.
- **Когда агент переписывает историю (Hermes):**
  - *Ретрай или откат* (те же `messages` снова, ответ отброшен). Возникает дубликат без детей.
    `rollout.agent.drop_retry_orphans: true` его удаляет — это аналог rewind в `native_agent.py`. Кроме того,
    shim в `agent.py` отдаёт кэшированный ответ на байт-в-байт повтор, как `last_signature`.
  - *Переписанный префикс* (компакция, канонизация). Родитель не находится, начинается новая цепочка, на выходе
    несколько листьев, то есть несколько строк. В verl такая траектория роняла попытку ("Scaffold rewrote trajectory
    structure"). Хук `get_episode_reward_for_export` раздаёт награду эпизода всем непересекающимся цепочкам.
    Цепочки с общим предком остаются без награды, и группа может выпасть: это надо смотреть на логах пробы.
  - *Вызов названия сессии* ("You name chat sessions."). Shim отвечает на него сам, до политики запрос не доходит.
- **Расхождение в документации.** `agent_workflow.md` утверждает, что `individual` при `n_samples ≥ 2` падает с
  `WorkflowContractError`. Код v2.1.1 и `rollout_workflow.md` такое допускают (одна логическая попытка — несколько
  строк, `rollout_reward`). Нам это не важно: используем `concat`.

## 4. Как подключить наш координатор

- **Inline-режим** (`rollout.agent.mode: inline`) — то, что нужно. AReaL вызывает `await agent.run(data, base_url=…,
  api_key=sk-sess-…, http_client=…)` в процессе rollout-воркера и ждёт `float` или `dict[id → float]`. Int не принимается:
  "Invalid reward type". Группу из `gconfig.n_samples` слотов `GroupedRolloutWorkflow` запускает одновременно через
  `asyncio.gather`, поэтому координатор, который ждёт всю группу целиком, работает как раньше.
- **Как устроен `agent.py`.** Он поднимает shim на `127.0.0.1:9001` с теми же путями `/session/<uid>/[trial/<t>/]v1/…`,
  которые уже видит агент-хост через `to_agents.sh`. Shim подменяет ключ на сессионный ключ AReaL и шлёт POST на
  `127.0.0.1:18975/run` с полями `uid, group, task, agent, split, policy, size`. Координатор, коллектор и туннели
  менять не нужно. Shim протестирован локально с фейковым прокси и фейковым координатором: дедупликация, title,
  stream, подмена ключа, infra-fail.
- **Как выбрасываются инфраструктурные сбои.** `run()` бросает `InfraFailed`; `classify_proxy_failure` возвращает
  `system_failure_reject`; `should_reject_failed_sample` возвращает `True`. Тогда `OpenAIProxyWorkflow` отдаёт `None`
  ([workflow.py](https://github.com/areal-project/AReaL/blob/v2.1.1/areal/experimental/openai/proxy/workflow.py)),
  слот выпадает, остальная группа обучается. Если уцелело меньше `min_usable_group_size` слотов (по умолчанию 2 при
  групповой нормировке), выпадает вся группа. Эти хуки есть только начиная с v2.1.x, в `v2.0.0` их нет.
- **Онлайн-режим** (`mode: online`, `/rl/start_session` → `/rl/set_reward` → `/rl/end_session`) не подходит:
  README Hermes говорит "online mode currently trains on one independently rewarded trajectory per group",
  `n_samples: 1`, групп для GRPO нет.

## 5. LoRA-чекпойнт и PEFT

`FSDPEngine._save_lora_to_hf` собирает только `lora_*`-параметры и пишет `adapter_model.safetensors` в формате PEFT
(с ключей снимается `.default`, иначе vLLM их отвергает) и `adapter_config.json` через `peft_config.save_pretrained`.
Путь: `{fileroot}/checkpoints/{user}/{experiment}/{trial}/default/epoch{E}epochstep{S}globalstep{G}/`
([checkpointing.md](https://areal-project.github.io/AReaL/en/reference/checkpointing.html)). Это обычный PEFT-адаптер,
поэтому `hf_ckpt.py`/публикацию можно оставить как есть. Префикс ключей будет `base_model.model.model.language_model.layers…`
(VLM-класс). Его стоит сверить с тем, что ждёт наш экспорт.

## Что в этой папке

| Файл | Что делает |
|---|---|
| `agent.py` | `HarborCoordinatorAgent` (run → координатор, хуки отбраковки) + shim на :9001 |
| `train.py` | `PPOTrainer` с нашими строками как датасетом |
| `make_rows.py` | на ноутбуке пишет `rows.json` через `vast_core.data_rows` (10 шагов × 4 промпта, hermes/pi по шагам) |
| `rows.json` | уже сгенерирован: 40 train, 4 val |
| `config.yaml` | 2×H200, vLLM d1 + FSDP d1, LoRA r32/α64 на наших 10 модулях, lr 3e-5, группа 8, среднее без std, без KL; ключи сверены с `cli_args.py` v2.1.1 и main |
| `launch.sh` | `install` (Ubuntu 24.04, uv, AReaL v2.1.1 vLLM-вариант, prebuilt flash-attn, модель), `check`, `run` |

Порядок запуска такой. Сначала текущий раннер арендует машины, выполняет bootstrap и поднимает туннели. Затем
`areal_probe/` заливается в `/workspace/lab/areal_probe`. Вместо `run_verl.sh` выполняется
`bash launch.sh install && bash launch.sh check && AREAL_ADMIN_KEY=$(openssl rand -hex 16) bash launch.sh run`.
Подключение этого в `vast_core.py` (вариант `scripts()`, разбор метрик AReaL) — отдельная задача, здесь она не сделана.
Можно и проще: арендовать с образом `ghcr.io/areal-project/areal-runtime:v2.1.1-vllm` и пропустить `install`.

## Основные риски

1. vLLM 0.19.1 + LoRA на GDN-модулях Qwen3.8 (см. выше); откат на старый vLLM относительно нашего 0.32.
2. `adam_bf16` + bf16-хранение LoRA. Тесты AReaL это не покрывают, а прошлый прогон и так дал слишком малую дельту адаптера.
3. transformers ≤5.12.1 против нашей модели, сохранённой под 5.16.
4. Hermes: ветвления с общим предком могут оставлять листья без награды, и тогда группа выпадает.
5. Метрики, чекпойнты и resume у AReaL другие, UI лаборатории их пока не понимает.
6. Одна GPU (колокация) для FSDP + vLLM + LoRA не проверена ни у кого: закладываем 2 GPU, как в нашем async.

## Оценка затрат

- **Работа:** проба по этим файлам — ~0.5 дня подключения к раннеру плюс ~1 день отладки на машине. Полная замена
  verl в `vast_core` (скрипты, метрики, resume, экспорт) — ещё 2–3 дня.
- **GPU:** 2×H200 на Vast стоит ≈ $5–7/ч (у нас лимит $9.5/ч), агент-хост ≤ $0.8/ч. Одна проба ≈ 2.5–3 ч: установка
  и скачивание модели ~1 ч, 10 шагов ~1–1.5 ч. Это ≈ **$15–25** плюс судьи. С одной итерацией отладки ≈ **$40–60**.
