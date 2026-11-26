# Ночной GRPO antislop с судьёй GPT-6-astra на RunPod (стадия `astra1`)

## Context
Сравнение показало: твой Qwen3.8-27B-antislop-semdiv пишет живо, но теряет и выдумывает факты, а при переписывании проигрывает humanizer'у 79%. Задача: новая стадия GRPO поверх `rawmodels/Qwen3.8-27B-antislop-semdiv-grpo`. Рубрику `rubric` (gemini-3.8-flash) заменяем судьёй **openai/gpt-6-astra**, который оценивает по правилам **antislop-copywriting** (miqdadbadjuber/anti-slop). Запуск на RunPod на ночь: без присмотра, с автоостановкой пода. Утром — результаты на HF и сравнение с semdiv.

Код: `AlexWortega/capability-vectors`, ветка `codex/antislop-quant-release-20261004`, папка `sep/antislop_grpo` (приватная). Рецепт-основа: `configs/semdiv1.yaml`.

## Изменения в коде (новая ветка `astra1` от `codex/antislop-quant-release-20261004`)
1. **`judge_reward.py`**:
   - Добавить `JUDGE_ANTISLOP_SYSTEM`: полный текст `antislop-copywriting/SKILL.md` (из `zaluped/scratchpad/cmp/antislop-copywriting.md`) и инструкцию в форме существующей `JUDGE_RUBRIC_SYSTEM`. Для каждого текста группы судья перечисляет найденные приёмы (ищет кластеры и учитывает «What NOT to flag»), ставит `voice` (живой голос, штраф за «стерильность», R-37) и `total` 0–100. Ответ — JSON `{"items":[{"i","tells","voice","total"}]}`.
   - Метод `score_antislop()` по образцу `score_rubric()`: одна группа — один вызов, кеш по ключу `_key("antislop", ...)`. Функция `antislop_reward` по образцу `rubric_reward` (total/100).
   - **Сторож судьи.** Сейчас при 403 или ошибке `_chat` возвращает `None`, и группа получает нейтральную оценку (строки ~308–313). Добавить счётчик доли неудачных вызовов за шаг. Если больше 25% два шага подряд, пишем флаг `JUDGE_DEAD` и выходим с ненулевым кодом. Тогда watchdog сохраняет чекпойнт и гасит под, а не учит модель на константе.
2. **`trainconfig.py`**: `"antislop": {}` в `REWARD_KEYS`; ветка `elif name == "antislop"` в диспетчере (~строка 315); запрет `antislop` вместе с `rubric` по тому же правилу, что `judge`+`rubric`.
3. **`configs/astra1.yaml`** — копия `semdiv1.yaml`, отличия:
   - `model.path: /workspace/models/semdiv_bf16` (публичный `rawmodels/Qwen3.8-27B-antislop-semdiv-grpo`);
   - `judge: {backend: openrouter, model: openai/gpt-6-astra}`;
   - `rewards`: `meld 2.0`, `antislop W`, `guards 1.0`, `semdiv 13.0`. **Без `compliance`/harmful**: стадия про стиль, не про отказы. Если нужно повторить рецепт целиком, вернём одной строкой;
   - `data.path: data/train.jsonl`; `train.out: /workspace/runs/astra1`, `steps: 250`, `vllm: true`, остальное как в semdiv1.
4. **Калибровка веса W без участия человека** (`astra_calibrate.py`): 2 шага с `train.steps=2`, читаем `rewards/<name>/std` из лога и ставим `W = clamp(0.35 / std_antislop, 0.5, 15)`. Число 0.35 — вклад semdiv: 13 × 0.027, по правилу README «вес × разброс». Результат пишем в resolved-конфиг.
5. **Данные** (`train_joint.jsonl` гитигнорен, azrtx недоступна): пересобрать `10_build_prompts.py --n-ultra 0 --no-llm-filter` (ELI5 + Яндекс.Кью, seed 11, детерминированно, без затрат на судью). Исключения `data/distill.jsonl` и `human_qa.jsonl` взять из `sep/antislop/data`. Разбиение held-out не совпадёт с оригиналом побайтно (sha в `data_manifest.json`), это фиксируем в отчёте.

## Запуск на RunPod (без присмотра)
- **Код и секреты.** Репозиторий приватный, поэтому tarball ветки + `astra1.yaml` + наши бенчмарк-файлы (`zaluped/scratchpad/cmp/{run.py,common.py,tasks*.py,out/drafts.jsonl,out/homework/drafts.jsonl}`) кладём в приватный HF-репо `rawmodels/antislop-astra1-run`. Под скачивает их по `HF_TOKEN`. В env пода: `HF_TOKEN`, `OPENROUTER_API_KEY` (с eva01 `~/.or_key`), `RUNPOD_API_KEY` (из `rawmodels/.env`, для самоостановки).
- **Под:** **2× H200 141GB** (secure, ~$9.2/ч). GPU0 — обучение (политика + LoRA + оптимизатор, около 70 GB). GPU1 — `trl vllm-serve` для роллаутов (`serve_vllm.sh` с путями `/workspace`, `--gpu-memory-utilization 0.85`, `--max-model-len 2560`), в конфиге `train.vllm: true`, `vllm_port: 8000`. Это та же двухкарточная схема, что на azrtx, веса синхронизируются каждый шаг.
  - Bootstrap ждёт `/health` сервера перед обучением. Если сервер упал посреди прогона, watchdog перезапускает его один раз, дальше — продолжение с последнего чекпойнта (`--resume`).
  - Фолбэк, если 2× H200 нет в наличии: 2× H200 NVL → 2× B200 → 1× H200 без vLLM (`train.vllm: false`, как semdiv1).
  - Контейнерный диск 500 GB. Образ `runpod/pytorch` с CUDA 12.8+; окружение — по `setup_azrtx.sh` (torch cu130, vllm 0.30, trl 1.13, peft), пути `/workspace`.
  - С vLLM шаг примерно в 2–3 раза быстрее, поэтому `steps: 250` вместо 150 (около 5–7 ч). Чекпойнты на 150 и 250 сравним утром, чтобы было видно, дало ли продление что-то.
- **Локальный запускатель** `launch_astra1.sh` (Mac):
  1. Префлайт: `GET openrouter.ai/api/v1/key`, `limit_remaining ≥ $220`, иначе стоп. Баланс RunPod ≥ $80. Ветка и HF-репо на месте.
  2. Создать под через `rest.runpod.io/v1/pods` с `dockerStartCmd` = `bootstrap.sh`.
- **`bootstrap.sh` на поде** (весь вывод в `/workspace/logs/*.log`, каждые 15 минут синк логов на HF):
  1. setup → скачать semdiv bf16 + MELD → собрать данные → `09_smoke_reward.py` (санити всех трёх наград) → калибровка W → `run.sh configs/astra1.yaml`.
  2. Чекпойнты каждые 25 шагов адаптером пушатся в HF (`runs/astra1/checkpoint-*`), так что смерть пода не обнуляет ночь.
  3. После обучения: `13_merge.py` → `30_heldout_eval.py` для semdiv и astra1 (`--n-harm 0`) → наш бенчмарк (домашки + деловые, режимы R и D, судья Opus 5.5, те же черновики) → `results.md` на HF.
  4. **Всегда:** `trap EXIT` + watchdog с жёстким лимитом 12 ч → финальный пуш логов → `DELETE /v1/pods/$RUNPOD_POD_ID`. Под удаляется при любом исходе: успех, ошибка, мёртвый судья, таймаут.

## Бюджет (оценка)
- astra-судья: 250 шагов × 6 групп = 1500 вызовов, около 9k токенов на вход (правила кешируются) и ~1.5k на выход → **~$200–250**. Плюс калибровка и смоук ~$5, утренний бенчмарк (судья Opus) ~$15. Лимит ключа нужен **≥ $300** (префлайт проверяет ≥ $270). Если хочется дешевле — `steps: 150`, тогда ~$150.
- RunPod 2× H200 secure ~$9.2/ч × 6–8 ч → **$55–75**, жёсткий лимит watchdog — 12 ч (≤ $110). Баланс $179 хватает, префлайт проверяет ≥ $120.

## Проверка
- До запуска пода: `python 11_grpo_train.py --config configs/astra1.yaml --dry-run` на eva01 (резолв конфига и сборка данных). Юнит-проверка `score_antislop` на 2 группах из `runs/heldout/semdiv1_samples.jsonl` с реальным вызовом astra: в JSON 8 оценок, разброс > 0.
- На поде фейл-фаст: смоук-награды и калибровка обязаны пройти до основного обучения, иначе под гасится.
- Утром: `results.md` в `rawmodels/antislop-astra1-run`. В нём таблица semdiv против astra1: held-out (MELD, Flesch, semdiv, слоп-лексикон) и наш бенчмарк («похоже на человека», слоп, потери и выдумки фактов, попарно против humanizer'а). Плюс `step_stats.jsonl` с динамикой `antislop`, и под в статусе удалён.
