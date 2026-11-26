"""Export the dev-selected prompt and a report; never select based on test."""
import json
import hashlib
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,CONTRACT,read_prompt,digest
OUT=ROOT/'round2'

def percent(x):return '—' if x is None else f'{100*x:.1f}%'
def pair(m):return f"{m['tp']+m['tn']}/{m['n']} ({percent(m['agreement'])})"

def main():
 result=json.loads((OUT/'results.json').read_text());old=result['test']['previous'];new=result['test']['expanded']
 selection=json.loads((OUT/'selection.json').read_text())['selected']
 prompt=read_prompt(OUT/'best_candidate.py');assert digest(prompt)==selection['prompt_hash']==new['prompt_hash']
 for f in ['best_candidate.py','best_prompt.txt','inference_system.txt']:shutil.copyfile(OUT/f,ROOT/f)
 template=json.loads((OUT/'previous_request_template.json').read_text());template['messages'][0]['content']=prompt+'\n\n'+CONTRACT
 (ROOT/'request_template.json').write_text(json.dumps(template,ensure_ascii=False,indent=2))
 (ROOT/'results.json').write_text(json.dumps(result,indent=2))
 for filename in ['manifest.json','selection.json']:
  if (ROOT/filename).exists() and not (OUT/('previous_'+filename)).exists():shutil.copyfile(ROOT/filename,OUT/('previous_'+filename))
 shutil.copyfile(OUT/'selection.json',ROOT/'selection.json')
 manifest={'active_round':'round2','model':template['model'],'teacher_requested':'pangram-4','teacher_returned':'4.1','selected_prompt_hash':digest(prompt),'selected_candidate':selection['candidate'],'tokenizer':'o200k_base','single_text_final_evaluation':True,'dataset_files':{name:{'sha256':hashlib.sha256((OUT/(name+'.jsonl')).read_bytes()).hexdigest(),'rows':len((OUT/(name+'.jsonl')).read_text().splitlines())} for name in ['train_pool','search','dev','test']},'shinka_commit':'8adc053a2ce4511ad2ac310e004c530a73fb974a'}
 (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 lines=['# Pangram 4.1 → Opus 5.5: расширенная калибровка','',
 'Рабочий промпт выбран на отдельной dev-выборке после расширения данных и оптимизации Shinka Evolve. Это имитация решений Pangram, а не доказательство AI-авторства или оценка литературного качества. Полная эквивалентность и устойчивость к неизвестным моделям не установлены.','',
 '## Независимая проверка','',
 'Новый test: 138 текстов, 52 группы исходных документов/тем, 110 отрицательных и 28 положительных меток Pangram. Промпт зафиксирован до оценки test; все запросы одиночные и используют одинаковую строгую JSON schema. Постоянный ответ «не слоп» дал бы 79,7% совпадений и 50% balanced accuracy.','',
 '| Метрика | Предыдущий промпт | Выбранный на dev |','|---|---:|---:|',
 f"| Совпадение с Pangram | {pair(old['overall'])} | {pair(new['overall'])} |",
 f"| Balanced accuracy | {percent(old['overall']['balanced_accuracy'])} | {percent(new['overall']['balanced_accuracy'])} |",
 f"| Ложные срабатывания / Pangram-human | {old['overall']['fp']}/110 | {new['overall']['fp']}/110 |",
 f"| Пропуски / Pangram-AI | {old['overall']['fn']}/28 | {new['overall']['fn']}/28 |",
 f"| Отказ провайдера / нет классификации | {old['overall']['unavailable']} | {new['overall']['unavailable']} |",
 f"| Тексты <200 токенов | {pair(old['under_200'])} | {pair(new['under_200'])} |",
 f"| Подлинные human-оригиналы | {pair(old['human_origin_teacher_negative'])} | {pair(new['human_origin_teacher_negative'])} |",'',
 '| Длина, o200k_base | Предыдущий | Выбранный |','|---|---:|---:|']
 for b in ['lt25','25-49','50-99','100-199','200plus']:
  if b in new['by_bucket']:lines.append(f"| {b} | {pair(old['by_bucket'][b])} | {pair(new['by_bucket'][b])} |")
 comparison=result['paired'];ci=result['group_bootstrap_balanced_accuracy_delta']['percentile95']
 lines+=['',f"Исправлено {comparison['improved']} прежних ошибок, появилось {comparison['regressed']} новых. Парный McNemar p={comparison['exact_mcnemar_p']:.4g} (тексты внутри групп зависимы). Исследовательский 95% bootstrap-интервал изменения balanced accuracy с пересэмплированием исходных групп: [{100*ci[0]:.1f}; {100*ci[1]:.1f}] п.п. Он не доказывает перенос на другие распределения.",'',
 'Отказы провайдера (`content_filter`) сохранены как отсутствие предсказания, не заменяются меткой false, не исключаются и считаются ошибкой при расчёте agreement/recall/balanced accuracy. Повторные запросы для обхода отказа не выполняются. Рабочий CLI возвращает понятную ошибку с кодом выхода 2. Обработка добавлена после двух отказов на dev старого промпта, до выбора кандидата и открытия test; промпты и параметры API не менялись.','',
 '## Данные и поиск','',
 'Добавлено 576 новых размеченных случаев, 570 уникальных. Из них 320 подлинных человеческих фрагментов: научная/официальная проза, речи, письма, отзывы. Также добавлены Markdown-варианты, лёгкая правка, разговорное и antislop-переписывание, смешанные и сгенерированные с нуля тексты. Генераторы: GPT-5.5, Sonnet 5.5, Gemini 3.5 Flash, DeepSeek v4.1 Flash, Qwen3.6 Plus и Opus 5.5. Все новые ответы учителя — версия 4.1.','',
 'Train pool вырос с 97 до 552: 327 новых + 97 прежних + 128 прежних диагностических. Из него отобраны фиксированные 256 случаев с балансом классов и приоритетом ошибок старого промпта. Новые dev/test: 105/138, разделены по документам/темам и близким дубликатам. Прежний стресс-тест теперь часть обучения и больше не является независимой проверкой. Старые 31 тестовый пример не включены в обучение.','',
 'Shinka изменял только строковый PROMPT; кандидаты разбирались через AST и не исполнялись. Поиск использовал пакеты по 8 текстов и метрику, усредняющую общую balanced accuracy и balanced accuracy диапазонов длины. Предыдущий промпт и два лучших различных кандидата затем сравнивались на dev одиночными запросами. Победитель выбирался по dev-метрике с заранее заданными правилами разрешения равенства. Состав пакета влияет на ответы, поэтому training score не считается оценкой рабочего одиночного режима.','',
 f"Выбран: `{Path(selection['candidate']).relative_to(ROOT)}`; dev objective {selection['report']['objective']:.4f}. Test не использовался для выбора или редактирования промпта.",'',
 'Подробный протокол и источники: [round2/DATASET.md](round2/DATASET.md). Размеры считаются через o200k_base, а не через закрытый токенизатор Opus. Pangram рекомендует >=50 слов, хотя API принимает более короткий ввод: результаты для него характеризуют имитацию API вне рекомендованной области. Истинное AI-происхождение не подменяет метки Pangram. Небольшое число тем и примеров на модель ограничивает выводы об устойчивости.','',
 '## Использование','',
 '`inference_system.txt` — готовый системный промпт с контрактом; `best_prompt.txt` — критерии; `request_template.json` — точное тело запроса без ключа. `results.json` — сводка; `round2/test_results.json` — все предсказания и разрезы по длине, происхождению, модели и сценарию.','',
 'Из этой папки, с `OPENROUTER_API_KEY` в окружении:','',
 '```sh','.venv/bin/python classify.py example.txt','```','',
 'Можно передать текст через stdin. Ответ `{"slop": true}` или `{"slop": false}`. Модель `anthropic/claude-opus-5.5`, strict JSON schema, `provider.require_parameters=true`, reasoning minimal, max_tokens 2048; temperature не задаётся. Не добавляйте к одному рабочему запросу посторонние тексты: финальные метрики измерены для одного текста.','',
 '## Расходы и воспроизведение','']
 for provider,c in result['budget_session'].items():
  lines.append(f"- {provider}: консервативно учтено ${c['conservative_total']:.2f} из ${c['limit']:.0f} за всю сессию, включая запас и незакрытые резервы.")
 lines+=['', 'Pangram учитывается по опубликованному тарифу с округлением до блоков 100 слов и 10% запасом; это расчёт, не выписка счёта. OpenRouter учитывает возвращённую стоимость с запасом; неизвестные запросы сохраняют резервы. Не удаляйте `cache/budget.json` при продолжении. API-ключей в файлах нет.','',
 'Зависимости закреплены в `requirements-lock.txt`, Shinka — commit `8adc053a2ce4511ad2ac310e004c530a73fb974a`. Локальные тесты: `.venv/bin/python -m unittest discover -s . -p test_core.py`. Скрипты раунда находятся в `round2/`: `data_build.py`, `expand.py`, `direct.py`, `prepare_round.py`, `run_round.py`, `finish_round.py`, `summarize_round.py`. Повторные запросы используют кэш; для нового независимого эксперимента нужны отдельные артефакты и заранее зафиксированный протокол.','',
 'Предыдущий отчёт и промпт сохранены как `round2/previous_README.md` и `round2/previous_best.py`.']
 dev_lines=['## Выбор на dev','', '| Промпт | Совпадение | Balanced accuracy | Критерий с учётом длины | Ложные срабатывания | Пропуски AI | Нет ответа |','|---|---:|---:|---:|---:|---:|---:|']
 for r in result['validation']:
  m=r['report']['overall'];name=str(Path(r['candidate']).relative_to(ROOT))
  dev_lines.append(f"| `{name}` | {pair(m)} | {percent(m['balanced_accuracy'])} | {r['report']['objective']:.4f} | {m['fp']} | {m['fn']} | {m['unavailable']} |")
 dev_lines+=['','Общая доля совпадений не была критерием выбора: он также учитывает оба класса и диапазоны длины. Поэтому снижение ложных срабатываний при росте пропусков коротких AI-текстов не гарантирует победу кандидата.','']
 lines[4:4]=dev_lines
 weak=['## Ограничения, выявленные новым тестом','',
 'Особенно слабое место — тексты <25 токенов: все 7 положительных меток Pangram в этом диапазоне пропущены (0/7 обнаружено). Это не подтверждает устойчивость на сверхкоротком вводе. Подлинные человеческие оригиналы распознаны в 68/69 случаев, но простое Markdown-оформление — лишь 5/10, смешанные тексты — 6/13. Поэтому высокая точность на человеческих оригиналах не переносится автоматически на соседние сценарии.','',
 '| Модель происхождения, все сценарии test | Совпадение |','|---|---:|']
 for model,m in new['by_model'].items():
  if model!='not_applicable':weak.append(f'| `{model}` | {pair(m)} |')
 weak+=['','На модель приходится лишь 8–12 примеров, поэтому это диагностические разрезы, а не устойчивый рейтинг моделей. Проверка неизвестных семейств генераторов не проводилась.','']
 lines[lines.index('## Использование'):lines.index('## Использование')]=weak
 if new['prompt_hash']==old['prompt_hash']:
  lines=[line for line in lines if not line.startswith('Исправлено ')]
  lines.insert(2,'**Сохранён предыдущий рабочий промпт:** новые кандидаты не превзошли его по заранее заданной dev-метрике. Расширенный датасет и все результаты поиска сохранены. Сравнение на test в этом случае показывает один и тот же промпт, а не улучшение.\n')
 (ROOT/'README.md').write_text('\n'.join(lines)+'\n');(OUT/'REPORT.md').write_text(('\n'.join(lines)+'\n').replace('](round2/DATASET.md)','](DATASET.md)'))
 print('Published dev-selected prompt',selection['prompt_hash'])

if __name__=='__main__':main()
