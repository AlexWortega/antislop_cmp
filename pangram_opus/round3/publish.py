"""Publish the frozen dev-selected classifier; do not select using fresh test."""
import hashlib,json,shutil
from pathlib import Path
from evaluate import ROOT,OUT,read,read_prompt
from core import CONTRACT,digest

def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def hit(m):return f"{m['tp']+m['tn']}/{m['n']} ({pct(m['agreement'])})"
if __name__=='__main__':
 from runtime import body_for
 from repeat import check
 from goal_check import check as check_goal
 repeatability=check()
 reports=json.loads((OUT/'test_results.json').read_text());selection=json.loads((OUT/'selection.json').read_text());runtime_selection=json.loads((OUT/'runtime_selection.json').read_text());validation=json.loads((OUT/'validation.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text());rows=read(OUT/'test.jsonl')
 old,new=reports['previous'],reports['selected'];goal_assessment=check_goal(new,json.loads((OUT/'goal_criteria.json').read_text()));(OUT/'goal_assessment.json').write_text(json.dumps(goal_assessment,indent=2));prompt=read_prompt(OUT/'best_candidate.py')
 assert new['prompt_hash']==digest(prompt)==selection['selected']['prompt_hash']
 budget=json.loads((ROOT/'cache/budget.json').read_text());costs={}
 for k,limit in budget['limits'].items():
  total=budget['spent'][k]+sum(v['amount'] for v in budget['pending'].values() if v['provider']==k);assert total<=limit;costs[k]={'conservative_total':total,'limit':limit,'settled_with_margin':budget['spent'][k]}
 search=[]
 for path in sorted((ROOT/'runs/single_pairwise2').glob('gen_*/results/metrics.json')):
  r=json.loads(path.read_text());search.append({'generation':path.parts[-3],'correct':json.loads((path.parent/'correct.json').read_text())['correct'],'objective':r['combined_score'],'metrics':r.get('public')})
 result={'round':'round3','runtime_selection':runtime_selection,'runtime_validation':[{k:v for k,v in r.items() if k!='predictions'} for r in json.loads((OUT/'runtime_validation.json').read_text())],'anchor_validation':[{k:v for k,v in r.items() if k!='predictions'} for r in json.loads((OUT/'anchor_validation.json').read_text())],'goal_assessment':goal_assessment,'repeatability':repeatability,'support_order_check':json.loads((OUT/'support_order_check.json').read_text()),'reference':json.loads((OUT/'reference.json').read_text()),'protocol':protocol,'fresh_audit':json.loads((OUT/'fresh_audit.json').read_text()),'selected':selection['selected'],'validation':[{k:v for k,v in r.items() if k!='predictions'} for r in validation],'test':{name:{k:v for k,v in r.items() if k!='predictions'} for name,r in reports.items()},'comparison':json.loads((OUT/'comparison.json').read_text()),'budget_session':costs,'search_results':search,'teacher_versions':sorted({r['teacher_version'] for r in rows})}
 result['fine_status']=json.loads((OUT/'fine_status.json').read_text())
 (OUT/'results.json').write_text(json.dumps(result,indent=2));shutil.copyfile(OUT/'results.json',ROOT/'results.json')
 for f in ['best_candidate.py','best_prompt.txt','inference_system.txt','selection.json']:shutil.copyfile(OUT/f,ROOT/f)
 template=json.loads((OUT/'previous_request_template.json').read_text());template['messages'][0]['content']=prompt+'\n\n'+CONTRACT;
 if runtime_selection['mode']!='minimal':template=body_for(prompt,'REPLACE WITH THE ENGLISH TEXT TO CLASSIFY',runtime_selection['mode'])
 (ROOT/'request_template.json').write_text(json.dumps(template,ensure_ascii=False,indent=2));(ROOT/'inference_config.json').write_text(json.dumps({'mode':runtime_selection['mode'],'prompt_hash':digest(prompt)},indent=2))
 manifest={'active_round':'round3','model':template['model'],'teacher_requested':'pangram-4','teacher_returned':result['teacher_versions'],'selected_prompt_hash':digest(prompt),'selected_candidate':selection['selected']['candidate'],'source_git_commit':result['reference']['checkout_commit'],'tokenizer':'o200k_base','single_text_search_and_evaluation':True,'inference_mode':runtime_selection['mode'],'dataset_files':{name:{'sha256':hashlib.sha256((OUT/(name+'.jsonl')).read_bytes()).hexdigest(),'rows':len(read(OUT/(name+'.jsonl')))} for name in ['train_pool','search','support','dev','test']},'shinka_commit':'8adc053a2ce4511ad2ac310e004c530a73fb974a'};(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 changed=new['prompt_hash']!=old['prompt_hash'];outcome='Рабочий промпт заменён результатом описанной процедуры калибровки.' if changed else 'Сохранён предыдущий рабочий промпт.'
 lines=['# Pangram → Opus 5.5: одиночная калибровка с опытом pairwise judge','',outcome+' Промпт и runtime выбраны по обучению и раннему dev до просмотра результатов нового теста. Разметка свежих текстов не передавалась селектору. Метрика измеряет имитацию Pangram, а не достоверность определения авторства или литературное качество.','',
 '## Что использовано из прежней калибровки','',
 'Найден локальный Opus 5.5 judge в `cmp/common.py`, `cmp/calib.py`, с результатами `cmp/out/calib_report.json`. Его rulebook точно совпадает с [miqdadbadjuber/anti-slop, commit 388cbe3](https://github.com/miqdadbadjuber/anti-slop/blob/388cbe3b6c37d5175b9f460015bb092ef9e34894/skills/antislop-copywriting/SKILL.md). Снимок, SHA-256 и MIT license сохранены в `round3/reference*`.','',
 'В исходной проверке: pairwise human-win 96,7% для обычного AI и 87,8% для humanlike AI; порядок устойчив в 44/45 и 42/45 парах. AUROC 0,993/0,941 относится к отдельной абсолютной шкале human, а не к парному сравнению. Исходные фрагменты имели 90–300 слов. Эти результаты не доказывают совпадение с Pangram или работу на сверхкоротких текстах.','',
 'Отдельный вызов Opus 5.5 получил rulebook, отчёт и обучающие примеры и предложил три стратегии: контрфактическое сравнение формулировок, нормировку признаков по длине и исправленные жанровые правила. К ним добавлены пять пар с противоположными метками Pangram как фиксируемые в промпте примеры. Каждая пара происходит из одной исходной группы; все эти группы исключены из оцениваемого train и dev. Эти десять показанных примеров не входят в training score этапа Shinka. Новые парные оценки авторства не подменяют метки Pangram.','',
 '## Протокол поиска','',
 f"Общий пул: {protocol['train_n']} текстов. Прежний test раунда 2 теперь обучающий и больше не считается независимым. Подбор идёт на фиксированных 128 случаях, 64 на класс, с настоящими human-контролями и короткими положительными примерами. Support: 10 текстов из пяти отдельных групп. Предварительный выбор стартового промпта: 32 случая из train. Shinka: до восьми попыток поколений или до достижения потолка метрики на train, {len(search)} оценённых вариантов. Во всех стадиях — один классифицируемый текст на API-запрос, включая поиск; показанные размеченные reference-пары являются частью самого промпта.",'',
 'Критерий: 50% общей balanced accuracy + 50% средней balanced accuracy диапазонов длины, где есть оба класса. Отказ провайдера считается ошибкой, а не меткой human; случаи не исключаются. Dev из 105 текстов используется повторно, поэтому не считается новой независимой оценкой. По dev выбирается между прежним промптом и лучшим обучающим кандидатом; ни fresh-test тексты, ни метки не передавались оптимизатору.','',
 '| Dev | Совпадение | Balanced accuracy | Критерий с учётом длины | FP | FN | Нет ответа |','|---|---:|---:|---:|---:|---:|---:|']
 for r in validation:
  m=r['report']['overall'];lines.append(f"| `{Path(r['candidate']).relative_to(ROOT)}` | {hit(m)} | {pct(m['balanced_accuracy'])} | {r['report']['objective']:.4f} | {m['fp']} | {m['fn']} | {m['unavailable']} |")
 lines+=['','## Новый независимый test','',f"{len(rows)} текстов из {len({r['split_group'] for r in rows})} групп новых документов/тем, без нормализованных и близких текстовых совпадений с прежними данными. Настоящие human-оригиналы, генерации шести моделей, обычные/минимальные/antislop-переписывания Markdown-only и смеси человеческих/AI-предложений. Удалены {result['fresh_audit']['raw_n']-len(rows)} точных дубля/дублей; исходное human-происхождение сохранено. Всего положительных меток Pangram: {sum(r['slop'] for r in rows)}, отрицательных: {sum(not r['slop'] for r in rows)}. Фактические версии учителя: {', '.join(result['teacher_versions'])}.",'',
 '| Метрика | Прежний промпт | Выбранный |','|---|---:|---:|',f"| Совпадение | {hit(old['overall'])} | {hit(new['overall'])} |",f"| Balanced accuracy | {pct(old['overall']['balanced_accuracy'])} | {pct(new['overall']['balanced_accuracy'])} |",f"| FP | {old['overall']['fp']} | {new['overall']['fp']} |",f"| FN | {old['overall']['fn']} | {new['overall']['fn']} |",f"| Нет классификации | {old['overall']['unavailable']} | {new['overall']['unavailable']} |",f"| <200 токенов | {hit(old['under200'])} | {hit(new['under200'])} |",'',
 '| Длина, o200k_base | Прежний | Выбранный | Recall Pangram-AI, выбранный |','|---|---:|---:|---:|']
 for b in ['lt25','25-49','50-99','100-199','200plus']:
  if b in new['by_bucket']:lines.append(f"| {b} | {hit(old['by_bucket'][b])} | {hit(new['by_bucket'][b])} | {pct(new['by_bucket'][b]['slop_recall'])} |")
 if changed:
  c=result['comparison'];lo,hi=c['group_bootstrap_delta95'];lines+=['',f"Исправлено {c['improved']} прежних ошибок, появилось {c['regressed']} новых. Исследовательский 95% bootstrap-интервал изменения balanced accuracy по исходным группам: [{100*lo:.1f}; {100*hi:.1f}] п.п. Парный McNemar p={c['mcnemar_casewise_exploratory_p']:.4g}, но тексты внутри групп зависимы. На основании этого test дальнейший подбор не проводился."]
 else:lines+=['','Обе колонки относятся к одному и тому же промпту; одинаковые предсказания взяты из кэша. Это не доказательство улучшения.']
 for field,title in [('origin','Происхождение'),('model','Модель происхождения')]:
  lines+=['',f'| {title} | Прежний | Выбранный |','|---|---:|---:|']
  for key,m in new['by_'+field].items():lines.append(f"| {key} | {hit(old['by_'+field][key])} | {hit(m)} |")
 lines+=['','Токены считаются через o200k_base как воспроизводимую меру, не как закрытый токенизатор Opus. Pangram рекомендует минимум 50 слов, хотя API принимает более короткий ввод; результаты ниже этой длины описывают имитацию API вне рекомендованной области. Малые разрезы не доказывают устойчивость к неизвестным моделям или скаффолдам. Истинное AI-происхождение не заменяет метку учителя.','',
 '## Использование и артефакты','',
 '`best_prompt.txt` — выбранные критерии и reference-примеры (если сохранены кандидатом); `inference_system.txt` — полный system с контрактом; `request_template.json` — точный API-запрос без ключа. Модель `anthropic/claude-opus-5.5`, strict JSON schema, выбранный на dev режим reasoning, max_tokens зависит от выбранного runtime, temperature не задаётся. Точные настройки находятся в request_template.json и inference_config.json. Один классифицируемый текст на запрос.','',
 'С `OPENROUTER_API_KEY` в окружении, из этой папки:','', '```sh','.venv/bin/python classify.py example.txt','```','',
 'Или stdin. Ответ `{"slop": true}` / `{"slop": false}`. Отказ провайдера возвращает JSON-ошибку в stderr и код выхода 2.','',
 'Полные результаты: `round3/results.json`, `round3/test_results.json`, `round3/validation.json`, `round3/screen_results.json`, `round3/advisor.json`. Оригинальные ответы API и неудачная первая попытка консультанта сохранены отдельно. Исходные данные содержат provenance; условия Brown (некоммерческие исследования), Enron/Yelp и прочих корпусов сохраняют силу. Ключи не записываются в проект.','',
 '## Расходы за всю сессию','']
 for k,c in costs.items():lines.append(f"- {k}: консервативно ${c['conservative_total']:.2f} из ${c['limit']:.0f}, включая запас и неизвестные резервы.")
 lines+=['','Pangram — расчёт по тарифу за начатые 100 слов с запасом, не выписка счёта. OpenRouter — возвращённые стоимости с запасом и незакрытые резервы. `cache/budget.json` сохраняет общий лимит; не сбрасывайте его при продолжении.','',
 'Зависимости: `requirements-lock.txt`; Shinka commit `8adc053a2ce4511ad2ac310e004c530a73fb974a`. Скрипты полного раунда находятся в `round3/`. Предыдущий промпт, результаты и README сохранены с префиксом `round3/previous_`.']
 lines+=['',f"Повторяемость выбранного промпта: {repeatability['stable_cases']}/{repeatability['n']} заранее отобранных случаев дали одинаковый ответ в трёх вызовах; отсутствующих ответов: {repeatability['unavailable_calls']}. Это малая диагностическая проверка, не гарантия детерминизма."]
 lines+=['', 'Операционный критерий приближения к Pangram: '+('ДОСТИГНУТ на этом fresh-test.' if goal_assessment['met'] else 'ПОКА НЕ ДОСТИГНУТ. Не пройдены: '+', '.join(goal_assessment['failed']))+' Точные заранее записанные пороги — round3/goal_criteria.json; проверка — round3/goal_assessment.json.']
 lines+=['', 'Дополнительная dev-проверка runtime до просмотра fresh-outcomes: minimal / provider-default / medium с тем же промптом и строгой схемой. Выбран режим **'+runtime_selection['mode']+'**. Кеширование system-prefix для default/medium настроено по [документации OpenRouter](https://openrouter.ai/docs/guides/best-practices/prompt-caching); reasoning — по [документации](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens). Это повторное использование dev, а не независимая оценка.']
 lines+=['', 'После runtime-проверки разрешены ещё два статических варианта: расширенные Pangram-примеры и полный исходный rulebook с расширенными примерами. Дополнительные примеры взяты только из train; они могут пересекаться с прежним оцениваемым train, поэтому исходный training score Shinka не относится к этим вариантам. Их выбор делался только на повторно используемом dev, до просмотра fresh-outcomes. Подробности — round3/anchor_validation.json.']
 fine=result['fine_status']
 if not fine.get('skipped'):
  note=['## Ограниченный шаг по оставшимся ошибкам','',f"Прежний dev был явно переведён в калибровочное обучение. Shinka получил {fine['n']} случаев: 105 бывших dev и {fine['controls_n']} контрольных текстов вне reference-групп. Выполнена одна попытка мутации после baseline. Эти данные уже не являются независимой проверкой; итоговое качество измеряется только на новом test. Метки reference-примеров не менялись, отказы провайдера не оптимизировались.",'']
  ix=lines.index('## Новый независимый test');lines[ix:ix]=note
 text='\n'.join(lines)+'\n';(ROOT/'README.md').write_text(text);(OUT/'REPORT.md').write_text(text)
 print(json.dumps({'changed':changed,'old':old['overall'],'selected':new['overall'],'costs':costs},indent=2))
