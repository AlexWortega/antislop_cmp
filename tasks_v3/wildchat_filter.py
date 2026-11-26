"""Real user requests to an LLM from WildChat-1M (allenai, ODC-BY): first user turn of each conversation.

  <venv with pyarrow>/bin/python -I tasks_v3/wildchat_filter.py <dir with train-*.parquet> tasks_v3/sources/wildchat_candidates.jsonl
Keeps RU and EN requests a writing model should handle in prose: questions, explanations, advice, emails, posts,
letters, stories. Drops code, translation, jailbreaks and role-play setups, prompt-generator templates, math,
moderation-flagged or redacted turns, and near-duplicates (the same user often pastes one template many times).
The parquet files are untrusted data; this script only reads them.
"""
import collections
import glob
import json
import re
import sys

import pyarrow.parquet as pq

LANGS = {'Russian': 'ru', 'English': 'en'}
CODE = re.compile(r'```|\bdef |\bfunction\b|\bclass \w+[:({]|#include|\bSELECT\b|\bimport \w|</?\w+>|\{\s*\n|;\s*\n|=>|\bconsole\.|\bnpm\b|\bpip\b|'
                  r'\b(python|javascript|java|c\+\+|c#|sql|html|css|react|excel formula|regex|код|скрипт|функци[юя]|программ[уы])\b', re.I)
SKIP = re.compile(r'translat|перевед|перевод|midjourney|prompt generator|\bDAN\b|jailbreak|ignore (all |previous )|act as|pretend|role ?play|'
                  r'roleplay|притворись|представь, что ты|ты [—-] |you are (a|an) |rewrite|перепиши|paraphrase|summari[sz]e|резюмир|'
                  r'continue|продолж|essay|эссе|реферат|курсов|диплом|homework|домашн|\bsolve\b|реши|вычисл|calculate|'
                  r'fanfic|фанфик|nsfw|sex|секс|erotic|эрот|https?://|www\.|\[.*?\]\(|\{\{', re.I)
DIGITS = re.compile(r'\d')


def keep(text):
    if not 25 <= len(text) <= 900 or text.count('\n') > 8:
        return False
    if CODE.search(text) or SKIP.search(text):
        return False
    if len(DIGITS.findall(text)) > len(text) * 0.08:
        return False
    return len(re.findall(r'[A-Za-zА-Яа-яЁё]', text)) > len(text) * 0.6


def main(folder, out_path):
    seen, out, counts = set(), [], collections.Counter()
    for path in sorted(glob.glob(folder.rstrip('/') + '/train-*.parquet')):
        table = pq.read_table(path, columns=['conversation_hash', 'conversation', 'language', 'toxic', 'redacted', 'model'])
        for row in table.to_pylist():
            counts['conversations'] += 1
            lang = LANGS.get(row['language'])
            if not lang or row['toxic'] or row['redacted'] or not row['conversation']:
                continue
            first = next((t for t in row['conversation'] if t['role'] == 'user'), None)
            if not first or first.get('toxic') or first.get('redacted'):
                continue
            text = (first.get('content') or '').strip()
            counts[f'{lang} first turns'] += 1
            if not keep(text):
                continue
            key = re.sub(r'\W+', ' ', text.lower())[:60]
            if key in seen:
                counts['duplicates'] += 1
                continue
            seen.add(key)
            reply = next((t for t in row['conversation'] if t['role'] == 'assistant'), None)
            out.append({'lang': lang, 'id': row['conversation_hash'], 'prompt': text, 'source_model': row['model'],
                        'reply_words': len((reply or {}).get('content', '').split())})
            counts[f'{lang} kept'] += 1
    with open(out_path, 'w') as handle:
        for r in out:
            handle.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps(counts, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
