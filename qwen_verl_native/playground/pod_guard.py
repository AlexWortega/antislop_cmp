"""Runs on the playground GPU: destroys this rental once its deadline passes, so a forgotten pod stops billing."""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def credentials():
    env = dict(os.environ)
    try:
        for item in Path('/proc/1/environ').read_bytes().split(b'\0'):
            key, _, value = item.decode(errors='replace').partition('=')
            env.setdefault(key, value)
    except OSError:
        pass
    return env.get('CONTAINER_ID'), env.get('CONTAINER_API_KEY')


def main():
    deadline = float(sys.argv[1])
    ident, key = credentials()
    if not ident or not key:
        raise SystemExit('CONTAINER_ID or CONTAINER_API_KEY missing: cannot self-destruct')
    Path('/workspace/guard_started').write_text(json.dumps({'id': ident, 'deadline': deadline}))
    while True:
        # The laptop can move the deadline by rewriting this file.
        try:
            deadline = float(Path('/workspace/deadline').read_text())
        except (OSError, ValueError):
            pass
        if time.time() >= deadline:
            for _ in range(20):
                try:
                    request = urllib.request.Request(f'https://console.vast.ai/api/v0/instances/{ident}/', method='DELETE',
                                                     headers={'Authorization': 'Bearer ' + key})
                    urllib.request.urlopen(request, timeout=40).read()
                except Exception as exc:  # keep trying: an undeleted pod keeps billing
                    print('destroy failed', exc, flush=True)
                    time.sleep(30)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
