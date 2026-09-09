#!/usr/bin/env python3
"""Benchmark cold prefill and useful long-response decode with one loaded model.

Run under memory_guard.py. All requests are sequential and disable prefix reuse.
Each case has a discarded warmup and repeated measured runs. Load time is excluded.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--cases', type=Path, help='JSON chat cases; --max-tokens sets the common output cap')
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--workspace-cache-mib', type=int, default=2048)
    p.add_argument('--prefill-only', action='store_true')
    p.add_argument('--decode-only', action='store_true')
    p.add_argument('--server-arg', action='append', default=[],
                   help='Additional minnow argument; repeat as --server-arg=--flag')
    args = p.parse_args()
    assert 1 <= args.iterations <= 20
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    base = f'http://127.0.0.1:{port}'

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        try:
            with urlopen(Request(base+path, data=data, headers={'Content-Type': 'application/json'}), timeout=600) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f'{path}: HTTP {error.code}: {error.read().decode()}') from error

    report = {'model': str(args.model), 'iterations': args.iterations, 'warmups': 1,
              'workspace_cache_mib': args.workspace_cache_mib, 'server_args': args.server_arg,
              'prefill': [], 'decode': []}

    def save():
        args.report.write_text(json.dumps(report, indent=2)+'\n')

    with args.report.with_suffix('.server.log').open('w') as log:
        child = subprocess.Popen(['target/release/minnow', '--model', str(args.model),
            '--workspace-cache-mib', str(args.workspace_cache_mib), *args.server_arg, 'serve',
            '--listen', f'127.0.0.1:{port}', '--parallel', '1'], stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(1800):
                if child.poll() is not None:
                    raise RuntimeError('server exited; see server log')
                try:
                    report['health'] = request('/health')
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError('startup timed out')
            report['settings'] = request('/props')['minnow']['generation_defaults']
            if not args.decode_only:
                seed = json.loads(Path('tests/prefill_benchmark_ids.json').read_text())
                for n in [512, 2048, 4096, 8192]:
                    ids = (seed*(n//len(seed)+1))[:n]
                    runs = []
                    for iteration in range(args.iterations+1):
                        out = request('/v1/completions', {'prompt': ids, 'max_tokens': 1, 'cache_prompt': False})
                        assert out['minnow']['prefill_tokens'] == n and out['minnow']['cached_tokens'] == 0
                        if iteration:
                            runs.append(out['timings']['prompt_ms']/1000)
                    item = {'tokens': n, 'seconds': runs, 'tokens_per_second': n*len(runs)/sum(runs)}
                    report['prefill'].append(item)
                    print(json.dumps({'prefill': item}), flush=True)
                    save()
            if not args.prefill_only:
                cases = json.loads(args.cases.read_text()) if args.cases else [
                    {'name': name, 'messages': [{'role': 'user', 'content': prompt}]}
                    for name, prompt in [
                        ('lhc', 'What is the LHC?'),
                        ('react', 'Write a React TypeScript example'),
                        ('fellowship', 'List the main characters of the Fellowship of the Ring with a short back story for each.')]]
                for case in cases:
                    name, messages = case['name'], case['messages']
                    prompt = '\n'.join(m['content'] for m in messages)
                    runs, warm_text = [], None
                    for iteration in range(args.iterations+1):
                        start = time.monotonic()
                        out = request('/v1/chat/completions', {'messages': messages,
                            'max_tokens': args.max_tokens, 'cache_prompt': False})
                        text = out['choices'][0]['message']['content']
                        print(json.dumps({'case':name,'iteration':iteration,
                            'useful_tokens':out['timings']['predicted_n'],
                            'useful_decode_tokens_per_second':out['timings']['predicted_per_second'],
                            'finish_reason':out['choices'][0]['finish_reason']}),flush=True)
                        if not iteration:
                            warm_text = text
                        else:
                            runs.append({'wall_seconds': time.monotonic()-start, 'timings': out['timings'],
                                'minnow': out['minnow'], 'usage': out['usage'],
                                'finish_reason': out['choices'][0]['finish_reason'],
                                'matches_warmup': text == warm_text, 'text': text})
                    useful = sum(r['timings']['predicted_n'] for r in runs)
                    decode = sum(r['timings']['predicted_ms'] for r in runs)/1000
                    refinements = sum(r['minnow']['denoise_forwards'] for r in runs)
                    blocks = sum(r['minnow']['blocks'] for r in runs)
                    item = {'name': name, 'prompt': prompt, 'max_tokens': args.max_tokens,
                            'useful_decode_tokens_per_second': useful/decode,
                            'mean_useful_tokens': useful/len(runs), 'refinements_per_block': refinements/blocks,
                            'runs': runs}
                    report['decode'].append(item)
                    print(json.dumps({'decode': {k:v for k,v in item.items() if k != 'runs'}}), flush=True)
                    save()
            report['complete'] = True
            save()
        finally:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                raise
            assert child.returncode == 0, child.returncode


if __name__ == '__main__':
    main()
