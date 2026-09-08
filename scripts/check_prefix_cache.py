#!/usr/bin/env python3
"""Validate real multi-slot K/V reuse, forks, growth, LRU, and SSE cache accounting.

Run sequentially under memory_guard.py against a server with at least four cache
slots and the default similarity threshold. No checkpoint is loaded by this script.
"""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--report', type=Path, default=Path('artifacts/prefix-cache/api-cache.json'))
    args = parser.parse_args()
    records = []

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        with urlopen(Request(args.url + path, data=data, headers={'Content-Type': 'application/json'}), timeout=180) as response:
            return json.load(response)

    props = request('/props')
    assert props['minnow']['cache']['cache_slots'] >= 4
    assert props['minnow']['inference_workers'] == 1

    def complete(label, prompt, stream=False, cache=True):
        body = {'prompt': prompt, 'max_tokens': 1, 'cache_prompt': cache,
                'stream': stream, 'return_progress': True,
                'stream_options': {'include_usage': True}}
        if stream:
            chunks = []
            with urlopen(Request(args.url + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'}), timeout=180) as response:
                for line in response:
                    if line.startswith(b'data:'):
                        data = line[5:].strip()
                        if data == b'[DONE]':
                            break
                        chunk = json.loads(data)
                        assert 'error' not in chunk, chunk
                        chunks.append(chunk)
                else:
                    raise AssertionError('missing DONE')
            out = next(c for c in reversed(chunks) if (c.get('choices') or [{}])[0].get('finish_reason'))
            out['usage'] = chunks[-1]['usage']
            text = ''.join((c.get('choices') or [{}])[0].get('text', '') for c in chunks)
            progress = [c['prompt_progress'] for c in chunks if 'prompt_progress' in c]
            meta = out['minnow']
            if meta['prefill_tokens']:
                assert progress[0]['processed'] == meta['cached_tokens']
                assert progress[-1]['processed'] == progress[-1]['total']
                assert all(p['cache'] == meta['cached_tokens'] for p in progress)
                last = max(i for i, c in enumerate(chunks) if 'prompt_progress' in c)
                assert chunks[last + 1]['minnow']['phase'] == 'prefill_complete'
            else:
                assert not progress
            done = next(i for i, c in enumerate(chunks) if c.get('minnow', {}).get('phase') == 'prefill_complete')
            assert all('prompt_progress' not in c for c in chunks[done:])
            assert any(c.get('minnow', {}).get('phase') == 'refinement' for c in chunks[done + 1:])
        else:
            out = request('/v1/completions', body)
            text = out['choices'][0]['text']
        meta = out['minnow']
        assert meta['cached_tokens'] == out['timings']['cache_n'] == out['usage']['prompt_tokens_details']['cached_tokens']
        assert meta['prefill_tokens'] + meta['cached_tokens'] == meta['prompt_tokens'] // 32 * 32
        assert meta['processed_tokens'] == meta['prefill_tokens'] + 32 * (meta['denoise_forwards'] + meta['commit_forwards'])
        if not cache:
            assert meta['cached_tokens'] == 0 and meta['cache_slot'] is None
        records.append({'case': label, 'timings': out['timings'], 'minnow': meta})
        print(label, 'slot', meta['cache_slot'], 'cached', meta['cached_tokens'],
              'copied', meta['cache_copied_tokens'], 'prefill', meta['prefill_tokens'], flush=True)
        return meta, text

    shared = 'Shared reading: ' + 'A prefix block remains stable after it is committed. ' * 24 + '\n'
    a = shared + 'Alpha notes: ' + 'Hash tables map keys to buckets and handle collisions. ' * 180
    b = shared + 'Beta notes: ' + 'Particle accelerators use electric fields to increase particle energies. ' * 180
    a0, text = complete('A prime', a)
    a1, warm_text = complete('A warm', a, stream=True)
    assert a1['cached_tokens'] == a1['prompt_tokens'] // 32 * 32
    assert a1['cache_slot'] == a0['cache_slot'] and text == warm_text
    before = request('/slots')[a1['cache_slot']]['cache']['capacity_tokens']
    extended = a + '\n' + 'Separate chaining stores colliding entries in a bucket list. ' * 500
    growing, growing_text = complete('A grows', extended, stream=True)
    assert growing['cache_slot'] == a0['cache_slot'] and growing['cached_tokens'] >= a1['cached_tokens'] - 32
    after = request('/slots')[growing['cache_slot']]['cache']['capacity_tokens']
    assert after > before
    _, cold_text = complete('A grown cold oracle', extended, cache=False)
    assert growing_text == cold_text
    branch, branch_text = complete('B forks', b, stream=True)
    assert branch['cache_slot'] != a0['cache_slot']
    assert branch['cache_copied_tokens'] == branch['cached_tokens'] > 0
    _, branch_cold_text = complete('B cold oracle', b, cache=False)
    assert branch_text == branch_cold_text
    again, again_text = complete('A survives fork', a, stream=True)
    assert again['cache_slot'] == a0['cache_slot'] and again['cached_tokens'] == a1['cached_tokens']
    assert again_text == text
    for label in ['C', 'D']:
        complete(label, label + ' independent conversation. ' + 'Describe this topic with care. ' * 180)
    complete('A touched', a)
    e, _ = complete('E evicts B', 'E separate request. ' + 'Give clear and concise examples. ' * 180)
    assert e['cache_slot'] == branch['cache_slot']
    missed, _ = complete('B after eviction', b)
    assert missed['cached_tokens'] == branch['cached_tokens'] < missed['prompt_tokens'] // 32 * 32
    slots = request('/slots')
    assert not any(s['is_processing'] for s in slots)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'passed': True, 'records': records, 'slots': slots}, indent=2) + '\n')
    print('passed', flush=True)


if __name__ == '__main__':
    main()
