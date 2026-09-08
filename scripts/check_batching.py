#!/usr/bin/env python3
"""Exercise concurrent real HTTP generation, late admission, and cancellation.

Run under memory_guard.py. The spawned server is the only resident model.
Checks scheduling and stream integrity without requiring batch-invariant logits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--report', type=Path, default=Path('artifacts/batching.json'))
    args = p.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(base + path, data=data,
                             headers={'Content-Type': 'application/json'}), timeout=180) as response:
            return json.load(response)

    def stream(prompt, maximum, first_block=None, barrier=None, cancel=False):
        if barrier:
            barrier.wait(timeout=15)
        start = time.monotonic()
        payload = {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': maximum,
                   'stream': True, 'stream_options': {'include_usage': True},
                   'cache_prompt': False, 'return_progress': True}
        chunks, first, final = [], None, None
        with urlopen(Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                             headers={'Content-Type': 'application/json'}), timeout=180) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                raw = line[5:].strip()
                if raw == b'[DONE]':
                    break
                chunk = json.loads(raw)
                assert 'error' not in chunk, chunk
                chunks.append(chunk)
                phase = chunk.get('minnow', {}).get('phase')
                if cancel and phase == 'refinement':
                    return {'cancelled': True}
                if phase == 'block' and first is None:
                    first = time.monotonic()
                    if first_block:
                        first_block.set()
                if (chunk.get('choices') or [{}])[0].get('finish_reason'):
                    final = chunk
            else:
                raise AssertionError('stream ended without DONE')
        assert final and first is not None
        assert len({c['id'] for c in chunks}) == 1
        assert chunks[-1]['choices'] == [] and chunks[-1]['usage']
        meta = final['minnow']
        assert meta['completion_tokens'] == sum(b['completion_tokens'] for b in meta['batches'])
        assert meta['evaluated_tokens'] == 32 * meta['denoise_forwards']
        text = ''.join((c.get('choices') or [{}])[0].get('delta', {}).get('content') or '' for c in chunks)
        assert text and '<|' not in text
        return {'started': start, 'first_block': first, 'finished': time.monotonic(),
                'text': text, 'minnow': meta, 'timings': final['timings']}

    with args.report.with_suffix('.server.log').open('w') as log:
        child = subprocess.Popen(['target/release/minnow', '--model', str(args.model),
                                  'serve', '--listen', f'127.0.0.1:{port}', '--parallel', '4'],
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(1200):
                if child.poll() is not None:
                    raise RuntimeError('server exited; see server log')
                try:
                    request('/health')
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError('startup timed out')
            # A lone request must make progress without a second arrival.
            lone = stream('What is 2 + 2? Answer briefly.', 32)
            first_block, barrier = threading.Event(), threading.Barrier(2)
            with ThreadPoolExecutor(max_workers=3) as executor:
                long = executor.submit(stream,
                    'Write a detailed tutorial of at least 1500 words about hash tables. '
                    'Cover collisions, chaining, open addressing, resizing, amortized complexity, '
                    'and implementation mistakes. Include worked examples and pseudocode.',
                    1024, first_block, barrier)
                react = executor.submit(stream, 'Write a React TypeScript example.', 384, None, barrier)
                assert first_block.wait(timeout=90), 'long request did not start decoding'
                late = executor.submit(stream, 'What is 3 + 4? Answer in one sentence.', 32).result(timeout=90)
                long_result, react_result = long.result(timeout=180), react.result(timeout=180)
            assert late['finished'] < long_result['finished'], 'late request waited for long response'
            health = request('/health')
            assert health['batching']['max_batch_size'] >= 2, health
            assert health['batching']['sequence_forwards'] > health['batching']['forward_batches']
            assert stream('Write a long science fiction story with detailed scenes.', 2048, cancel=True)['cancelled']
            cancelled = time.monotonic()
            for _ in range(100):
                if not any(s['is_processing'] for s in request('/slots')):
                    break
                time.sleep(0.05)
            else:
                raise AssertionError('cancelled request retained execution slot')
            cancel_seconds = time.monotonic() - cancelled
            # Also verifies the worker survived cancellation.
            after = stream('What is 5 + 6? Answer briefly.', 32)
            report = {'passed': True, 'health': health, 'lone': lone, 'long': long_result,
                      'react': react_result, 'late': late, 'after_cancel': after,
                      'cancel_seconds': cancel_seconds}
            args.report.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'passed': True, 'batching': health['batching'],
                              'late_request_seconds': late['finished'] - late['started']}), flush=True)
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
