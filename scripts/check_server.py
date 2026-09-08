#!/usr/bin/env python3
"""Check the local HTTP server and compare a small corpus to saved generations."""
import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--cases', type=Path, default=Path('tests/generation_cases.json'))
    p.add_argument('--reference', type=Path, default=Path('artifacts/reference-generation.json'))
    p.add_argument('--output', type=Path, default=Path('artifacts/server-check.json'))
    args = p.parse_args()

    def request(path, payload=None, status=200):
        data = json.dumps(payload).encode() if payload is not None else None
        req = Request(args.url + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            response = urlopen(req, timeout=180)
        except HTTPError as error:
            response = error
        with response:
            body = json.loads(response.read())
            assert response.status == status, (path, response.status, status, body)
            if status >= 400:
                assert body['error']['type'] in ('invalid_request_error', 'exceed_context_size_error'), body
            return body

    health = request('/health')
    assert health['status'] == 'ok'
    assert request('/v1/models')['data'][0]['id'] == health['model']
    request('/v1/completions', {'prompt': 'hello', 'model': 'missing'}, status=400)
    request('/v1/completions', {'prompt': 'hello', 'max_tokens': health['max_context']}, status=400)
    request('/v1/completions', {'prompt': ''}, status=400)
    request('/v1/completions', {'prompt': 'hello', 'unknown_field': True}, status=400)
    request('/v1/chat/completions', {'messages': [], 'prompt': 'hello'}, status=400)
    reference = {r['name']: r for r in json.loads(args.reference.read_text())['results']}
    cases = json.loads(args.cases.read_text())
    # Warm the exact request path, then measure each saved case once.
    request('/v1/chat/completions', {'messages': cases[0]['messages'], 'max_tokens': 32})
    results = []
    for case in cases:
        start = time.perf_counter()
        body = request('/v1/chat/completions', {'messages': case['messages'], 'max_tokens': case['max_tokens']})
        seconds = time.perf_counter() - start
        target = reference[case['name']]
        text = body['choices'][0]['message']['content']
        assert body['usage']['prompt_tokens'] == len(target['prompt_ids']), 'chat template/tokenizer mismatch'
        assert body['minnow']['processed_tokens'] % 32 == 0
        row = {'name': case['name'], 'text_equal': text == target['text'], 'text': text,
            'reference_text': target['text'], 'http_seconds': seconds,
            'reference_seconds': target['elapsed_seconds'], 'response': body}
        results.append(row)
        print(f'{case["name"]}: text_equal={row["text_equal"]}, {body["minnow"]["tokens_per_second"]:.1f} tokens/s', flush=True)
    body = request('/v1/completions', {'prompt': 'The capital of France is', 'max_tokens': 32})
    assert isinstance(body['choices'][0]['text'], str)
    assert request('/health')['status'] == 'ok'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'health': health, 'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
