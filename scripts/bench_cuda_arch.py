#!/usr/bin/env python3
"""Compare two already-validated minnow builds in A/B/B/A order.

Run under memory_guard.py with other inference servers unloaded. Each command
owns one model, exits, and releases it before the next command starts. Timed
sections exclude checkpoint loading and include the CLI's normal warmups.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import statistics
import time


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def summarize(report):
    results = {'prefill': [], 'cached': [], 'decode': []}
    for mode in results:
        runs = [r for r in report['runs'] if r['mode'] == mode]
        key = {'prefill': 'prompt_tokens', 'cached': 'prefix_tokens', 'decode': 'name'}[mode]
        names = list(dict.fromkeys(row[key] for run in runs for row in run['results']))
        for name in names:
            row = {key: name}
            identities = []
            for target in ['a', 'b']:
                measurements = [r for run in runs if run['target'] == target for r in run['results'] if r[key] == name]
                samples = []
                counts = []
                rounds = []
                for measurement in measurements:
                    if mode == 'prefill':
                        times = measurement['seconds']
                        tokens = [name] * len(times)
                    elif mode == 'cached':
                        times = [measurement['cached_forward_seconds']]
                        tokens = [1]
                    else:
                        times = [r['generation']['stats']['decode_seconds'] for r in measurement['runs']]
                        tokens = [r['text_tokens'] for r in measurement['runs']]
                        for r in measurement['runs']:
                            g = r['generation']
                            identity = {'token_ids': g['token_ids'], 'finish_reason': g['finish_reason'],
                                        'counters': {k: g['stats'][k] for k in ['blocks', 'denoise_forwards', 'commit_forwards', 'reused_commits', 'total_forwards', 'processed_tokens']}}
                            identities.append(identity)
                    rounds.append(sum(tokens) / sum(times))
                    samples.extend(times)
                    counts.extend(tokens)
                row[target] = {'seconds': samples, 'rate_per_second': sum(counts) / sum(samples),
                               'mean_seconds': statistics.mean(samples), 'stdev_seconds': statistics.stdev(samples),
                               'round_rates': rounds}
                if mode == 'decode':
                    row[target]['text_tokens'] = counts
            row['rate_change_percent'] = (row['b']['rate_per_second'] / row['a']['rate_per_second'] - 1) * 100
            if mode == 'decode':
                row['identical_tokens_and_work'] = all(i == identities[0] for i in identities)
                row['output_sha256'] = hashlib.sha256(json.dumps(identities[0]['token_ids']).encode()).hexdigest()
                row['counters'] = identities[0]['counters']
            results[mode].append(row)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--workspace-cache-mib', type=int, default=512)
    p.add_argument('--a-int8-expert-activations', action='store_true')
    p.add_argument('--b-int8-expert-activations', action='store_true')
    p.add_argument('--cases', type=Path, default=Path('tests/decode_natural_cases.json'))
    p.add_argument('--output', type=Path, default=Path('artifacts/arch-comparison'))
    p.add_argument('--prefill-iterations', type=int, default=5)
    p.add_argument('--decode-iterations', type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binaries = {'a': args.a.resolve(), 'b': args.b.resolve()}
    report = {
        'order': ['a', 'b', 'b', 'a'],
        'binaries': {k: {'path': str(v), 'sha256': digest(v)} for k, v in binaries.items()},
        'input_sha256': digest(args.input), 'cases_sha256': digest(args.cases),
        'model': str(args.model.resolve()),
        'workspace_cache_mib': args.workspace_cache_mib,
        'int8_expert_activations': {'a': args.a_int8_expert_activations, 'b': args.b_int8_expert_activations},
        'source_sha256': {str(f): digest(f) for f in sorted(Path('src').rglob('*')) if f.is_file()},
        'build_rs_sha256': digest('build.rs'),
        'runs': [],
    }
    commands = {
        'prefill': ['prefill-bench', '--input', str(args.input), '--tokens', '2048,4096', '--iterations', str(args.prefill_iterations)],
        'cached': ['bench', '--input', str(args.input), '--prefix-blocks', '0,32,128', '--cached-only', '--iterations', '30'],
        'decode': ['decode-bench', '--cases', str(args.cases), '--iterations', str(args.decode_iterations)],
    }
    for index, target in enumerate(report['order']):
        for mode, tail in commands.items():
            stem = args.output / f'{index + 1}-{target}-{mode}'
            command = [str(binaries[target]), '--model', str(args.model.resolve()),
                       '--workspace-cache-mib', str(args.workspace_cache_mib), *tail]
            if report['int8_expert_activations'][target]:
                command.insert(1, '--int8-expert-activations')
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm', '--format=csv,noheader'], text=True).strip()
            print(f'Round {index + 1}/4 target {target}: {mode}; GPU {gpu}', flush=True)
            start = time.monotonic()
            with stem.with_suffix('.json').open('w') as out, stem.with_suffix('.log').open('w') as err:
                subprocess.run(command, stdout=out, stderr=err, check=True)
            rows = json.loads(stem.with_suffix('.json').read_text())
            report['runs'].append({'round': index + 1, 'target': target, 'mode': mode,
                                   'command': command, 'gpu_before': gpu,
                                   'wall_seconds': time.monotonic() - start, 'results': rows})
            (args.output / 'raw.json').write_text(json.dumps(report, indent=2) + '\n')
            if mode == 'decode':
                for row in rows:
                    assert all(r['matches_warmup'] for r in row['runs']), row['name']
                    print(f"  {row['name']}: {row['mean_text_tokens']:.0f} text tokens, {row['decode_text_tokens_per_second']:.2f} decode text tok/s", flush=True)
            elif mode == 'prefill':
                for row in rows:
                    print(f"  {row['prompt_tokens']} tokens: {row['tokens_per_second']:.2f} tok/s", flush=True)
            else:
                for row in rows:
                    print(f"  {row['prefix_tokens']} prefix tokens: {row['cached_forward_seconds'] * 1000:.3f} ms/forward", flush=True)
    (args.output / 'summary.json').write_text(json.dumps(summarize(report), indent=2) + '\n')
    print('A/B/B/A comparison complete.', flush=True)


if __name__ == '__main__':
    main()
