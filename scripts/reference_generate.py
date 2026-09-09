#!/usr/bin/env python3
"""Run the original joint decoder with one directly loaded BF16 weight copy.

Run sequentially under memory_guard.py. No mmap, CPU model, complete state dict,
or full-model dtype/device conversion. FP32 forwards use layerwise_reference.py.
"""
import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from weight_io import WeightReader, available_memory, RESERVE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--cases', type=Path, default=Path('tests/generation_cases.json'))
    p.add_argument('--output', type=Path, default=Path('artifacts/reference-generation.json'))
    p.add_argument('--prefill-input', type=Path, help='also time prompt-to-KV construction from these token IDs')
    p.add_argument('--prefill-lengths', type=int, nargs='+', default=[512, 2048, 4096])
    p.add_argument('--prefill-iterations', type=int, default=5)
    p.add_argument('--prefill-only', action='store_true')
    p.add_argument('--generation-iterations', type=int, default=1)
    p.add_argument('--generation-warmups', type=int, default=0)
    p.add_argument('--attention', choices=['eager', 'sdpa'], default='eager')
    args = p.parse_args()
    if args.prefill_only and not args.prefill_input:
        p.error('--prefill-only requires --prefill-input')
    if not 1 <= args.prefill_iterations <= 100:
        p.error('--prefill-iterations must be 1–100')
    if not 1 <= args.generation_iterations <= 20 or not 0 <= args.generation_warmups <= 5:
        p.error('generation requires 1–20 iterations and 0–5 warmups')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    reader = WeightReader(args.model)
    try:
        required = sum(t[2] for t in reader.tensors.values())
        if available_memory() < required + RESERVE:
            raise RuntimeError('insufficient headroom for one BF16 reference model')
        source = str(args.model.resolve())
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)
        config._attn_implementation = args.attention
        config.use_cache = False
        cls = get_class_from_dynamic_module('modeling_llada2_moe.LLaDA2MoeModelLM', source)
        with torch.device('meta'):
            model = cls(config)
        # RoPE has a computed, non-persistent FP32 frequency buffer.
        module = sys.modules[cls.__module__]
        model.model.rotary_emb = module.LLaDA2MoeRotaryEmbedding(config=config, device='cuda')
        started = time.perf_counter()
        loaded = set()
        for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
            if name not in reader.tensors:
                if name in ('model.rotary_emb.inv_freq', 'model.rotary_emb.original_inv_freq'):
                    continue
                raise ValueError(f'missing weight: {name}')
            value = reader.read(name, torch.bfloat16, 'cuda')
            parent, _, leaf = name.rpartition('.')
            owner = model.get_submodule(parent) if parent else model
            if leaf in owner._parameters:
                owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
            else:
                owner._buffers[leaf] = value
            loaded.add(name)
        del value, owner
        if loaded != reader.tensors.keys():
            raise ValueError(f'unused weights: {reader.tensors.keys() - loaded}')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        results = []
        prefill_results = []
        cases = [] if args.prefill_only else json.loads(args.cases.read_text())
        with torch.inference_mode():
            if args.prefill_input:
                seed = json.loads(args.prefill_input.read_text())
                if not seed:
                    raise ValueError('prefill seed must not be empty')
                for n in args.prefill_lengths:
                    if not 0 < n <= 8192 or n % config.block_size:
                        raise ValueError('prefill lengths require whole blocks, up to 8192 tokens')
                    prompt = list(itertools.islice(itertools.cycle(seed), n))
                    times = []
                    for iteration in range(args.prefill_iterations + 2):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        inputs = torch.tensor([prompt], device='cuda')
                        positions = torch.arange(n, device='cuda')[None]
                        blocks = positions[0] // config.block_size
                        mask = torch.where(blocks[None, :] <= blocks[:, None], 0.0, -float('inf'))[None, None].to(torch.bfloat16)
                        output = model.model(inputs, attention_mask=mask, position_ids=positions, use_cache=True, return_dict=True)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        assert output.past_key_values.get_seq_length() == n
                        if iteration >= 2:
                            times.append(elapsed)
                        del output, inputs, mask, positions, blocks
                    mean = sum(times) / len(times)
                    prefill_results.append({'prompt_tokens': n, 'iterations': args.prefill_iterations, 'warmups': 2,
                        'seconds': times, 'mean_seconds': mean, 'tokens_per_second': n / mean,
                        'execution': 'one full prompt forward; use_cache=True; no vocabulary projection'})
                    print(f'reference prefill {n}: {n / mean:.1f} tokens/s', file=sys.stderr, flush=True)
            # Warm kernels without constructing an N-by-N generation mask.
            model(torch.tensor([[220] * 32], device='cuda'), use_cache=False)
            for case in cases:
                prompt = tokenizer.apply_chat_template(case['messages'], tokenize=False, add_generation_prompt=True)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(ids) + case['max_tokens'] > 1024:
                    raise ValueError('reference generation is limited to 1024 total tokens')
                inputs = torch.tensor([ids], device='cuda')
                runs, warm_ids = [], None
                special_ids = set(tokenizer.all_special_ids)
                for iteration in range(args.generation_warmups + args.generation_iterations):
                    torch.manual_seed(42)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = model.generate(inputs, gen_length=case['max_tokens'], eos_early_stop=True)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    generated = output[0].tolist()
                    text_tokens = sum(token not in special_ids for token in generated)
                    if warm_ids is None:
                        warm_ids = generated
                    if iteration >= args.generation_warmups:
                        runs.append({'generated_ids': generated, 'elapsed_seconds': seconds,
                            'text_tokens': text_tokens, 'matches_warmup': generated == warm_ids})
                    print(f'{case["name"]} iteration {iteration}: {text_tokens} text tokens in {seconds:.3f}s',
                          file=sys.stderr, flush=True)
                seconds = sum(r['elapsed_seconds'] for r in runs) / len(runs)
                results.append({'name': case['name'], 'prompt_ids': ids, 'generated_ids': generated,
                    'text': tokenizer.decode(generated, skip_special_tokens=True), 'elapsed_seconds': seconds,
                    'tokens_per_second': sum(len(r['generated_ids']) for r in runs) / sum(r['elapsed_seconds'] for r in runs),
                    'text_tokens_per_second': sum(r['text_tokens'] for r in runs) / sum(r['elapsed_seconds'] for r in runs),
                    'iterations': args.generation_iterations, 'warmups': args.generation_warmups, 'runs': runs})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'dtype': 'bfloat16', 'attention': model.config._attn_implementation,
            'prefill_input_sha256': hashlib.sha256(args.prefill_input.read_bytes()).hexdigest() if args.prefill_input else None,
            'load_seconds': load_seconds,
            'torch': torch.__version__, 'source_sha256': hashlib.sha256((args.model / 'modeling_llada2_moe.py').read_bytes()).hexdigest(),
            'results': results, 'prefill': prefill_results}, indent=2) + '\n')
    finally:
        reader.close()


if __name__ == '__main__':
    main()
