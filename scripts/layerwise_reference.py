#!/usr/bin/env python3
"""Real reference forward with only ONE transformer layer resident at a time.

Uses the checkpoint's actual classes, but never constructs a complete model.
Run benchmarks separately from serving so memory and timings are controlled.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from weight_io import WeightReader


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    code_model = getattr(args, 'code_model', None) or args.model
    source = str(code_model.resolve())
    if args.model.is_file():
        if not args.code_model:
            raise ValueError('.mnw references require --code-model with upstream Python code/config assets')
        from quantized_reference import ContainerReader
        reader = ContainerReader(args.model, getattr(args, 'int8_expert_activations', False))
    else:
        if getattr(args, 'int8_expert_activations', False):
            raise ValueError('INT8 expert activations require a quantized .mnw checkpoint')
        reader = WeightReader(args.model)
    try:
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)
        config = config.__class__.from_dict(reader.config if args.model.is_file()
            else json.loads((args.model / 'config.json').read_text()))
        config._attn_implementation = 'eager'
        cls = get_class_from_dynamic_module('modeling_llada2_moe.LLaDA2MoeModelLM', source)
        module = sys.modules[cls.__module__]
        if args.input:
            prompt, prompt_ids = '', []
            ids = json.loads(args.input.read_text())
        else:
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}], tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            ids = prompt_ids + [156895] * (32 - len(prompt_ids) % 32)
        if not ids or len(ids) > 512 or len(ids) % config.block_size:
            raise ValueError('reference diagnostics require 1–512 tokens, aligned to the model block size')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'input.json').write_text(json.dumps(ids))
        (args.output / 'prompt.json').write_text(json.dumps({'text': prompt, 'input_ids': prompt_ids}, indent=2))
        trace = {}
        def save(name, value):
            trace[name] = value.detach().squeeze(0).contiguous().cpu().clone()
        def release():
            gc.collect()
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        started = time.perf_counter()
        with torch.inference_mode():
            x = torch.tensor([ids], device=args.device)
            embedding = reader.read('model.word_embeddings.weight', dtype, args.device)
            hidden = torch.nn.functional.embedding(x, embedding)
            del embedding
            release()
            save('embeddings', hidden)
            n = len(ids)
            blocks = torch.arange(n, device=args.device) // config.block_size
            mask = torch.where(blocks[None, :] <= blocks[:, None], 0.0, -float('inf'))[None, None].to(dtype)
            positions = torch.arange(n, device=args.device)[None]
            rope = module.LLaDA2MoeRotaryEmbedding(config=config, device=args.device)
            cos_sin = rope(hidden, positions)
            for i in range(config.num_hidden_layers):
                print(f'reference layer {i + 1}/{config.num_hidden_layers}', file=sys.stderr, flush=True)
                with torch.device('meta'):
                    layer = module.LLaDA2MoeDecoderLayer(config, i)
                # Assign directly into the final parameter/buffer slots. There is
                # no second state dict containing all of this layer's weights.
                for name, _ in list(layer.named_parameters()) + list(layer.named_buffers()):
                    value = reader.read(f'model.layers.{i}.{name}', dtype, args.device)
                    parent, _, leaf = name.rpartition('.')
                    owner = layer.get_submodule(parent) if parent else layer
                    if leaf in owner._parameters:
                        owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
                    else:
                        owner._buffers[leaf] = value
                    if args.model.is_file():
                        reader.configure_projection(f'model.layers.{i}.{name}', owner)
                del value, owner
                layer.eval()
                handles = [layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a: save(f'layers.{i}.attention', a[0]))]
                if hasattr(layer.mlp, 'gate'):
                    def gate_hook(m,a,o):
                        save(f'layers.{i}.router_logits', o[2])
                        save(f'layers.{i}.router_ids', o[0].long())
                    handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
                hidden = layer(hidden, attention_mask=mask, position_ids=positions, position_embeddings=cos_sin, use_cache=False)[0]
                save(f'layers.{i}.hidden', hidden)
                for h in handles:
                    h.remove()
                del layer, handles
                release()
            norm = module.LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            norm.weight = torch.nn.Parameter(reader.read('model.norm.weight', dtype, args.device), requires_grad=False)
            hidden = norm(hidden)
            save('normalized', hidden)
            head = reader.read('lm_head.weight', dtype, args.device)
            save('logits', torch.nn.functional.linear(hidden, head).float())
            del head, norm, hidden
            release()
        save_file(trace, args.output / 'reference.safetensors')
        metadata = {'dtype': args.dtype, 'device': args.device, 'execution': 'one-layer-at-a-time; direct I/O',
            'elapsed_seconds_including_io': time.perf_counter() - started, 'torch': torch.__version__,
            'source_sha256': {name: hashlib.sha256((code_model / name).read_bytes()).hexdigest()
                for name in ['config.json', 'modeling_llada2_moe.py', 'configuration_llada2_moe.py']}}
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2))
    finally:
        reader.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('artifacts/mini-layerwise'))
    p.add_argument('--input', type=Path)
    p.add_argument('--code-model', type=Path, help='upstream class/config assets; required for .mnw input')
    p.add_argument('--int8-expert-activations', action='store_true', help='independent W4A8/W8A8 group-dot oracle')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    p.add_argument('--prompt', default='Calculate 1+5-28*0.5-200=?')
    run(p.parse_args())
