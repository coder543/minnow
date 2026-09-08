#!/usr/bin/env python3
"""Generate independent fixtures using the read-only checkpoint's actual classes.

Run with PYTHONDONTWRITEBYTECODE=1. All generated files go to --output.
Python/Transformers is a validation dependency, never a serving dependency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import sys
sys.dont_write_bytecode = True
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path.home() / "models/hf/inclusionAI/LLaDA2.2-mini")
    p.add_argument("--output", type=Path, default=Path("artifacts/tiny"))
    p.add_argument("--real", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--input", type=Path)
    p.add_argument("--prompt", default="Calculate 1+5-28*0.5-200=?")
    p.add_argument("--generate", type=int, default=0)
    p.add_argument("--max-post-steps", type=int, default=16)
    args = p.parse_args()
    if args.real:
        if args.generate:
            p.error('use reference_generate.py for bounded single-copy BF16 generation, or layerwise_reference.py for FP32 numerical validation')
        from layerwise_reference import run
        run(args)
        return
    if args.generate:
        p.error("tiny fixture generation does not support --generate")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    source = str(args.model.resolve())
    c = AutoConfig.from_pretrained(source, trust_remote_code=True)
    for key, value in dict(vocab_size=259, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, num_experts=8, num_experts_per_tok=2, expert_capacity=4,
            moe_intermediate_size=16, num_shared_experts=1, max_position_embeddings=256,
            pad_token_id=0, delete_token_id=256, split_token_id=257).items():
        setattr(c, key, value)
    c._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(c, trust_remote_code=True).to(dtype=dtype, device=args.device).eval()
    with torch.no_grad():
        for name, w in model.named_parameters():
            if "norm" in name:
                w.uniform_(0.7, 1.3)
            else:
                w.normal_(0.0, 0.1)
        for layer in model.model.layers[1:]:
            layer.mlp.gate.expert_bias.uniform_(-0.2, 0.2)
    c.save_pretrained(args.output)
    # No save_pretrained(model): avoid copying remote code or modifying source.
    save_file({k: v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, args.output / "model.safetensors")
    ids = [1 + (i * 13) % 240 for i in range(96)]
    ids[69:] = [255] * 27
    if args.input:
        ids = json.loads(args.input.read_text())
    (args.output / "input.json").write_text(json.dumps(ids))
    x = torch.tensor([ids], device=args.device)
    n = len(ids)
    blocks = torch.arange(n, device=args.device) // model.config.block_size
    mask = torch.where(blocks[None, :] <= blocks[:, None], 0.0, -float("inf"))[None, None].to(dtype)
    trace = {}
    handles = []
    def save(name, value):
        trace[name] = value.detach().squeeze(0).contiguous().cpu().clone()
    handles.append(model.model.word_embeddings.register_forward_hook(lambda m,a,o: save("embeddings", o)))
    handles.append(model.model.norm.register_forward_hook(lambda m,a,o: save("normalized", o)))
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a,i=i: save(f"layers.{i}.attention", a[0])))
        handles.append(layer.register_forward_hook(lambda m,a,o,i=i: save(f"layers.{i}.hidden", o[0])))
        if hasattr(layer.mlp, "gate"):
            def gate_hook(m,a,o,i=i):
                save(f"layers.{i}.router_logits", o[2])
                save(f"layers.{i}.router_ids", o[0].to(torch.int64))
            handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
    with torch.inference_mode():
        start = time.perf_counter()
        out = model(x, attention_mask=mask, position_ids=torch.arange(n, device=args.device)[None], use_cache=False)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        save("logits", out.logits)
    save_file(trace, args.output / "reference.safetensors")
    for h in handles:
        h.remove()
    metadata = {"dtype": args.dtype, "device": args.device, "forward_seconds_with_trace": elapsed,
        "torch": torch.__version__, "source_sha256": {name: hashlib.sha256((args.model/name).read_bytes()).hexdigest()
            for name in ["config.json", "modeling_llada2_moe.py", "configuration_llada2_moe.py"]}}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
