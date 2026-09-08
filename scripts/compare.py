#!/usr/bin/env python3
"""Report numerical differences without hiding routing or argmax changes."""
import argparse
import json
import torch
from pathlib import Path
from safetensors.torch import load
torch.set_num_threads(4)

p=argparse.ArgumentParser()
p.add_argument("reference")
p.add_argument("actual")
p.add_argument("--last",type=int)
args=p.parse_args()
a=load(Path(args.actual).read_bytes())
b=load(Path(args.reference).read_bytes())
report={}
for name in sorted(b):
    x,y=a[name],b[name]
    if args.last:
        x,y=x[-args.last:],y[-args.last:]
    assert x.shape==y.shape,(name,x.shape,y.shape)
    if name.endswith("router_ids"):
        x,y=x.long(),y.long()
        report[name]={"ordered_agreement":(x==y).float().mean().item(),
            "set_agreement":(x.sort(-1).values==y.sort(-1).values).all(-1).float().mean().item()}
        continue
    x,y=x.float(),y.float()
    d=x-y
    report[name]={"max_abs_error":d.abs().max().item(),"rms_error":d.square().mean().sqrt().item(),
        "relative_rms_error":(d.square().mean()/y.square().mean()).sqrt().item()}
    if name=="logits":
        report[name]["top1_agreement"]=(x.argmax(-1)==y.argmax(-1)).float().mean().item()
        report[name]["actual_top1"]=x.argmax(-1).tolist()
        report[name]["reference_top1"]=y.argmax(-1).tolist()
print(json.dumps(report,indent=2))
