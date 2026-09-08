#!/usr/bin/env python3
"""Run one validation command with a system-memory floor and record its peak.

This monitors total system available memory, including GB10 shared GPU memory,
because process RSS and nvidia-smi do not account for that allocation reliably.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def memory():
    info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=3}
    swaps={line.split()[0]:int(line.split()[1]) for line in Path('/proc/vmstat').read_text().splitlines() if line.startswith(('pswpin ','pswpout '))}
    return {'available_bytes':info['MemAvailable'],'used_bytes':info['MemTotal']-info['MemAvailable'],**swaps}


p=argparse.ArgumentParser()
p.add_argument('--report',type=Path,required=True)
p.add_argument('--reserve-gib',type=float,default=20)
p.add_argument('--max-growth-gib',type=float,default=40)
p.add_argument('command',nargs=argparse.REMAINDER)
args=p.parse_args()
command=args.command[1:] if args.command[:1]==['--'] else args.command
if not command:
    p.error('a command is required after --')
baseline=memory()
if baseline['available_bytes'] < args.reserve_gib*1024**3:
    p.error('not enough available memory to start')
child=subprocess.Popen(command,start_new_session=True)
peak=baseline['used_bytes']
reason=None
started=time.monotonic()
try:
    while child.poll() is None:
        sample=memory()
        peak=max(peak,sample['used_bytes'])
        if sample['available_bytes'] < args.reserve_gib*1024**3:
            reason='system memory reserve reached'
        elif sample['used_bytes']-baseline['used_bytes'] > args.max_growth_gib*1024**3:
            reason='maximum allowed system-memory growth exceeded'
        elif sample['pswpout'] > baseline['pswpout']:
            reason='new swap-out activity detected'
        if reason:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
            break
        time.sleep(0.1)
except KeyboardInterrupt:
    reason='interrupted'
finally:
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
    code=child.wait()
report={'command':command,'exit_code':code,'abort_reason':reason,'elapsed_seconds':time.monotonic()-started,
    'baseline':baseline,'peak_used_bytes':peak,'peak_growth_bytes':peak-baseline['used_bytes'],'final':memory()}
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),file=sys.stderr)
raise SystemExit(code if code else (1 if reason else 0))
