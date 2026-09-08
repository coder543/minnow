#!/usr/bin/env python3
"""Run one validation command with a system-memory floor and record its peak.

This monitors total system available memory, including GB10 shared GPU memory,
because process RSS and nvidia-smi do not account for that allocation reliably.
Small swap-out events are tolerated; sustained reclaim pressure is not.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from memory_pressure import Guard, Limits


def memory():
    info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=3}
    swaps={line.split()[0]:int(line.split()[1]) for line in Path('/proc/vmstat').read_text().splitlines() if line.startswith(('pswpin ','pswpout '))}
    result={'available_bytes':info['MemAvailable'],'used_bytes':info['MemTotal']-info['MemAvailable'],**swaps}
    try:
        line=next(line for line in Path('/proc/pressure/memory').read_text().splitlines() if line.startswith('full '))
        result['full_stall_us']=int(dict(field.split('=') for field in line.split()[1:])['total'])
    except (OSError,StopIteration,KeyError):
        result['full_stall_us']=None
    return result


def stop_child(child):
    def signal_group(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass  # The command may exit between poll() and killpg().
    signal_group(signal.SIGTERM)
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        signal_group(signal.SIGKILL)
        child.wait()


p=argparse.ArgumentParser()
p.add_argument('--report',type=Path,required=True)
for key,default in asdict(Limits()).items():
    p.add_argument('--'+key.replace('_','-'),type=float,default=default)
p.add_argument('command',nargs=argparse.REMAINDER)
args=p.parse_args()
limits=Limits(**{key:getattr(args,key) for key in asdict(Limits())})
limits.critical_reserve_gib=min(limits.critical_reserve_gib,limits.reserve_gib)
try:
    limits.validate()
except ValueError as error:
    p.error(str(error))
command=args.command[1:] if args.command[:1]==['--'] else args.command
if not command:
    p.error('a command is required after --')
baseline=memory()
if baseline['available_bytes'] < args.reserve_gib*1024**3:
    p.error('not enough available memory to start')
child=subprocess.Popen(command,start_new_session=True)
guard=Guard(baseline,limits)
peak=baseline['used_bytes']
reason=None
started=time.monotonic()
try:
    while child.poll() is None:
        sample=memory()
        peak=max(peak,sample['used_bytes'])
        reason=guard.check(sample,time.monotonic()-started)
        if reason:
            stop_child(child)
            break
        time.sleep(0.1)
except KeyboardInterrupt:
    reason='interrupted'
finally:
    if child.poll() is None:
        stop_child(child)
    code=child.wait()
final=memory()
peak=max(peak,final['used_bytes'])
report={'command':command,'exit_code':code,'abort_reason':reason,'elapsed_seconds':time.monotonic()-started,
    'baseline':baseline,'peak_used_bytes':peak,'peak_growth_bytes':peak-baseline['used_bytes'],'final':final,
    'limits':asdict(limits),'last_pressure':guard.metrics,
    'peak_swap_out_bytes_per_second':guard.peak_swap_rate,'peak_full_stall_percent':guard.peak_stall_percent,
    'swap_out_bytes':max(0,final['pswpout']-baseline['pswpout'])*guard.page_size}
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),file=sys.stderr)
raise SystemExit(code if code else (1 if reason else 0))
