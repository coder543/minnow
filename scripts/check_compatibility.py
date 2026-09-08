#!/usr/bin/env python3
"""Exercise real HTTP/SSE chat, tool round trips, live progress, and cancellation.

Run under memory_guard.py. --spawn loads one model and always terminates it.
Without --spawn, --url can target a running minnow or its llama-swap upstream URL.
"""
import argparse
import copy
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--spawn', action='store_true')
    p.add_argument('--model', type=Path, help='checkpoint directory or .mnw file for --spawn')
    p.add_argument('--ui-dir', type=Path)
    p.add_argument('--report',type=Path,default=Path('artifacts/api-compatibility.json'))
    args=p.parse_args()
    if args.spawn and not args.model: p.error("--spawn requires --model")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    process=None
    logfile=None
    if args.spawn:
        with socket.socket() as s:
            s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
        args.url=f'http://127.0.0.1:{port}'
        command=['target/release/minnow','--prefill-chunk-tokens','128','serve','--listen',f'127.0.0.1:{port}','--threshold','0.5','--editing-threshold','0','--max-post-steps','16']
        if args.model: command += ['--model',str(args.model)]
        if args.ui_dir: command+=['--ui-dir',str(args.ui_dir)]
        logfile=args.report.with_suffix('.server.log').open('w')
        process=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT)

    def request(path,payload=None,status=200):
        req=Request(args.url+path,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
        try: response=urlopen(req,timeout=180)
        except HTTPError as e: response=e
        with response:
            data=response.read()
            assert response.status==status,(path,response.status,data[:2000])
            return json.loads(data)

    def stream(payload):
        payload={**payload,'stream':True,'stream_options':{'include_usage':True},'return_progress':True}
        started=time.monotonic(); events=[]; done=False
        with urlopen(Request(args.url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
            assert response.headers['Content-Type'].startswith('text/event-stream')
            assert response.headers.get('X-Accel-Buffering')=='no'
            for line in response:
                if not line.startswith(b'data:'): continue
                data=line[5:].strip()
                if data==b'[DONE]': done=True; break
                event=json.loads(data); assert 'error' not in event,event
                events.append({'seconds':time.monotonic()-started,'event':event})
        assert done
        assert len({e['event']['id'] for e in events})==1
        assert len({e['event']['created'] for e in events})==1
        chunks=[e['event'] for e in events]
        text=''.join((e.get('choices') or [{}])[0].get('delta',{}).get('content','') or '' for e in chunks)
        calls={}
        for e in chunks:
            for call in (e.get('choices') or [{}])[0].get('delta',{}).get('tool_calls',[]):
                i=call['index']; current=calls.setdefault(i,{'id':'','type':'function','function':{'name':'','arguments':''}})
                if 'id' in call: current['id']=call['id']
                for k in ['name','arguments']: current['function'][k]+=call.get('function',{}).get(k,'')
        final=next(e for e in reversed(chunks) if (e.get('choices') or [{}])[0].get('finish_reason'))
        assert chunks[-1]['choices']==[] and chunks[-1]['usage']
        meta=final['minnow']
        assert meta['evaluated_tokens']==32*meta['denoise_forwards']
        assert meta['evaluated_tokens']==sum(b['evaluated_tokens'] for b in meta['batches'])
        assert meta['completion_tokens']==sum(b['completion_tokens'] for b in meta['batches'])
        assert abs(meta['refinement_steps_per_block']-meta['denoise_forwards']/meta['blocks'])<1e-8
        if meta['prefill_tokens']:
            progress=[e['prompt_progress'] for e in chunks if 'prompt_progress' in e]
            assert progress[0]['processed']==meta['cached_tokens'] and progress[-1]['processed']==progress[-1]['total']
            assert all(e['cache']==meta['cached_tokens'] for e in progress)
            assert [e['processed'] for e in progress]==sorted(e['processed'] for e in progress)
            final_prefill=max(i for i,e in enumerate(chunks) if 'prompt_progress' in e)
            assert chunks[final_prefill+1]['minnow']['phase']=='prefill_complete'
            assert 'prompt_progress' not in chunks[final_prefill+1]
        for e in chunks:
            delta=(e.get('choices') or [{}])[0].get('delta',{})
            if delta.get('content') or delta.get('tool_calls'):
                assert 'prompt_progress' not in e
                assert e['minnow']['phase'] in ['block','complete']
                assert e['timings']['predicted_n']>0
        return {'text':text,'tool_calls':list(calls.values()),'final':final,'usage':chunks[-1]['usage'],'events':events}

    try:
        for _ in range(1200):
            if process and process.poll() is not None: raise RuntimeError('server exited; see server log')
            try:
                health=request('/health'); break
            except (OSError,AssertionError): time.sleep(0.1)
        else: raise RuntimeError('server startup timed out')
        model=health['model']
        props=request('/props'); assert props['role']=='model' and props['minnow']['generation_defaults']['threshold']==0.5
        assert request('/v1/models')['data'][0]['id']==model
        assert request('/v1/models/'+model)['id']==model
        assert request('/v1/streams/lookup',{'conversation_ids':[]})==[]
        sample='Hello, Montréal and 東京!'
        ids=request('/tokenize',{'content':sample})['tokens']; assert request('/detokenize',{'tokens':ids})['content']==sample
        base={'model':model,'messages':[{'role':'user','content':'What is the capital of France?'}],'max_tokens':64}
        factual=request('/v1/chat/completions',base)
        streamed=stream(base)
        assert streamed['text']==factual['choices'][0]['message']['content']
        for key in ['prompt_tokens','completion_tokens','total_tokens']:
            assert streamed['usage'][key]==factual['usage'][key]
        assert streamed['usage']['prompt_tokens_details']['cached_tokens']==streamed['final']['minnow']['cached_tokens']
        assert streamed['final']['minnow']['denoise_forwards']==factual['minnow']['denoise_forwards']
        assert '<|' not in streamed['text']
        stopped=stream({**base,'stop':'Paris'})
        assert 'Paris' not in stopped['text'] and stopped['final']['choices'][0]['finish_reason']=='stop'
        override=request('/v1/chat/completions',{**base,'max_completion_tokens':32,'minnow':{'max_post_steps':2},'threshold':0.6,'editing_threshold':0.1})
        o=override['minnow']['generation_settings']; assert abs(o['threshold']-0.6)<1e-6 and abs(o['editing_threshold']-0.1)<1e-6 and o['max_post_steps']==2
        request('/v1/chat/completions',{**base,'threshold':2},400)
        request('/v1/chat/completions',{**base,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:...'}}]}]},400)
        request('/v1/chat/completions',{**base,'max_tokens':1000000,'stream':True},400)
        request('/v1/chat/completions',{**base,'logprobs':True},400)
        # Several prefill batches and several finalized output blocks.
        long=stream({**base,'messages':[{'role':'user','content':'Reference notes: '+('Hash tables map keys to values and use collision handling. '*800)+'\nWrite a detailed explanation of hash tables, collisions, and resizing.'}],'max_tokens':128})
        assert long['final']['minnow']['prompt_tokens']>8192
        assert long['final']['minnow']['blocks']>=4
        assert len([e for e in long['events'] if 'prompt_progress' in e['event']])>=2
        block_events=[e for e in long['events'] if e['event'].get('minnow',{}).get('phase')=='block']
        assert block_events[0]['seconds']<block_events[-1]['seconds']
        tool={'type':'function','function':{'name':'get_weather','description':'Get the current weather for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}
        tool_base={'model':model,'messages':[{'role':'user','content':'What is the weather in Paris? Use the get_weather tool.'}],'tools':[tool],'tool_choice':{'type':'function','function':{'name':'get_weather'}},'max_tokens':256}
        tool_stream=stream(tool_base)
        assert tool_stream['final']['choices'][0]['finish_reason']=='tool_calls',tool_stream
        assert len(tool_stream['tool_calls'])==1
        call=tool_stream['tool_calls'][0]
        assert call['function']['name']=='get_weather'
        assert json.loads(call['function']['arguments'])['city'].lower()=='paris'
        tool_plain=request('/v1/chat/completions',tool_base)
        plain_calls=tool_plain['choices'][0]['message']['tool_calls']
        assert plain_calls[0]['function']==call['function']
        history=tool_base['messages']+[{'role':'assistant','content':tool_stream['text'] or None,'tool_calls':[call]},{'role':'tool','tool_call_id':call['id'],'content':'{"city":"Paris","temperature_c":21,"conditions":"sunny"}'}]
        template=request('/apply-template',{'messages':history,'tools':[tool]})['prompt']
        assert '## Return of functions.get_weather:0' in template
        followup=stream({'model':model,'messages':history,'tools':[tool],'tool_choice':'none','max_tokens':128})
        assert not followup['tool_calls'] and followup['text']
        # Close an in-flight request after observing refinement. The worker must
        # release its slot rather than finishing the long response in the background.
        cancel={**base,'messages':[{'role':'user','content':'Write a long tutorial about programming.'}],'max_tokens':2048,'stream':True,'return_progress':True}
        with urlopen(Request(args.url+'/v1/chat/completions',data=json.dumps(cancel).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
            for line in response:
                if line.startswith(b'data:'):
                    e=json.loads(line[5:]);
                    if e.get('minnow',{}).get('phase')=='refinement': break
        started=time.monotonic()
        for _ in range(100):
            if not any(s['is_processing'] for s in request('/slots')):break
            time.sleep(0.05)
        else:raise AssertionError('disconnected request retained its inference slot')
        report={'url':args.url,'health':health,'props':props,'factual':factual,'streamed':streamed,'stopped':stopped,'override':override,'long':long,'tools':tool_stream,'tool_plain':tool_plain,'tool_followup':followup,'cancel_seconds':time.monotonic()-started}
        if args.ui_dir:
            with urlopen(args.url+'/',timeout=30) as response:
                assert response.read()==(args.ui_dir/'index.html').read_bytes()
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'url':args.url,'tool_call':call,'tool_followup':followup['text'],'cancel_seconds':report['cancel_seconds']}),flush=True)
    finally:
        if process:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill();process.wait();raise
            assert process.returncode==0,process.returncode
        if logfile: logfile.close()

if __name__=='__main__':main()
