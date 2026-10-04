"""Logical 1F1B and dense attention oracle on the same authored model/data.

No GPU, PP processes, TE kernels or timing measurements.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from experiments.sft_data_reference import calculate
from experiments.gqa_reference import compute, authored_fixture

ROOT=Path(__file__).resolve().parents[1]


def pipeline(pp=2,microbatches=4):
    if type(pp) is not int or pp not in (1,2) or type(microbatches) is not int or not 1<=microbatches<=8:
        raise ValueError('two-layer model supports PP1/2 and 1–8 microbatches')
    order={}
    for stage in range(pp):
        warm=min(pp-stage-1,microbatches)
        tasks=[('F',m,'warmup') for m in range(warm)]
        tasks += [task for i in range(microbatches-warm) for task in [('F',warm+i,'steady'),('B',i,'steady')]]
        tasks += [('B',m,'cooldown') for m in range(microbatches-warm,microbatches)]
        order[stage]=tasks
    previous={}
    phases={}
    for stage,tasks in order.items():
        for i,(kind,m,phase) in enumerate(tasks):
            key=(stage,kind,m);phases[key]=phase
            previous[key]=(stage,*tasks[i-1][:2]) if i else None
    done={}
    active=set()
    def visit(key):
        if key in done:return done[key]
        if key in active:raise ValueError('pipeline dependency cycle')
        active.add(key)
        stage,kind,m=key
        deps=[previous[key]] if previous[key] else []
        if kind=='F' and stage>0:deps.append((stage-1,'F',m))
        if kind=='B':
            deps.append((stage,'F',m))
            if stage+1<pp:deps.append((stage+1,'B',m))
        start=max([visit(d)['end'] for d in deps],default=0)
        value=dict(stage=stage,kind=kind,microbatch=m,phase=phases[key],start=start,end=start+2//pp,layers=list(range(stage*2//pp,(stage+1)*2//pp)))
        active.remove(key);done[key]=value
        return value
    for key in previous:visit(key)
    events=sorted(done.values(),key=lambda e:(e['start'],e['stage']))
    duration=max(e['end'] for e in events)
    lives=[dict(stage=s,microbatch=m,start=done[(s,'F',m)]['start'],end=done[(s,'B',m)]['end']) for s in range(pp) for m in range(microbatches)]
    peak=[max(sum(v['stage']==s and v['start']<=t<v['end'] for v in lives) for t in range(duration)) for s in range(pp)]
    idle=pp*duration-sum(e['end']-e['start'] for e in events)
    return dict(events=events,duration=duration,lifetimes=lives,peak_activations=peak,idle_slots=idle,bubble=idle/(pp*duration))


def partition(boundaries,cp):
    if type(cp) is not int or cp not in (1,2):raise ValueError('CP must be 1 or 2')
    if len(boundaries)<2 or boundaries[0]!=0 or any(type(x) is not int for x in boundaries) or any(a>=b for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('invalid cumulative physical lengths')
    if cp==1:return [list(range(boundaries[-1]))]
    ranks=[[] for _ in range(cp)]
    for lo,hi in zip(boundaries,boundaries[1:]):
        if (hi-lo)%(2*cp):raise ValueError('every physical document length must divide 2*CP; choose padding=8')
        width=(hi-lo)//(2*cp)
        for r in range(cp):
            for chunk in (r,2*cp-r-1):
                ranks[r].extend(range(lo+chunk*width,lo+(chunk+1)*width))
    return ranks


def cp_reference(layout='ordinary',padding=1,cp=2,fault='none'):
    if layout not in ('ordinary','thd') or fault not in ('none','local_kv','leak'):raise ValueError('unknown layout/fault')
    data=json.loads((ROOT/'content/fixtures/sft-data.json').read_text())
    model=json.loads((ROOT/'content/fixtures/decoder-reference.json').read_text())
    meta=calculate(data,indices=[0] if layout=='ordinary' else [1,2],padding=padding)
    shards=partition(meta['cuSeqlensPadded'] if layout=='thd' else [0,len(meta['ids'])],cp)
    f=authored_fixture();w=model['parameters']
    f.update(dimensions=dict(B=1,S=len(meta['ids']),H=8,nq=4,nkv=2,d=4),
             x=[w['embedding'][i] for i in meta['ids']],positions=meta['positions'],epsilon=model['epsilon'],theta=model['theta'],
             wqkv=list(map(list,zip(*w['l0_qkv']))),wo=list(map(list,zip(*w['l0_out']))),
             input_gain=w['l0_input_gain'],q_gain=w['l0_q_gain'],k_gain=w['l0_k_gain'])
    trace=compute(f)
    outputs=[]
    for t in range(len(meta['ids'])):
        heads=[]
        for h in range(4):
            if not meta['valid'][t]:heads.append([0.]*4);continue
            owned=next(indices for indices in shards if t in indices)
            keys=[j for j in range(len(meta['ids'])) if meta['valid'][j] and j<=t
                  and (fault=='leak' or meta['segments'][j]==meta['segments'][t])
                  and (fault!='local_kv' or j in owned)]
            scores=[math.fsum(a*b for a,b in zip(trace['qrope'][t][h],trace['krope'][j][h//2]))/2 for j in keys]
            maximum=max(scores);exps=[math.exp(x-maximum) for x in scores];total=math.fsum(exps)
            heads.append([math.fsum(e*trace['v'][j][h//2][d] for e,j in zip(exps,keys))/total for d in range(4)])
        outputs.append(heads)
    return dict(outputs=outputs,shards=shards,positions=meta['positions'],valid=meta['valid'],segments=meta['segments'],cuSeqlens=meta['cuSeqlens'],cuSeqlensPadded=meta['cuSeqlensPadded'])


def sp_reference(layer=0):
    if layer not in (0,1):raise ValueError('invalid layer')
    from experiments.decoder_reference import TinyDecoder,sample_data
    trace=TinyDecoder().eval()(*sample_data())
    return {k:trace['layers'][layer][name].detach().tolist() for k,name in [('input','attention'),('norm','ffn_norm'),('output','down')]}


def main():
    requests=json.load(sys.stdin)
    result=[]
    for r in requests:
        r=dict(r);kind=r.pop('kind')
        result.append(pipeline(**r) if kind=='pipeline' else cp_reference(**r) if kind=='cp' else sp_reference(**r) if kind=='sp' else None)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
