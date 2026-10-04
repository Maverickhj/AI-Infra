"""Authored MoE/MLA mechanisms inside the shared two-layer decoder (CPU only).

This is not a checkpoint, distributed backend, or Bridge execution. Float64
reference math intentionally differs from the production fp32 router precision.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import torch
from torch import nn
from experiments.decoder_reference import TinyDecoder, FIXTURE as BASE, rms, sample_data, plain_trace

FIXTURE=Path(__file__).resolve().parents[1]/'content/fixtures/moe-mla-reference.json'
FAMILIES=('qwen3-moe','deepseek-v2-lite','deepseek-v3')


def authored_fixture():
    def mat(rows,cols,salt):
        return [[(((i+1)*17+(j+1)*11+salt*(i+j+2))%43-21)/55 for j in range(cols)] for i in range(rows)]
    families={}
    for family in FAMILIES:
        shared={'qwen3-moe':0,'deepseek-v2-lite':2,'deepseek-v3':1}[family]
        qrank=4 if family=='deepseek-v3' else None
        config=dict(attention='gqa' if family=='qwen3-moe' else 'mla',q_rank=qrank,
                    moe_layers=[0,1] if family=='qwen3-moe' else [1],experts=4,topk=2,expert_width=4,
                    shared_experts=shared,score='sigmoid' if family=='deepseek-v3' else 'softmax',
                    pre_softmax=family!='qwen3-moe',groups=2 if family=='deepseek-v3' else 1,
                    group_topk=1,scaling=2.5 if family=='deepseek-v3' else 1.,
                    aux_type='aux_loss' if family=='qwen3-moe' else 'seq_aux_loss',
                    aux_coeff=.0001 if family=='deepseek-v3' else .001,
                    bias_rate=.01,bias=[.08,-.04,.03,-.02] if family=='deepseek-v3' else None)
        params={}
        for l in range(2):
            k=lambda n:f'l{l}_{n}'
            if l in config['moe_layers']:
                params[k('router')]=mat(4,8,21+l)
                for e in range(4):
                    params[k(f'e{e}_gate_up')]=mat(8,8,30+e*3+l)
                    params[k(f'e{e}_down')]=mat(8,4,44+e*2+l)
                if shared:
                    params[k('shared_gate_up')]=mat(8*shared,8,63+l)
                    params[k('shared_down')]=mat(8,4*shared,65+l)
            if config['attention']=='mla':
                params[k('kv_down')]=mat(5,8,72+l)
                params[k('kv_gain')]=[.9,1.2,1.05]
                params[k('kv_up')]=mat(8,3,76+l)
                params[k('mla_out')]=mat(8,4,80+l)
                if qrank:
                    params[k('q_down')]=mat(4,8,82+l)
                    params[k('q_latent_gain')]=[1.,.8,1.1,1.2]
                    params[k('q_up')]=mat(8,4,86+l)
                else:params[k('q_direct')]=mat(8,8,89+l)
        families[family]=dict(config=config,parameters=params)
    return dict(schema_version=1,provenance='architecture_scaled_authored',weights_origin='authored; no pretrained weights',
                dimensions=dict(V=27,H=8,layers=2,mla_heads=2,d_nope=2,d_rope=2,d_value=2,kv_rank=3),
                assumptions=['B=1; two layers; dropout off; no MTP/capacity drop/z-loss',
                             'family route mechanisms from fixed Bridge/Core; authored small dimensions and bias',
                             'float64 math; no fp32 bitwise production equivalence',
                             'ordinary causal sequence; optional suffix padding compacted out of MoE computation',
                             'MLA standard RoPE with mscale=1; no YaRN length extension',
                             'EP1/2 ETP1/2 EDP1 in-memory reference only'],families=families)


def expert_groups(ep=1,etp=1,edp=1):
    if type(ep) is not int or ep not in (1,2) or type(etp) is not int or etp not in (1,2) or edp!=1:
        raise ValueError('verified expert groups require EP1/2, ETP1/2, EDP1')
    ranks=[]
    for e in range(ep):
        for t in range(etp):
            ranks.append(dict(rank=e*etp+t,ep=list(range(t,ep*etp,etp)),etp=list(range(e*etp,(e+1)*etp)),
                              edp=[e*etp+t],experts=list(range(e*4//ep,(e+1)*4//ep)),
                              ffn_columns=[t*4//etp,(t+1)*4//etp]))
    return ranks


def sorted_top(scores,k):
    # Stable authored tie policy; production torch.topk ties are not guaranteed.
    return torch.argsort(scores,dim=-1,descending=True,stable=True)[...,:k]


def route(logits,config,padding,fault='none'):
    if logits.ndim!=2 or logits.shape[1]!=4 or padding.shape!=logits.shape[:1] or padding.dtype!=torch.bool:
        raise ValueError('invalid router logits/padding')
    if not torch.isfinite(logits).all() or padding.all():raise ValueError('no valid router tokens or nonfinite logits')
    score=torch.sigmoid(logits) if config['score']=='sigmoid' else torch.softmax(logits,dim=-1)
    selection=score if config['score']=='sigmoid' or config['pre_softmax'] else logits
    bias=logits.new_tensor(config['bias']) if config['bias'] is not None else torch.zeros(4,dtype=logits.dtype)
    if config['bias'] is not None:selection=selection+bias
    if config['groups']>1:
        group=selection.reshape(-1,2,2)
        group_score=group.topk(config['topk']//config['group_topk'],dim=-1).values.sum(-1)
        selected_group=sorted_top(group_score,config['group_topk'])
        allowed=torch.zeros_like(group_score,dtype=torch.bool).scatter(1,selected_group,True).repeat_interleave(2,dim=-1)
        selection=selection.masked_fill(~allowed,-torch.inf)
    indices=sorted_top(selection,config['topk'])
    if config['score']=='sigmoid':
        chosen=score.gather(1,indices)
        weights=chosen/(chosen.sum(-1,keepdim=True)+1e-20)
    elif config['pre_softmax']:weights=score.gather(1,indices)
    else:weights=torch.softmax(logits.gather(1,indices),dim=-1)
    if fault=='wrong_weights':
        if config['score']=='sigmoid':weights=(score+bias).gather(1,indices)
        else:weights=score.gather(1,indices) if not config['pre_softmax'] else torch.softmax(logits.gather(1,indices),-1)
    weights=weights*config['scaling']
    # Explicit valid-token compaction policy. Core's padding argument itself masks
    # auxiliary scores/counts; it does not universally erase main routing probs.
    weights=weights.masked_fill(padding[:,None],0)
    routing=torch.zeros_like(logits).scatter(1,indices,weights)
    route_map=torch.zeros_like(logits,dtype=torch.bool).scatter(1,indices,True)&(~padding[:,None])
    aux_scores=score/(score.sum(-1,keepdim=True)+1e-20) if config['score']=='sigmoid' else score
    aux_indices=sorted_top(aux_scores,config['topk'])
    aux_map=torch.zeros_like(logits).scatter(1,aux_indices,1.)*(~padding[:,None])
    aux_scores=aux_scores*(~padding[:,None]);count=(~padding).sum()
    aux=config['aux_coeff']*4*(aux_scores.sum(0)*aux_map.sum(0)).sum()/(config['topk']*count.square())
    with torch.no_grad():
        counts=route_map.sum(0)
        updated=bias+torch.sign(counts.to(logits.dtype).mean()-counts)*config['bias_rate'] if config['bias'] is not None else bias
    return dict(logits=logits,scores=score,indices=indices,weights=weights,routing=routing,
                counts=counts,aux_counts=aux_map.sum(0),aux_scores=aux_scores,aux_loss=aux,
                bias=bias,updated_bias=updated,valid_tokens=count)


def swiglu(x,gate_up,down,etp=1):
    width=down.shape[1];parts=[]
    for t in range(etp):
        a=t*width//etp;b=(t+1)*width//etp
        gate=x@gate_up[a:b].T;up=x@gate_up[width+a:width+b].T
        parts.append((torch.nn.functional.silu(gate)*up)@down[:,a:b].T)
    return torch.stack(parts).sum(0)


def validate_dispatch(dispatch,indices,padding):
    expected={(t,int(e)) for t,row in enumerate(indices) if not bool(padding[t]) for e in row}
    actual=[(d['token'],d['expert']) for d in dispatch]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('duplicate or missing token/expert dispatch')


def moe(x,params,config,padding,ep=1,etp=1,fault='none'):
    expert_groups(ep,etp)
    r=route(x@params['router'].T,config,padding,fault)
    result=torch.zeros_like(x);dispatch=[]
    for rank in range(ep):
        for e in range(rank*4//ep,(rank+1)*4//ep):
            tokens=[t for t,row in enumerate(r['indices']) if not bool(padding[t]) and e in row]
            if not tokens:continue
            local=swiglu(x[tokens],params[f'e{e}_gate_up'],params[f'e{e}_down'],etp)
            weight=r['routing'][tokens,e]
            weighted=local*weight[:,None]
            result=result.index_add(0,torch.tensor(tokens),weighted)
            for i,t in enumerate(tokens):
                dispatch.append(dict(token=t,expert=e,rank=rank,weight=weight[i],output=local[i],weighted=weighted[i]))
    if fault=='duplicate_dispatch' and dispatch:dispatch.append(dispatch[0])
    validate_dispatch(dispatch,r['indices'],padding)
    shared=torch.zeros_like(x)
    if config['shared_experts']:
        shared=swiglu(x,params['shared_gate_up'],params['shared_down'])*(~padding[:,None])
    down=result+shared if fault!='missing_shared' else result
    return dict(down=down,moe=dict(**r,dispatch=dispatch,routed=result,shared=shared,combined=down))


def rotate2(x,positions):
    angle=positions.to(x.dtype)
    while angle.ndim<x.ndim-1:angle=angle.unsqueeze(-1)
    a,b=x.unbind(-1);return torch.stack([a*angle.cos()-b*angle.sin(),a*angle.sin()+b*angle.cos()],-1)


def mla(x,params,qrank,epsilon=1e-6,fault='none'):
    # Input norm is outside this function, exactly as in the common decoder.
    count=len(x);positions=torch.arange(count)
    qraw=x@params['q_down'].T if qrank else x
    qlatent=rms(qraw,params['q_latent_gain'],epsilon) if qrank else qraw
    q=(qlatent@params['q_up' if qrank else 'q_direct'].T).reshape(count,2,4)
    kvraw=x@params['kv_down'].T;latent=rms(kvraw[:,:3],params['kv_gain'],epsilon)
    q_nope=q[:,:,:2];q_rope=rotate2(q[:,:,2:],positions);k_rope=rotate2(kvraw[:,3:],positions)
    up=params['kv_up'].reshape(2,4,3);wk=up[:,:2];wv=up[:,2:]
    expanded=torch.einsum('tr,hdr->thd',latent,up);k=expanded[:,:,:2];v=expanded[:,:,2:]
    score=(torch.einsum('thd,shd->hts',q_nope,k)+torch.einsum('thd,sd->hts',q_rope,k_rope))/2
    allowed=torch.ones(count,count,dtype=torch.bool).tril()
    probs=torch.softmax(score.masked_fill(~allowed,-torch.inf),dim=-1)
    context=torch.einsum('hts,shd->thd',probs,v)
    absorbed_q=torch.einsum('thd,hdr->thr',q_nope,wk)
    used_latent=kvraw[:,:3] if fault=='wrong_norm' else latent
    positional=torch.einsum('thd,sd->hts',q_rope,k_rope)
    if fault=='missing_rope':positional=torch.einsum('thd,sd->hts',q[:,:,2:],kvraw[:,3:])
    scale=math.sqrt(5) if fault=='wrong_scale' else 2
    latent_scores=(torch.einsum('thr,sr->hts',absorbed_q,used_latent)+positional)/scale
    latent_probs=torch.softmax(latent_scores.masked_fill(~allowed,-torch.inf),dim=-1)
    latent_context=torch.einsum('hts,sr->thr',latent_probs,used_latent)
    absorbed=torch.einsum('thr,hdr->thd',latent_context,wv)
    projected=context.reshape(count,4)@params['mla_out'].T
    # A true incremental prefix read of normalized latent + rotated positional key.
    cache=[];cached=[]
    for t in range(count):
        cache.append(torch.cat([used_latent[t],k_rope[t]]))
        now=torch.stack(cache)
        logits=(absorbed_q[t]@now[:,:3].T+q_rope[t]@now[:,3:].T)/2
        if fault=='wrong_scale':logits=logits*(2/math.sqrt(5))
        probability=torch.softmax(logits,-1)
        z=probability@now[:,:3]
        cached.append(torch.einsum('hr,hdr->hd',z,wv))
    cached=torch.stack(cached)
    return dict(q_raw=qraw,q_latent=qlatent,kv_raw=kvraw,kv_latent=latent,q_nope=q_nope,q_rope=q_rope,
                k_rope=k_rope,k_expanded=k,v_expanded=v,probabilities=probs,context=context,
                absorbed_q=absorbed_q,latent_context=latent_context,absorbed=absorbed,cache=torch.stack(cache),
                cached_context=cached,projected=projected,
                max_absorption_error=(context-absorbed).abs().max(),max_cache_error=(context-cached).abs().max())


class FamilyDecoder(TinyDecoder):
    def __init__(self,family='deepseek-v3',ep=1,etp=1,fault='none'):
        if family not in FAMILIES:raise ValueError('unsupported family')
        expert_groups(ep,etp)
        source=json.loads(FIXTURE.read_text())['families'][family]
        fixture=json.loads(BASE.read_text());fixture['training_dropout']=0.
        fixture['parameters'].update(source['parameters'])
        super().__init__(fixture,tied=False)
        self.family=family;self.config=source['config'];self.ep=ep;self.etp=etp;self.fault=fault
        self.padding=None
        # No unused dense/expert or GQA/MLA parameters masquerade as trained paths.
        for l in range(2):
            names=['gate_up','down'] if l in self.config['moe_layers'] else []
            if self.config['attention']=='mla':names+=['qkv','out','q_gain','k_gain']
            for name in names:del self.p[f'l{l}_{name}']

    def layer_params(self,n):
        return {name[len(f'l{n}_'):]:value for name,value in self.p.items() if name.startswith(f'l{n}_')}

    def attention(self,x,n,parallel=None):
        if self.config['attention']=='gqa':return super().attention(x,n)
        params=self.layer_params(n);norm=rms(x,params['input_gain'],self.fixture['epsilon'])
        data=mla(norm,params,self.config['q_rank'],self.fixture['epsilon'],self.fault)
        self.mla_traces[n]=data
        return dict(norm=norm,residual=x+data['projected'])

    def feed_forward(self,norm,n,parallel=None):
        if n not in self.config['moe_layers']:return super().feed_forward(norm,n)
        return moe(norm,self.layer_params(n),self.config,self.padding,self.ep,self.etp,self.fault)

    def forward(self,ids,labels,mask,padding=None,**kwargs):
        if kwargs:raise ValueError('family reference has its own scoped EP/ETP; dense TP options are unsupported')
        if padding is None:padding=torch.zeros_like(ids,dtype=torch.bool)
        if padding.dtype!=torch.bool or padding.shape!=ids.shape or (mask[padding]!=0).any():raise ValueError('invalid padding mask or padded supervision')
        if padding.any() and not padding[padding.nonzero()[0,0]:].all():raise ValueError('only suffix padding supported')
        self.padding=padding;self.mla_traces={}
        result=super().forward(ids,labels,mask)
        aux=sum(result['layers'][l]['moe']['aux_loss'] for l in self.config['moe_layers'])
        result.update(aux_loss=aux,total_loss=result['loss']+aux,mla=[self.mla_traces.get(l) for l in range(2)])
        return result


def family_data(sample=1,padding=0):
    ids,labels,mask=sample_data(sample)
    valid=len(ids)
    if padding:
        ids=torch.cat([ids,torch.zeros(padding,dtype=torch.long)])
        labels=torch.cat([labels,torch.zeros(padding,dtype=torch.long)])
        mask=torch.cat([mask,torch.zeros(padding,dtype=torch.float64)])
    return ids,labels,mask,torch.arange(len(ids))>=valid


def references():
    results=[]
    for family in FAMILIES:
        for sample,padding in [(1,0),(1,2),(0,0)]:
            model=FamilyDecoder(family).eval()
            trace=model(*family_data(sample,padding))
            results.append(dict(family=family,sample=sample,padding=padding,trace=plain_trace(trace)))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-fixture',action='store_true');parser.add_argument('--forward',action='store_true')
    args=parser.parse_args()
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(authored_fixture(),indent=2)+'\n');print(str(FIXTURE))
    elif args.forward:print(json.dumps(references(),allow_nan=False))
