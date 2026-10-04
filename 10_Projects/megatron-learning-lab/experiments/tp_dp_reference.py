"""CPU in-memory TP/DP reference on the G03 model. No process group or NCCL."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]


def tensor(values):
    if isinstance(values,list):return torch.stack([tensor(v) for v in values])
    return values


def scalars(value):
    return [scalars(v) for v in value.unbind()] if value.ndim else value


class TensorParallelReference:
    def __init__(self,parameters,tp=2,fault='none',capture=False):
        if type(tp) is not int or tp not in (1,2):raise ValueError('TP must be 1 or 2; nq=4,nkv=2,F=12 must divide TP')
        if fault not in ('none','omit_reduce','concat_row','wrong_gate_pair','include_padded_vocab'):raise ValueError('unknown TP fault')
        self.p,self.tp,self.fault,self.capture=parameters,tp,fault,capture
        self.events=[]

    def record(self,name,layer,ranks,collective,outputs,inputs):
        if not self.capture:return
        row=name in ('attention_output','ffn_output')
        logical=torch.stack(outputs).sum(dim=0) if row else torch.cat(outputs,dim=-1)
        self.events.append(dict(name=name,layer=layer,group=list(range(self.tp)),collective=collective,
            global_shape=list(logical.shape),global_token7=logical[min(7,len(logical)-1)].detach().tolist(),
            ranks=[{**meta,'input_shape':list(x.shape),'input_token7':x[min(7,len(x)-1)].detach().tolist(),
                    'output_shape':list(value.shape),'token7':value[min(7,len(value)-1)].detach().tolist()}
                   for meta,value,x in zip(ranks,outputs,inputs)]))

    def qkv(self,values,layer):
        x=tensor(values);w=self.p[f'l{layer}_qkv'];width=w.shape[0]//self.tp
        ranks=[dict(rank=r,weight_shape=[width,8],weight_rows=[r*width,(r+1)*width],
                    layout='grouped Q,Q,K,V; two Q heads per KV group') for r in range(self.tp)]
        outputs=[x@w[r*width:(r+1)*width].T for r in range(self.tp)]
        self.record('qkv',layer,ranks,'column partition; local Q/K/V, no forward gather required for attention',outputs,[x]*self.tp)
        # Concatenation exposes logical head order to the shared per-head GQA graph.
        return scalars(torch.cat(outputs,dim=-1))

    def reduce(self,parts):
        if self.fault=='concat_row' and self.tp>1:raise ValueError('row projection requires SUM, concat gives wrong hidden shape')
        return parts[0] if self.fault=='omit_reduce' else torch.stack(parts).sum(dim=0)

    def output(self,values,layer):
        x=tensor(values);w=self.p[f'l{layer}_out'];width=x.shape[-1]//self.tp
        parts=[x[:,r*width:(r+1)*width]@w[:,r*width:(r+1)*width].T for r in range(self.tp)]
        self.record('attention_output',layer,[dict(rank=r,weight_shape=[8,width],input_columns=[r*width,(r+1)*width]) for r in range(self.tp)],'SUM of rank-local partial output; logical all-reduce',parts,[x[:,r*width:(r+1)*width] for r in range(self.tp)])
        return scalars(self.reduce(parts))

    def ffn(self,x,layer):
        w=self.p[f'l{layer}_gate_up'];down=self.p[f'l{layer}_down'];width=12//self.tp
        gates,ups,silus,products,parts=[],[],[],[],[]
        ranks=[]
        for r in range(self.tp):
            start,end=r*width,(r+1)*width
            local=w[r*2*width:(r+1)*2*width] if self.fault=='wrong_gate_pair' else torch.cat([w[start:end],w[12+start:12+end]])
            gate,up=(x@local.T).chunk(2,dim=-1)
            silu=torch.nn.functional.silu(gate);product=silu*up
            gates.append(gate);ups.append(up);silus.append(silu);products.append(product)
            parts.append(product@down[:,start:end].T)
            ranks.append(dict(rank=r,weight_shape=[2*width,8],gate_rows=[start,end],up_rows=[12+start,12+end],down_shape=[8,width]))
        self.record('ffn_pair',layer,ranks,'paired gate/up column partition; product stays local until row SUM',products,[x]*self.tp)
        self.record('ffn_output',layer,[dict(rank=r,weight_shape=[8,width],input_columns=[r*width,(r+1)*width]) for r in range(self.tp)],'SUM of row partials; no intermediate gate/up all-gather',parts,products)
        return *(torch.cat(items,dim=-1) for items in (gates,ups,silus,products)),self.reduce(parts)

    def vocab(self,x,head,labels):
        vocab=head.shape[0];padded=math.ceil(vocab/self.tp)*self.tp;width=padded//self.tp
        padded_weight=torch.nn.functional.pad(head,(0,0,0,padded-vocab))
        locals=[]
        for r in range(self.tp):
            local=x@padded_weight[r*width:(r+1)*width].T
            if self.fault!='include_padded_vocab':
                local=local.masked_fill(torch.arange(r*width,(r+1)*width)[None,:]>=vocab,-torch.inf)
            locals.append(local)
        maximum=torch.stack([v.max(dim=-1).values for v in locals]).max(dim=0).values
        denominator=torch.stack([torch.exp(v-maximum[:,None]).sum(dim=-1) for v in locals]).sum(dim=0)
        log_partition=maximum+torch.log(denominator)
        targets=[]
        for r,local in enumerate(locals):
            owner=(labels>=r*width)&(labels<(r+1)*width)
            local_label=torch.where(owner,labels-r*width,0)
            targets.append(torch.where(owner,local.gather(-1,local_label[:,None]).squeeze(-1),0.))
        self.target_loss=log_partition-torch.stack(targets).sum(dim=0)
        self.record('vocab',2,[dict(rank=r,weight_shape=[width,8],vocab_range=[r*width,(r+1)*width],valid_range=[r*width,min((r+1)*width,vocab)],padding_excluded=True) for r in range(self.tp)],
                    'MAX(logit), SUM(exp), SUM(owner target logit); full logits only reassembled for inspection',locals,[x]*self.tp)
        if self.capture:
            t=min(7,len(x)-1)
            self.events[-1]['global_shape']=[len(x),vocab]
            self.events[-1]['global_token7']=torch.cat(locals,dim=-1)[t,:vocab].detach().tolist()
            self.events[-1]['ce_reductions']=dict(global_max=float(maximum[t].detach()),global_exp_sum=float(denominator[t].detach()),target=int(labels[t]),target_logit=float(torch.stack(targets).sum(dim=0)[t].detach()),local_max=[float(v[t].max().detach()) for v in locals],local_exp_sum=[float(torch.exp(v[t]-maximum[t]).sum().detach()) for v in locals])
        logits=torch.cat(locals,dim=-1)[:,:vocab]
        return logits,logits-log_partition[:,None]


def compare_tp(tp=2,sample=0,tied=True):
    from experiments.decoder_reference import TinyDecoder,sample_data
    data=sample_data(sample)
    baseline=TinyDecoder(tied=tied).eval();parallel=TinyDecoder(tied=tied).eval()
    expected=baseline(*data);actual=parallel(*data,tp=tp,capture_parallel=True)
    torch.testing.assert_close(actual['loss'],expected['loss'],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(actual['token_loss'],expected['token_loss'],atol=1e-10,rtol=1e-10)
    expected['loss'].backward();actual['loss'].backward()
    max_gradient=0.
    for name,p in baseline.p.items():
        other=parallel.p[name]
        if p.grad is None:
            assert other.grad is None;continue
        torch.testing.assert_close(p.grad,other.grad,atol=1e-10,rtol=1e-10)
        max_gradient=max(max_gradient,float((p.grad-other.grad).abs().max()))
    torch.testing.assert_close(actual['logits'],expected['logits'],atol=1e-10,rtol=1e-10)
    # Parameter gradients are from the complete scalar loss, not local-loss proxies.
    for event in parallel.parallel_trace:
        op=event['name'];layer=event['layer']
        parameter={'qkv':f'l{layer}_qkv','attention_output':f'l{layer}_out',
                   'ffn_pair':f'l{layer}_gate_up','ffn_output':f'l{layer}_down',
                   'vocab':'embedding' if tied else 'head'}[op]
        w=parallel.p[parameter];g=w.grad
        event['parameter']=parameter;event['global_weight_shape']=list(w.shape)
        event['global_weight_preview']=w[:2,:4].detach().tolist()
        for rank in event['ranks']:
            if op=='qkv':
                a,b=rank['weight_rows'];local,grad=w[a:b],g[a:b]
            elif op in ('attention_output','ffn_output'):
                a,b=rank['input_columns'];local,grad=w[:,a:b],g[:,a:b]
            elif op=='ffn_pair':
                a,b=rank['gate_rows'];c,d=rank['up_rows']
                local=torch.cat([w[a:b],w[c:d]]);grad=torch.cat([g[a:b],g[c:d]])
            else:
                a,b=rank['vocab_range'];padding=math.ceil(27/tp)*tp-27
                local=torch.nn.functional.pad(w,(0,0,0,padding))[a:b]
                grad=torch.nn.functional.pad(g,(0,0,0,padding))[a:b]
            rank['weight_preview']=local[:2,:4].detach().tolist()
            rank['gradient_preview']=grad[:2,:4].detach().tolist()
    s=len(data[0]);payload=s*8*8 # FP64 S×H logical tensor per rank
    # Ring all-reduce theoretical bytes sent by EACH rank; no bandwidth/latency measurement.
    ring=2*(tp-1)/tp*payload
    return dict(tp=tp,sample=sample,tied=tied,loss=float(actual['loss'].detach()),max_gradient_error=max_gradient,
                max_output_error=float((actual['logits']-expected['logits']).abs().max().detach()),
                rank_events=parallel.parallel_trace,communication=dict(scope='theoretical per-rank sent bytes for one ring SUM of [S,H]; fp64=8 bytes',payload_bytes=payload,ring_allreduce_sent_bytes=ring,measured_time=None),
                vocabulary=dict(logical=27,physical=math.ceil(27/tp)*tp,padded_class_policy='excluded from this reference CE; not a claim about arbitrary Core defaults'))


def dp_reference(wrong_means=False):
    from experiments.decoder_reference import TinyDecoder,sample_data
    assignments=[[1,1],[2,0]] # same two accumulation microbatches, unequal valid token counts
    all_data=[[sample_data(i) for i in rank] for rank in assignments]
    total_count=sum(float(d[2].sum()) for rank in all_data for d in rank)
    baseline=TinyDecoder().eval()
    reference_sum=sum(baseline(*d)['loss']*d[2].sum() for rank in all_data for d in rank)
    reference_loss=reference_sum/total_count;reference_loss.backward()
    rank_models=[];stats=[]
    for rank,data in enumerate(all_data):
        model=TinyDecoder().eval();count=sum(float(d[2].sum()) for d in data)
        loss_sum=sum(model(*d)['loss']*d[2].sum() for d in data)
        local=(loss_sum/count/len(all_data)) if wrong_means else loss_sum/total_count
        local.backward();rank_models.append(model)
        stats.append(dict(rank=rank,samples=assignments[rank],microbatches=len(data),valid_tokens=count,loss_sum=float(loss_sum.detach()),local_mean=float((loss_sum/count).detach())))
    reduced={k:sum(m.p[k].grad for m in rank_models) for k,v in baseline.p.items() if v.grad is not None}
    error=max(float((reduced[k]-baseline.p[k].grad).abs().max()) for k in reduced)
    if not wrong_means:
        for k in reduced:torch.testing.assert_close(reduced[k],baseline.p[k].grad,atol=1e-10,rtol=1e-10)
    return dict(ranks=stats,global_count=total_count,global_loss=float(reference_loss.detach()),mean_of_means=sum(r['local_mean'] for r in stats)/len(stats),
                max_gradient_error=error,reduction='local summed token losses / global count, then SUM gradients (not mean again)',wrong_means=wrong_means,
                selected_gradient=dict(parameter='embedding[1,0]',reference=float(baseline.p['embedding'].grad[1,0]),reduced=float(reduced['embedding'][1,0])))


def export():
    cases=[compare_tp(tp,sample,tied) for tp in (1,2) for sample in (0,1,2) for tied in (True,False)]
    correct=dp_reference();incorrect=dp_reference(True)
    if incorrect['max_gradient_error']<1e-5:raise AssertionError('wrong local means not detected')
    paths=['experiments/tp_dp_reference.py','experiments/decoder_reference.py','experiments/gqa_reference.py','content/fixtures/decoder-reference.json','content/fixtures/sft-data.json']
    return dict(schema_version=1,provenance='reference_simulation',device='cpu',distributed_backend='not_run',cases=cases,dp=correct,wrong_dp=incorrect,
                source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                limits=['no process groups, NCCL, timing or GPU measurements','V27 padded to storage V28 at TP2 with explicit CE exclusion','EP is not multiplied into dense world size'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export');args=parser.parse_args()
    result=export()
    if args.export:
        path=ROOT/args.export;path.parent.mkdir(parents=True,exist_ok=True)
        # Masked padded vocabulary logits are represented as a string, never invalid JSON Infinity.
        def safe(v):
            if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
            if isinstance(v,list):return [safe(x) for x in v]
            return '-Infinity' if isinstance(v,float) and v==-math.inf else v
        path.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':'exported','cases':len(result['cases']),'max_gradient_error':max(c['max_gradient_error'] for c in result['cases']),'dp_error':result['dp']['max_gradient_error']}))
