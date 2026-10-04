"""Two-layer architecture-scaled Qwen-style CPU reference using the G01 GQA graph.

No checkpoint, GPU, production trainer or model generation. Weight storage is
[out,in]; the GQA adapter passes logical [in,out] lists without detaching.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from experiments.gqa_reference import compute_graph, authored_fixture, validate
from experiments.sft_data_reference import FIXTURE as DATA_FIXTURE, encode

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'content/fixtures/decoder-reference.json'
torch.set_num_threads(1)


def authored_model():
    def mat(rows,cols,salt):
        return [[(((i+1)*13+(j+1)*7+salt*(i+2))%31-15)/90 for j in range(cols)] for i in range(rows)]
    params = {'embedding': mat(27,8,1), 'final_gain':[1+i/40 for i in range(8)]}
    for layer in range(2):
        params.update({f'l{layer}_'+name:value for name,value in {
            'qkv':mat(32,8,2+layer), 'out':mat(8,16,4+layer),
            'input_gain':[1+i/30 for i in range(8)], 'q_gain':[1,1.1,.9,1.2],
            'k_gain':[.95,1.05,1.15,.85], 'ffn_gain':[1-i/50 for i in range(8)],
            'gate_up':mat(24,8,6+layer),'down':mat(8,12,8+layer)
        }.items()})
    return {'schema_version':1,'provenance':'architecture_scaled_authored','weights_origin':'authored, not a Qwen checkpoint',
            'dimensions':{'V':27,'H':8,'F':12,'layers':2,'nq':4,'nkv':2,'d':4},
            'epsilon':1e-6,'theta':10000,'weight_storage':'out_in','default_tied':True,
            'training_dropout':0.1,'frozen':['l0_input_gain'],'optimizer':{'type':'SGD','lr':.03,'momentum':.9},
            'parameters':params}


def sample_data(index=0,mode='assistant'):
    fixture=json.loads(DATA_FIXTURE.read_text())
    sequence=encode(fixture,fixture['samples'][index],mode)
    ids=torch.tensor([v[0] for v in sequence],dtype=torch.long)
    labels=torch.tensor([v[0] for v in sequence[1:]]+[0],dtype=torch.long)
    mask=torch.tensor([int(v[1]) for v in sequence[1:]]+[0],dtype=torch.float64)
    return ids,labels,mask


def nested_tensor(values):
    if isinstance(values,list):
        return torch.stack([nested_tensor(v) for v in values])
    return values if isinstance(values,torch.Tensor) else torch.tensor(values,dtype=torch.float64)


def scalar_lists(tensor):
    return [scalar_lists(v) for v in tensor.unbind()] if tensor.ndim else tensor


def rms(x,gain,epsilon):
    return x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+epsilon)*gain


class TinyDecoder(nn.Module):
    def __init__(self,fixture=None,tied=True):
        super().__init__()
        self.fixture=fixture or json.loads(FIXTURE.read_text())
        self.tied=tied
        values=copy.deepcopy(self.fixture['parameters'])
        if not tied: values['head']=copy.deepcopy(values['embedding'])
        self.p=nn.ParameterDict({k:nn.Parameter(torch.tensor(v,dtype=torch.float64),requires_grad=k not in self.fixture['frozen']) for k,v in values.items()})

    def attention(self,x,n,parallel=None):
        key=lambda name:f'l{n}_{name}'
        f=authored_fixture()
        f.update(dimensions={'B':1,'S':len(x),'H':8,'nq':4,'nkv':2,'d':4},
                 x=scalar_lists(x),positions=list(range(len(x))),
                 epsilon=self.fixture['epsilon'],theta=self.fixture['theta'],
                 wqkv=scalar_lists(self.p[key('qkv')].T),wo=scalar_lists(self.p[key('out')].T),
                 input_gain=scalar_lists(self.p[key('input_gain')]),
                 q_gain=scalar_lists(self.p[key('q_gain')]),k_gain=scalar_lists(self.p[key('k_gain')]))
        # Validate a detached view, then evaluate that same G01 graph with autograd scalars.
        def plain(v):
            if isinstance(v,dict): return {k:plain(x) for k,x in v.items()}
            if isinstance(v,list): return [plain(x) for x in v]
            return float(v.detach()) if isinstance(v,torch.Tensor) else v
        validate(plain(f))
        gqa=compute_graph(f,sqrt=torch.sqrt,exp=lambda v:torch.exp(v) if isinstance(v,torch.Tensor) else math.exp(v),
                          qkv_project=(lambda value: parallel.qkv(value,n)) if parallel else None,
                          output_project=(lambda value: parallel.output(value,n)) if parallel else None)
        return dict(norm=nested_tensor(gqa['norm']),residual=nested_tensor(gqa['residual']))

    def feed_forward(self,norm,n,parallel=None):
        key=lambda name:f'l{n}_{name}'
        if parallel:
            gate,up,silu,product,down=parallel.ffn(norm,n)
        else:
            gate,up=(norm@self.p[key('gate_up')].T).chunk(2,dim=-1)
            silu=torch.nn.functional.silu(gate)
            product=silu*up
            down=product@self.p[key('down')].T
        return dict(gate=gate,up=up,silu=silu,product=product,down=down)

    def forward(self,ids,labels,mask,retain=False,tp=1,fault="none",capture_parallel=False):
        if ids.ndim != 1 or not 1<=ids.numel()<=64 or labels.shape!=ids.shape or mask.shape!=ids.shape:
            raise ValueError('invalid sequence/target/mask shape')
        if ids.min()<0 or ids.max()>=27 or labels.min()<0 or labels.max()>=27 or not torch.isfinite(mask).all() or ((mask!=0)&(mask!=1)).any():
            raise ValueError('invalid token or supervision mask')
        if mask.sum()<=0: raise ValueError('no_supervision: no backward/update')
        if type(tp) is not int or tp not in (1,2): raise ValueError("TP must be 1 or 2")
        from experiments.tp_dp_reference import TensorParallelReference
        parallel=TensorParallelReference(self.p,tp,fault,capture_parallel) if tp != 1 or capture_parallel or fault != "none" else None
        embedding=self.p['embedding'][ids]
        if retain: embedding.retain_grad()
        x=torch.nn.functional.dropout(embedding,p=self.fixture['training_dropout'],training=self.training)
        layers=[]
        for n in range(2):
            key=lambda name:f'l{n}_{name}'
            gqa=self.attention(x,n,parallel)
            attention=nested_tensor(gqa['residual'])
            norm=rms(attention,self.p[key('ffn_gain')],self.fixture['epsilon'])
            ffn=self.feed_forward(norm,n,parallel)
            x=attention+ffn['down']
            layers.append(dict(input_norm=gqa['norm'],attention=attention,ffn_norm=norm,**ffn,residual=x))
        final=rms(x,self.p['final_gain'],self.fixture['epsilon'])
        head=self.p['embedding'] if self.tied else self.p['head']
        if parallel:
            logits,logprobs=parallel.vocab(final,head,labels)
            self.parallel_trace=parallel.events
        else:
            logits=final@head.T
            logprobs=torch.log_softmax(logits,dim=-1)
        token_loss=parallel.target_loss if parallel else -logprobs.gather(-1,labels[:,None]).squeeze(-1)
        loss=(token_loss*mask).sum()/mask.sum()
        return dict(embedding=embedding,layers=layers,final_norm=final,logits=logits,logprobs=logprobs,
                    token_loss=token_loss,loss=loss,mask=mask)

    def optimizer(self):
        return torch.optim.SGD(self.parameters(),lr=self.fixture['optimizer']['lr'],momentum=self.fixture['optimizer']['momentum'])


def step(model,optimizer,data):
    model.train();optimizer.zero_grad(set_to_none=True)
    trace=model(*data);trace['loss'].backward();optimizer.step()
    return float(trace['loss'].detach())


def save(model,optimizer,path):
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'tied':model.tied},path)


def restore(model,optimizer,path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['tied']!=model.tied: raise ValueError('tied-head configuration mismatch')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])


def plain_trace(trace):
    def plain(v):
        if isinstance(v,torch.Tensor): return v.detach().tolist()
        if isinstance(v,dict): return {k:plain(x) for k,x in v.items()}
        if isinstance(v,list): return [plain(x) for x in v]
        return v
    return plain(trace)


def export_evidence():
    torch.manual_seed(1201)
    model=TinyDecoder().eval();data=sample_data();trace=model(*data,retain=True);trace['loss'].backward()
    selected=[('embedding',(1,0)),('l0_qkv',(0,0)),('l0_gate_up',(0,0)),('l1_down',(0,0)),('final_gain',(0,))]
    params=[]
    for name,index in selected:
        value=float(model.p[name][index].detach());analytic=float(model.p[name].grad[index])
        h=1e-5
        with torch.no_grad():
            model.p[name][index]=value+h;plus=float(model(*data)['loss'])
            model.p[name][index]=value-h;minus=float(model(*data)['loss'])
            model.p[name][index]=value
        finite=(plus-minus)/(2*h)
        if abs(analytic-finite)>1e-6+1e-4*abs(finite): raise AssertionError('finite difference mismatch')
        params.append(dict(name=name,index=list(index),value=value,gradient=analytic,finite_difference=finite,step_size=h))
    prompt_gradient=trace['embedding'].grad[0].detach().tolist()
    before={k:v.detach().clone() for k,v in model.p.items()}
    optimizer=model.optimizer();optimizer.step()
    for p in params:p['after']=float(model.p[p['name']][tuple(p['index'])].detach())
    frozen_unchanged=all(torch.equal(before[k],model.p[k]) for k in model.fixture['frozen'])
    if not frozen_unchanged or not any(abs(x)>1e-12 for x in prompt_gradient):raise AssertionError('freeze/prompt gradient failed')
    # Real CPU save/restore, optimizer momentum and nontrivial dropout RNG included.
    torch.manual_seed(77);continuous=TinyDecoder();opt=continuous.optimizer();step(continuous,opt,data)
    checkpoint=io.BytesIO();save(continuous,opt,checkpoint)
    next_loss=step(continuous,opt,data)
    restored=TinyDecoder();opt2=restored.optimizer();checkpoint.seek(0);restore(restored,opt2,checkpoint)
    resumed_loss=step(restored,opt2,data)
    difference=max(float((v-restored.p[k]).abs().max().detach()) for k,v in continuous.p.items())
    if difference!=0 or next_loss!=resumed_loss:raise AssertionError('resume mismatch')
    paths=['experiments/decoder_reference.py','experiments/gqa_reference.py','experiments/sft_data_reference.py',
           'content/fixtures/decoder-reference.json','content/fixtures/sft-data.json']
    fingerprints={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    return dict(schema_version=1,provenance='reference_cpu_autograd',runtime_training='not_run',
                architecture_origin=model.fixture['provenance'],created_at=datetime.now(timezone.utc).isoformat(),
                python=sys.version.split()[0],torch=torch.__version__,device='cpu',dtype='float64',source_hashes=fingerprints,
                sample='arithmetic-multiturn',mode='assistant',mask_count=int(data[2].sum()),
                loss_before=float(trace['loss'].detach()),loss_after=float(model(*data)['loss'].detach()),
                selected_parameters=params,prompt_position=0,prompt_direct_loss_mask=float(data[2][0]),
                prompt_activation_gradient=prompt_gradient,frozen_unchanged=frozen_unchanged,
                resume={'continuous_next_loss':next_loss,'resumed_next_loss':resumed_loss,'max_parameter_error':difference,
                        'saved':['model','optimizer momentum','CPU RNG'],'dropout':model.fixture['training_dropout']},
                limitations=['authored two-layer CPU model, not pretrained SFT','not observed_bridge','dropout off for forward/finite difference; on for resume'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--write-fixture',action='store_true')
    parser.add_argument('--export')
    parser.add_argument('--forward',action='store_true')
    args=parser.parse_args()
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(authored_model(),indent=2)+'\n')
    elif args.forward:
        rows=[]
        for tied in (True,False):
            model=TinyDecoder(tied=tied).eval()
            for index,mode in [(0,'assistant'),(1,'full'),(2,'last_turn')]:
                data=sample_data(index,mode);rows.append(dict(tied=tied,sample=index,mode=mode,trace=plain_trace(model(*data))))
        print(json.dumps(rows,allow_nan=False))
    else:
        result=export_evidence()
        if args.export:
            path=ROOT/args.export;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'status':'exported','path':args.export,'scope':result['provenance'],'resume':result['resume']}))
        else:print(json.dumps(result,allow_nan=False))
