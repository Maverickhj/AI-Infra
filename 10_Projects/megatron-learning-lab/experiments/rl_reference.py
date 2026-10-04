"""GRPO/PPO authored-trajectory CPU reference, using the complete G03 decoder.

The two-layer backbone is frozen. A full vocabulary LM head and an independent
critic head are the trainable policy/value parameterizations. Actions are fixed
fixtures, never model-generated rollouts. No production trainer is replaced.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import struct
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from experiments.decoder_reference import TinyDecoder,FIXTURE as DECODER,plain_trace

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'content/fixtures/rl-reference.json'


def authored_fixture():
    base=json.loads(DECODER.read_text())['parameters']['embedding']
    current=[[v+(((i*7+j*3)%13)-6)*.065 for j,v in enumerate(row)] for i,row in enumerate(base)]
    reference=[[v*.85+((i+j)%3-1)*.025 for j,v in enumerate(row)] for i,row in enumerate(base)]
    trajectories=[
      dict(id='sum-correct',group='2+3',prompt=[1,2,6,7,8,9,5,3],response=[10,4],response_mask=[1,1],reward=1.),
      dict(id='sum-incorrect',group='2+3',prompt=[1,2,6,7,8,9,5,3],response=[16,11,4],response_mask=[1,1,1],reward=0.),
      dict(id='double-a',group='4+4',prompt=[1,2,15,16,7,16,17,5,3],response=[18,4],response_mask=[1,1],reward=.5),
      dict(id='double-b-with-gap',group='4+4',prompt=[1,2,15,16,7,16,17,5,3],response=[10,5,3,4],response_mask=[1,0,0,1],reward=.5),
    ]
    return dict(schema_version=1,provenance='authored_trajectories',actions_origin='fixed visible tokens, not generated',
      policy_backbone='complete two-layer G03 decoder; frozen; float64 CPU',trainable=['LM head [27,8]','independent PPO critic head [8]'],
      trajectories=trajectories,heads=dict(generation=base,previous=base,current=current,reference=reference),
      critic_old=[.03,-.06,.08,-.02,.04,-.03,.02,.07],
      critic_current=[.18,-.13,.19,-.14,.17,-.15,.08,.16],
      config=dict(clip=.2,kl_beta=.02,kl_type='k3',kl_input_clamp=20.,kl_output_clamp=10.,
                  use_on_policy_kl_approximation=False,gamma=.9,gae_lambda=.95,value_clip=.2,
                  actor_lr=.08,critic_lr=.05,grpo_epsilon=1e-6,normalize_ppo_advantages=False),
      versions=dict(generation=0,previous=0,current=1,reference='fixed-reference'),
      limitations=['candidate head version1 is authored, not claimed to result from an earlier optimizer run',
                   'no filters, temperature=1, no off-policy correction, dual clipping, CISPO, OPD or async',
                   'force-on-policy starts from version0 and permits one update per batch',
                   'terminal episodes; no truncation bootstrap; masked gaps carry GAE across valid response tokens'])


@lru_cache(maxsize=4)
def frozen_batch(fixture_json):
    f=json.loads(fixture_json);length=max(len(t['prompt'])+len(t['response']) for t in f['trajectories'])
    ids=[];masks=[];features=[];groups=[];rewards=[]
    model=TinyDecoder(tied=False).eval()
    for p in model.parameters():p.requires_grad_(False)
    with torch.no_grad():
      for item in f['trajectories']:
        tokens=item['prompt']+item['response']
        mask=[0]*len(item['prompt'])+item['response_mask']
        if len(item['response'])!=len(item['response_mask']) or not any(item['response_mask']):raise ValueError('missing response mask')
        ids_t=torch.tensor(tokens,dtype=torch.long)
        labels=torch.tensor(tokens[1:]+[0],dtype=torch.long)
        next_mask=torch.tensor(mask[1:]+[0],dtype=torch.float64)
        trace=model(ids_t,labels,next_mask)
        # At action token j, state is the prefix ending at j-1. Position0 is a
        # dummy alignment slot. Generated logprobs and masks use action indices.
        x=torch.cat([torch.zeros(1,8,dtype=torch.float64),trace['final_norm'][:-1]],dim=0)
        pad=length-len(tokens)
        features.append(torch.nn.functional.pad(x,(0,0,0,pad)))
        ids.append(tokens+[0]*pad);masks.append(mask+[0]*pad);groups.append(item['group']);rewards.append(item['reward'])
    return dict(ids=torch.tensor(ids),mask=torch.tensor(masks,dtype=torch.float64),features=torch.stack(features),
                groups=groups,rewards=torch.tensor(rewards,dtype=torch.float64))


def group_advantages(groups,rewards,epsilon=1e-6):
    baseline=torch.zeros_like(rewards);std=torch.zeros_like(rewards)
    for group in dict.fromkeys(groups):
        indices=[i for i,g in enumerate(groups) if g==group]
        row=rewards[indices]
        baseline[indices]=row.mean()
        std[indices]=row.std(correction=1) if len(indices)>1 else 0
    centered=rewards-baseline
    advantage=torch.where(std>0,centered/(std+epsilon),centered)
    return dict(baseline=baseline,std=std,sequence_advantages=advantage)


def gae(rewards,values,mask,gamma=.9,lam=.95):
    # An independent reverse recurrence; TS reference will sum discounted TD
    # errors directly over valid response positions instead of copying this loop.
    tokens=torch.zeros_like(values)
    for i in range(len(rewards)):
        valid=mask[i].nonzero().flatten()
        if not len(valid):raise ValueError('no response tokens')
        tokens[i,valid[-1]]=rewards[i]
    nxt=torch.zeros_like(rewards);carry=torch.zeros_like(rewards);columns=[];deltas=[]
    for t in reversed(range(values.shape[1])):
        delta=tokens[:,t]+gamma*nxt-values[:,t]
        new=delta+gamma*lam*carry;m=mask[:,t]
        nxt=values[:,t]*m+(1-m)*nxt;carry=new*m+(1-m)*carry
        columns.append(carry);deltas.append(delta*m)
    raw=torch.stack(columns[::-1],dim=1)
    return dict(token_rewards=tokens,delta=torch.stack(deltas[::-1],dim=1),
                advantages=raw*mask,returns=raw+values)


def reduce_rows(values,mask,reduction):
    if reduction not in ('token','sequence') or mask.sum()<=0 or (mask.sum(-1)<=0).any():raise ValueError('invalid reduction or response mask')
    if reduction=='token':return (values*mask).sum()/mask.sum()
    return ((values*mask).sum(-1)/mask.sum(-1)).mean()


def logprobs(features,head,ids):
    logits=features@head.T;all_probs=torch.softmax(logits,dim=-1)
    lp=torch.log_softmax(logits,dim=-1).gather(-1,ids[:,:,None]).squeeze(-1)
    # Dummy j=0 is not a prediction; preserve leading slot matching upstream
    # full-token fields, whose actor loss subsequently slices [:,1:].
    lp=torch.cat([torch.zeros_like(lp[:,:1]),lp[:,1:]],dim=1)
    return lp,all_probs


def policy_terms(curr,prev,generation,reference,advantages,mask,config,reduction='token',force=False,anchor=None,fault='none'):
    anchor=curr.detach() if anchor is None else anchor
    if force:
        ratio=torch.ones_like(curr) if fault=='constant_ratio' else (curr-anchor).exp()
    else:ratio=(curr-prev).exp()
    clipped=ratio if force else ratio.clamp(1-config['clip'],1+config['clip'])
    term=torch.maximum(-advantages*ratio,-advantages*clipped)
    if fault=='wrong_clip':term=torch.minimum(-advantages*ratio,-advantages*clipped)
    actor=reduce_rows(term,mask,reduction)
    # Fixed-source non-IS KL branch: forward weight1, score-function gradient.
    delta=reference-curr;limited=delta.clamp(-config['kl_input_clamp'],config['kl_input_clamp'])
    weight=(curr-anchor).exp()
    if fault=='detach_kl_weight':weight=weight.detach()
    weight=torch.where(delta==limited,weight,weight.detach())
    kl_token=(weight*(limited.exp()-1-limited)).clamp(-config['kl_output_clamp'],config['kl_output_clamp'])
    kl=reduce_rows(kl_token,mask,reduction)*config['kl_beta']
    return dict(ratio=ratio,clipped_ratio=clipped,pg_token=term,kl_token=kl_token,
                actor_loss=actor,kl_loss=kl,policy_loss=actor+kl)


def value_terms(values,old,returns,mask,config):
    clipped=values.clamp(old-config['value_clip'],old+config['value_clip'])
    raw=(values-returns).square();limited=(clipped-returns).square()
    token=.5*torch.maximum(raw,limited)
    return dict(values=values,old_values=old,returns=returns,clipped_values=clipped,value_token=token,
                value_loss=reduce_rows(token,mask,'token'))


def reference_run(algorithm='grpo',reduction='token',kl=True,force=False,equal_rewards=False,fault='none'):
    if algorithm not in ('grpo','ppo'):raise ValueError('unsupported RL algorithm')
    f=json.loads(FIXTURE.read_text());cfg=copy.deepcopy(f['config']);cfg['kl_beta']=cfg['kl_beta'] if kl else 0.
    data=frozen_batch(json.dumps(f,sort_keys=True));mask=data['mask'];x=data['features'];ids=data['ids']
    rewards=torch.full_like(data['rewards'],.5) if equal_rewards else data['rewards']
    previous=torch.tensor(f['heads']['previous'],dtype=torch.float64)
    generation=torch.tensor(f['heads']['generation'],dtype=torch.float64)
    reference=torch.tensor(f['heads']['reference'],dtype=torch.float64)
    head=nn.Parameter(previous.clone() if force else torch.tensor(f['heads']['current'],dtype=torch.float64))
    critic=nn.Parameter(torch.tensor(f['critic_current'],dtype=torch.float64))
    old_critic=torch.tensor(f['critic_old'],dtype=torch.float64)
    curr,probs=logprobs(x,head,ids);curr.retain_grad()
    prev,_=logprobs(x,previous,ids);gen,_=logprobs(x,generation,ids);ref,_=logprobs(x,reference,ids)
    grouped=group_advantages(data['groups'],rewards,cfg['grpo_epsilon'])
    old_values=x@old_critic
    gae_trace=gae(rewards,old_values,mask,cfg['gamma'],cfg['gae_lambda'])
    advantages=grouped['sequence_advantages'][:,None].expand_as(mask)*mask if algorithm=='grpo' else gae_trace['advantages'].detach()
    terms=policy_terms(curr,prev,gen,ref,advantages,mask,cfg,reduction,force,fault=fault)
    values=x@critic
    vt=value_terms(values,old_values,gae_trace['returns'].detach(),mask,cfg)
    # Two independent optimizer/backward paths. No actor graph through critic.
    terms['policy_loss'].backward()
    if critic.grad is not None:raise AssertionError('actor gradient reached critic')
    policy_grad=head.grad.detach().clone();lp_grad=curr.grad.detach().clone()
    if algorithm=='ppo':
        vt['value_loss'].backward()
        if not torch.equal(head.grad,policy_grad):raise AssertionError('critic gradient reached actor')
    critic_grad=critic.grad.detach().clone() if critic.grad is not None else torch.zeros_like(critic)
    actor_optimizer=torch.optim.SGD([head],lr=cfg['actor_lr']);actor_optimizer.step()
    if algorithm=='ppo':torch.optim.SGD([critic],lr=cfg['critic_lr']).step()
    after,_=logprobs(x,head,ids)
    return dict(algorithm=algorithm,critic_used=algorithm=='ppo',reduction=reduction,kl_enabled=kl,force_on_policy=force,equal_rewards=equal_rewards,
       ids=ids,mask=mask,features=x,rewards=rewards,group=grouped,gae=gae_trace,advantages=advantages,
       probabilities=probs,generation_logprobs=gen,previous_logprobs=prev,current_logprobs=curr,
       reference_logprobs=ref,terms=terms,value=vt,head_gradient=policy_grad,logprob_gradient=lp_grad,
       critic_gradient=critic_grad,head_after=head,critic_after=critic,logprobs_after=after,
       current_version=0 if force else 1,next_version=1 if force else 2,
       scope='authored reference; frozen complete decoder; trainable policy/critic heads; no rollout')


def state_hash(weights):
    if not isinstance(weights,list) or len(weights)!=27 or any(not isinstance(row,list) or len(row)!=8 for row in weights):
        raise ValueError('policy head must have shape [27,8]')
    flat=[v for row in weights for v in row]
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in flat):raise ValueError('nonfinite policy head')
    # Canonical cross-language bytes avoid JSON differences such as 0 vs 0.0.
    return hashlib.sha256(b'f64le:27x8\n'+b''.join(struct.pack('<d',v) for v in flat)).hexdigest()


def refit_contract(policy_version,weights,exported_version,exported_hash,generation_version,ack_hash,completed):
    digest=state_hash(weights)
    if type(policy_version) is not int or policy_version<0 or exported_version!=policy_version:
        raise ValueError('policy/export version mismatch')
    if exported_hash!=digest:raise ValueError('export weight hash mismatch')
    if not completed:raise ValueError('refit not completed')
    if generation_version!=policy_version or ack_hash!=digest:raise ValueError('generation version/refit acknowledgment mismatch')
    return dict(status='reference_contract_synchronized',version=policy_version,weights_sha256=digest,
                actual_rollout='not_run')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-fixture',action='store_true');parser.add_argument('--forward',action='store_true')
    args=parser.parse_args()
    if args.write_fixture:FIXTURE.write_text(json.dumps(authored_fixture(),indent=2)+'\n');print(str(FIXTURE))
    elif args.forward:
        result=[]
        for algorithm in ('grpo','ppo'):
            for reduction in ('token','sequence'):
                for kl,force in ((False,False),(True,False),(False,True),(True,True)):
                    result.append(plain_trace(reference_run(algorithm,reduction,kl,force)))
        print(json.dumps(result,allow_nan=False))
