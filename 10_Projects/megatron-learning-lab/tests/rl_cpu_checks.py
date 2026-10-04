import copy
import json
import math
import unittest
import torch
from experiments.rl_reference import FIXTURE,frozen_batch,group_advantages,gae,reduce_rows,policy_terms,value_terms,reference_run,refit_contract,state_hash,logprobs

class RLCPUChecks(unittest.TestCase):
    def setUp(self):
        self.fixture=json.loads(FIXTURE.read_text());self.config=copy.deepcopy(self.fixture['config'])
    def close(self,a,b):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_group_sample_std_equal_scores_and_singleton(self):
        r=group_advantages(['a','a','b','b','c'],torch.tensor([1.,0.,.5,.5,9.],dtype=torch.float64))
        self.close(r['baseline'],torch.tensor([.5,.5,.5,.5,9.],dtype=torch.float64))
        self.close(r['std'],torch.tensor([math.sqrt(.5),math.sqrt(.5),0.,0.,0.],dtype=torch.float64))
        self.assertAlmostEqual(float(r['sequence_advantages'][0]),.5/(math.sqrt(.5)+1e-6),places=13)
        self.assertEqual(r['sequence_advantages'][2:].abs().sum(),0)

    def test_gae_hand_case_and_masked_gap_carry(self):
        values=torch.tensor([[.2,.4]],dtype=torch.float64)
        r=gae(torch.tensor([1.],dtype=torch.float64),values,torch.ones_like(values),gamma=.9,lam=.8)
        # Last delta=.6; first delta=.9*.4-.2=.16; A0=.16+.72*.6=.592.
        self.close(r['advantages'],torch.tensor([[.592,.6]],dtype=torch.float64))
        self.close(r['returns'],torch.tensor([[.792,1.]],dtype=torch.float64))
        gap=gae(torch.tensor([1.],dtype=torch.float64),torch.tensor([[.2,999.,.4]],dtype=torch.float64),torch.tensor([[1.,0.,1.]],dtype=torch.float64),gamma=.9,lam=.8)
        self.close(gap['advantages'][:,[0,2]],r['advantages']);self.assertEqual(float(gap['advantages'][0,1]),0)
        self.close(gap['token_rewards'],torch.tensor([[0.,0.,1.]],dtype=torch.float64))
        self.assertNotEqual(float(gap['returns'][0,1]),0) # masked returns are ignored, not fabricated zeros.

    def test_positive_negative_clipping_and_reduction_hand_cases(self):
        cfg={**self.config,'kl_beta':0.}
        curr=torch.tensor([[math.log(1.5),math.log(.5),math.log(.5),math.log(1.5)]],dtype=torch.float64,requires_grad=True)
        zero=torch.zeros_like(curr);adv=torch.tensor([[2.,2.,-2.,-2.]],dtype=torch.float64);mask=torch.ones_like(curr)
        r=policy_terms(curr,zero,zero,zero,adv,mask,cfg);r['policy_loss'].backward()
        self.close(r['pg_token'],torch.tensor([[-2.4,-1.,1.6,3.]],dtype=torch.float64))
        self.close(curr.grad,torch.tensor([[0.,-.25,0.,.75]],dtype=torch.float64))
        v=torch.tensor([[1.,0.,0.],[3.,3.,3.]],dtype=torch.float64);m=torch.tensor([[1.,0.,0.],[1.,1.,1.]],dtype=torch.float64)
        self.assertEqual(float(reduce_rows(v,m,'token')),2.5);self.assertEqual(float(reduce_rows(v,m,'sequence')),2.)
        wrong=policy_terms(curr.detach(),zero,zero,zero,adv,mask,cfg,fault='wrong_clip')
        self.assertNotEqual(float(wrong['actor_loss']),float(r['actor_loss'].detach()))

    def test_force_ratio_retains_gradient_and_constant_counterexample(self):
        cfg={**self.config,'kl_beta':0.};x=torch.tensor([[-2.,-1.]],dtype=torch.float64,requires_grad=True)
        mask=torch.ones_like(x);adv=torch.tensor([[1.,-2.]],dtype=torch.float64)
        r=policy_terms(x,x.detach(),x.detach(),x.detach(),adv,mask,cfg,force=True);r['policy_loss'].backward()
        self.close(r['ratio'],torch.ones_like(x));self.close(x.grad,torch.tensor([[-.5,1.]],dtype=torch.float64))
        bad=x.detach().clone().requires_grad_();b=policy_terms(bad,bad.detach(),bad.detach(),bad.detach(),adv,mask,cfg,force=True,fault='constant_ratio');b['policy_loss'].backward()
        self.assertEqual(float(bad.grad.abs().sum()),0)

    def test_kl_sampling_weight_gradient_and_clamps(self):
        cfg={**self.config,'kl_beta':1.};x=torch.tensor([[-1.2,-2.]],dtype=torch.float64,requires_grad=True)
        ref=torch.tensor([[-1.5,-1.4]],dtype=torch.float64);mask=torch.ones_like(x);adv=torch.zeros_like(x)
        r=policy_terms(x,x.detach(),x.detach(),ref,adv,mask,cfg);r['policy_loss'].backward()
        self.close(x.grad,(x.detach()-ref)/2)
        bad=x.detach().clone().requires_grad_();r=policy_terms(bad,bad.detach(),bad.detach(),ref,adv,mask,cfg,fault='detach_kl_weight');r['policy_loss'].backward()
        self.assertGreater(float((bad.grad-x.grad).abs().max()),.01)
        extreme=torch.tensor([[-50.]],dtype=torch.float64,requires_grad=True);zero=torch.zeros_like(extreme)
        r=policy_terms(extreme,extreme.detach(),zero,zero,zero,torch.ones_like(zero),cfg);r['policy_loss'].backward()
        self.assertEqual(float(extreme.grad),0);self.assertEqual(float(r['kl_loss'].detach()),10.)

    def test_value_clipping_hand_gradient_and_actor_critic_independence(self):
        cfg={**self.config,'value_clip':.2}
        v=torch.tensor([[1.,.5]],dtype=torch.float64,requires_grad=True);old=torch.tensor([[0.,0.]],dtype=torch.float64);ret=torch.tensor([[0.,1.]],dtype=torch.float64)
        r=value_terms(v,old,ret,torch.ones_like(v),cfg);r['value_loss'].backward()
        self.close(r['value_token'],torch.tensor([[.5,.32]],dtype=torch.float64));self.close(v.grad,torch.tensor([[.5,0.]],dtype=torch.float64))
        grpo=reference_run('grpo');ppo=reference_run('ppo')
        self.assertGreater(float(ppo['critic_gradient'].abs().max()),0)
        self.assertEqual(float(grpo['critic_gradient'].abs().sum()),0)
        self.assertGreater(float(ppo['head_gradient'].abs().max()),0)
        self.assertGreater(float((ppo['head_after'].detach()-torch.tensor(self.fixture['heads']['current'])).abs().max()),0)
        self.assertGreater(float((ppo['critic_after'].detach()-torch.tensor(self.fixture['critic_current'])).abs().max()),0)

    def test_complete_policy_alignment_analytic_head_gradient_and_anchored_finite_difference(self):
        r=reference_run('ppo','sequence',True,True);data=frozen_batch(json.dumps(self.fixture,sort_keys=True))
        features=r['features'];ids=r['ids'];prob=r['probabilities'].detach()
        onehot=torch.nn.functional.one_hot(ids,27).to(torch.float64)
        hand=torch.einsum('bs,bsv,bsh->vh',r['logprob_gradient'],onehot-prob,features)
        self.close(hand,r['head_gradient'])
        self.assertEqual(float(r['logprob_gradient'][r['mask']==0].abs().sum()),0)
        self.close(r['generation_logprobs'],r['previous_logprobs'])
        self.close(r['current_logprobs'],r['previous_logprobs'])
        # Detach values are constants in this local surrogate. Re-freezing the
        # denominator at each finite-difference point would test another function.
        anchor=r['current_logprobs'].detach();head=torch.tensor(self.fixture['heads']['previous'],dtype=torch.float64)
        def loss(w):
            lp,_=logprobs(features,w,ids)
            return policy_terms(lp,r['previous_logprobs'],r['generation_logprobs'],r['reference_logprobs'],r['advantages'],r['mask'],self.config,'sequence',True,anchor=anchor)['policy_loss']
        row,col=10,0;step=1e-5
        head[row,col]+=step;plus=loss(head)
        head[row,col]-=2*step;minus=loss(head)
        self.assertAlmostEqual(float((plus-minus)/(2*step)),float(r['head_gradient'][row,col]),delta=1e-6)
        self.assertTrue(all(item['prompt'][0]==1 for item in self.fixture['trajectories']))

    def test_reference_refit_rejects_wrong_versions_hash_and_missing_ack(self):
        weights=self.fixture['heads']['current'];digest=state_hash(weights)
        good=refit_contract(1,weights,1,digest,1,digest,True)
        self.assertEqual(good['actual_rollout'],'not_run')
        for args in [(1,weights,0,digest,1,digest,True),(1,weights,1,'bad',1,digest,True),(1,weights,1,digest,0,digest,True),(1,weights,1,digest,1,'bad',True),(1,weights,1,digest,1,digest,False)]:
            with self.assertRaises(ValueError):refit_contract(*args)

if __name__=='__main__':unittest.main(verbosity=2)
