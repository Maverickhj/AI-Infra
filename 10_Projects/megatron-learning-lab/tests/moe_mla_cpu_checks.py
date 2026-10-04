import copy
import math
import unittest
import torch
from experiments.decoder_reference import rms
from experiments.moe_mla_reference import FamilyDecoder,FAMILIES,family_data,route,moe,mla,expert_groups,validate_dispatch

class MoEMLAChecks(unittest.TestCase):
    def assertClose(self,a,b):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_hand_router_normalization_scaling_and_bias_update(self):
        q=FamilyDecoder('qwen3-moe').config
        logits=torch.tensor([[math.log(v) for v in [1,2,3,4]]],dtype=torch.float64)
        r=route(logits,q,torch.tensor([False]))
        self.assertEqual(r['indices'].tolist(),[[3,2]])
        self.assertClose(r['weights'],torch.tensor([[4/7,3/7]],dtype=torch.float64))
        v2=FamilyDecoder('deepseek-v2-lite').config
        self.assertClose(route(logits,v2,torch.tensor([False]))['weights'],torch.tensor([[.4,.3]],dtype=torch.float64))
        v3=copy.deepcopy(FamilyDecoder().config);v3['bias']=[.1,0.,-.1,-.2]
        r=route(torch.zeros(2,4,dtype=torch.float64),v3,torch.tensor([False,True]))
        self.assertEqual(r['indices'][0].tolist(),[0,1])
        self.assertClose(r['weights'],torch.tensor([[1.25,1.25],[0.,0.]],dtype=torch.float64))
        self.assertEqual(r['counts'].tolist(),[1,1,0,0])
        self.assertClose(r['updated_bias'],torch.tensor([.09,-.01,-.09,-.19],dtype=torch.float64))
        self.assertAlmostEqual(float(r['aux_loss']),v3['aux_coeff'],places=13)
        self.assertFalse(r['updated_bias'].requires_grad)

    def test_dispatch_restores_each_token_against_dense_expert_oracle_and_faults(self):
        for family in FAMILIES:
            m=FamilyDecoder(family).eval();data=family_data(1,2);trace=m(*data)
            l=m.config['moe_layers'][-1];block=trace['layers'][l];x=block['ffn_norm'];p=m.layer_params(l)
            expected=torch.zeros_like(x)
            for t in range(len(x)-2):
                for slot,e in enumerate(block['moe']['indices'][t].tolist()):
                    gate,up=(x[t]@p[f'e{e}_gate_up'].T).chunk(2)
                    output=(gate/(1+torch.exp(-gate))*up)@p[f'e{e}_down'].T
                    expected[t]=expected[t]+block['moe']['weights'][t,slot]*output
            self.assertClose(expected,block['moe']['routed'])
            self.assertClose(block['down'],expected+block['moe']['shared'])
            self.assertEqual(len(block['moe']['dispatch']),2*(len(x)-2))
            self.assertAlmostEqual(.75*2+.25*6,3.)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                moe(x,p,m.config,data[3],fault='duplicate_dispatch')
            wrong=moe(x,p,m.config,data[3],fault='wrong_weights')
            self.assertGreater(float((wrong['down']-block['down']).abs().max().detach()),1e-5)
            if m.config['shared_experts']:
                missing=moe(x,p,m.config,data[3],fault='missing_shared')
                self.assertGreater(float((missing['down']-block['down']).abs().max().detach()),1e-4)

    def test_aux_ignores_bias_groups_and_excludes_padding_with_analytic_gradient(self):
        cfg=copy.deepcopy(FamilyDecoder().config)
        cfg['bias']=[0.,0.,10.,10.]
        z=torch.tensor([[3.,2.,1.,0.],[2.,1.,0.,-1.],[-4.,8.,-2.,1.]],dtype=torch.float64,requires_grad=True)
        pad=torch.tensor([False,False,True]);r=route(z,cfg,pad)
        self.assertEqual(r['counts'].tolist(),[0,0,2,2])
        self.assertEqual(r['aux_counts'].tolist(),[2.,2.,0.,0.])
        r['aux_loss'].backward()
        a=torch.sigmoid(z.detach()[:2]);s=a.sum(-1,keepdim=True);c=r['aux_counts']
        # d sum_e c_e * sigmoid(z_e)/sum(sigmoid(z)) / dz_j.
        hand=cfg['aux_coeff']*4/(2*2**2)*a*(1-a)*(c*s-(a*c).sum(-1,keepdim=True))/s.square()
        self.assertClose(z.grad[:2],hand);self.assertEqual(z.grad[2].abs().sum(),0)
        r2=route(z.detach()[:2],cfg,torch.tensor([False,False]))
        self.assertClose(r['aux_loss'],r2['aux_loss'])
        self.assertClose(r['updated_bias'],r2['updated_bias'])

    def test_padding_is_not_loss_mask_and_does_not_change_valid_model_outputs(self):
        for family in FAMILIES:
            m=FamilyDecoder(family).eval();a=m(*family_data(1));b=m(*family_data(1,2))
            self.assertClose(a['logits'],b['logits'][:-2]);self.assertClose(a['loss'],b['loss'])
            self.assertClose(a['aux_loss'],b['aux_loss'])
            for l in m.config['moe_layers']:
                r=b['layers'][l]['moe']
                self.assertEqual(int(r['valid_tokens']),11)
                self.assertEqual(int(r['counts'].sum()),22)
                self.assertEqual(float(r['combined'][-2:].abs().sum().detach()),0)
                self.assertIn(0,[row['token'] for row in r['dispatch']])
            bad=list(family_data(1,2));bad[2][-1]=1
            with self.assertRaisesRegex(ValueError,'padded supervision'):m(*bad)

    def test_mla_expanded_absorbed_incremental_cache_and_wrong_conditions(self):
        for family in FAMILIES[1:]:
            model=FamilyDecoder(family).eval();trace=model(*family_data(1))
            for l,a in enumerate(trace['mla']):
                self.assertClose(a['context'],a['absorbed']);self.assertClose(a['context'],a['cached_context'])
                self.assertEqual(list(a['cache'].shape),[11,5])
                self.assertClose(a['cache'][:,:3],a['kv_latent'])
                self.assertClose(a['cache'][:,3:],a['k_rope'])
                self.assertEqual(a['q_raw'].shape[-1],4 if family=='deepseek-v3' else 8)
                for fault in ('wrong_norm','wrong_scale','missing_rope'):
                    bad=mla(trace['layers'][l]['input_norm'],model.layer_params(l),model.config['q_rank'],fault=fault)
                    self.assertGreater(float(bad['max_absorption_error'].detach()),1e-5,fault)
            c=torch.tensor([1.,2.,3.]);wk=torch.tensor([[1.,0.,0.],[0.,1.,1.]]);q=torch.tensor([2.,3.])
            self.assertEqual(float(q@(wk@c)),17.);self.assertEqual(float((q@wk)@c),17.)
            self.assertEqual(3/(1+3)*4+1/(1+3)*2,3.5)

    def test_expert_parallel_groups_and_all_parameter_gradients_match(self):
        for family in FAMILIES:
            base=FamilyDecoder(family).eval();a=base(*family_data(1));a['total_loss'].backward()
            for ep,etp in [(1,1),(1,2),(2,1),(2,2)]:
                candidate=FamilyDecoder(family,ep,etp).eval();b=candidate(*family_data(1));b['total_loss'].backward()
                self.assertClose(a['logits'],b['logits']);self.assertClose(a['total_loss'],b['total_loss'])
                for name,p in base.p.items():
                    if p.grad is None:self.assertIsNone(candidate.p[name].grad)
                    else:self.assertClose(p.grad,candidate.p[name].grad)
                ranks=expert_groups(ep,etp)
                self.assertEqual(len(ranks),ep*etp)
                for r in ranks:
                    self.assertIn(r['rank'],r['ep']);self.assertIn(r['rank'],r['etp']);self.assertEqual(r['edp'],[r['rank']])
            for args in [(3,1,1),(1,3,1),(2,2,2)]:
                with self.assertRaisesRegex(ValueError,'verified'):expert_groups(*args)

    def test_full_model_router_and_mla_autograd_finite_difference_update(self):
        for family,name in [('qwen3-moe','l1_router'),('deepseek-v2-lite','l1_kv_down'),('deepseek-v3','l1_q_down')]:
            model=FamilyDecoder(family).eval();data=family_data(1);trace=model(*data);trace['total_loss'].backward()
            p=model.p[name];g=float(p.grad[0,0]);value=float(p[0,0].detach());h=1e-5
            with torch.no_grad():
                p[0,0]=value+h;plus=float(model(*data)['total_loss'])
                p[0,0]=value-h;minus=float(model(*data)['total_loss'])
                p[0,0]=value
            self.assertAlmostEqual(g,(plus-minus)/(2*h),delta=1e-6+1e-4*abs(g))
            self.assertGreater(abs(g),1e-10)
            optimizer=model.optimizer();optimizer.step()
            self.assertNotEqual(float(p[0,0].detach()),value)
            self.assertIsNone(model.p['l0_input_gain'].grad)
            self.assertFalse(any('bias' in k for k in model.p))

    def test_invalid_router_and_family_contracts_reject(self):
        m=FamilyDecoder()
        with self.assertRaisesRegex(ValueError,'no valid'):route(torch.zeros(2,4),m.config,torch.tensor([True,True]))
        with self.assertRaisesRegex(ValueError,'unsupported family'):FamilyDecoder('dsr1-distill-qwen15b')
        with self.assertRaisesRegex(ValueError,'dense TP'):m(*family_data(1),tp=2)
        data=list(family_data(1));data[3][0]=True;data[2][0]=0
        with self.assertRaisesRegex(ValueError,'suffix'):m(*data)

if __name__=='__main__':unittest.main(verbosity=2)
