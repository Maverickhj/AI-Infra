"""Subprocess entry using installed CPU Torch; invoked by the stdlib test wrapper."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from experiments.decoder_reference import TinyDecoder,sample_data,step,save,restore,export_evidence


class DecoderCPUChecks(unittest.TestCase):
    def test_ce_head_analytic_gradient_and_prompt_path(self):
        model=TinyDecoder(tied=False).eval();ids,labels,mask=sample_data()
        t=model(ids,labels,mask,retain=True);t['loss'].backward()
        ce=torch.nn.functional.cross_entropy(t['logits'],labels,reduction='none')
        torch.testing.assert_close(ce,t['token_loss'],atol=1e-12,rtol=1e-12)
        manual=(t['logprobs'].exp()-torch.nn.functional.one_hot(labels,27))*mask[:,None]/mask.sum()
        torch.testing.assert_close(model.p['head'].grad,manual.T@t['final_norm'],atol=1e-12,rtol=1e-12)
        self.assertEqual(mask[0],0)
        self.assertGreater(float(t['embedding'].grad[0].abs().sum()),1e-8)
        wrong_mean=(ce*mask).mean()
        self.assertGreater(abs(float(wrong_mean.detach())-float(t['loss'].detach())),.1)

    def test_tied_gradient_is_embedding_plus_head_gradient(self):
        tied=TinyDecoder(tied=True).eval();untied=TinyDecoder(tied=False).eval();data=sample_data()
        tied(*data)['loss'].backward();untied(*data)['loss'].backward()
        torch.testing.assert_close(tied.p['embedding'].grad,untied.p['embedding'].grad+untied.p['head'].grad,atol=1e-11,rtol=1e-11)
        self.assertGreater(float((tied.p['embedding'].grad-untied.p['embedding'].grad).abs().max()),1e-5)

    def test_selected_autograd_matches_finite_difference_and_update(self):
        result=export_evidence()
        self.assertEqual(result['device'],'cpu');self.assertTrue(result['frozen_unchanged'])
        self.assertLess(result['loss_after'],result['loss_before'])
        for p in result['selected_parameters']:
            self.assertLessEqual(abs(p['gradient']-p['finite_difference']),1e-6+1e-4*abs(p['finite_difference']))
            self.assertAlmostEqual(p['after'],p['value']-.03*p['gradient'],places=12)
        self.assertTrue(any(abs(p['gradient'])>1e-5 for p in result['selected_parameters']))

    def test_file_resume_restores_optimizer_and_rng_and_wrong_rng_fails(self):
        data=sample_data();torch.manual_seed(392)
        model=TinyDecoder();optimizer=model.optimizer();step(model,optimizer,data)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt';save(model,optimizer,path)
            expected=step(model,optimizer,data)
            resumed=TinyDecoder();opt=resumed.optimizer();restore(resumed,opt,path)
            self.assertEqual(step(resumed,opt,data),expected)
            for key,value in model.p.items():torch.testing.assert_close(resumed.p[key],value,atol=0,rtol=0)
            wrong=TinyDecoder();opt_wrong=wrong.optimizer();restore(wrong,opt_wrong,path)
            torch.rand(11) # perturb restored RNG to prove this isn't a vacuous checkpoint check
            actual=step(wrong,opt_wrong,data)
            self.assertNotEqual(actual,expected)
            with self.assertRaisesRegex(ValueError,'tied-head'):restore(TinyDecoder(tied=False),opt_wrong,path)

    def test_frozen_parameters_have_no_grad_and_causal_inputs_do_not_leak(self):
        model=TinyDecoder().eval();ids,labels,mask=sample_data();trace=model(ids,labels,mask)
        trace['loss'].backward();self.assertIsNone(model.p['l0_input_gain'].grad)
        before=model.p['l0_input_gain'].detach().clone();model.optimizer().step()
        torch.testing.assert_close(model.p['l0_input_gain'],before,atol=0,rtol=0)
        original=model(ids,labels,mask)['logits'];changed=ids.clone();changed[10:]=(changed[10:]+1)%27
        after=model(changed,labels,mask)['logits']
        torch.testing.assert_close(original[:10],after[:10],atol=0,rtol=0)
        self.assertGreater(float((original[10:]-after[10:]).abs().max().detach()),1e-5)

    def test_zero_mask_and_invalid_ids_reject_training(self):
        model=TinyDecoder().eval();ids,labels,mask=sample_data()
        with self.assertRaisesRegex(ValueError,'no_supervision'):model(ids,labels,mask*0)
        bad=ids.clone();bad[0]=27
        with self.assertRaises(ValueError):model(bad,labels,mask)


    def test_rms_hand_example_and_swiglu_wrong_activation(self):
        from experiments.decoder_reference import rms
        x=torch.tensor([[3.,4.]],dtype=torch.float64)
        expected=torch.tensor([[.848528137423857,1.131370849898476]],dtype=torch.float64)
        torch.testing.assert_close(rms(x,torch.ones(2,dtype=torch.float64),0.),expected,atol=1e-14,rtol=1e-14)
        trace=TinyDecoder().eval()(*sample_data())
        for layer in trace['layers']:
            wrong=torch.sigmoid(layer['gate'])*layer['up']
            self.assertGreater(float((wrong-layer['product']).abs().max().detach()),1e-5)
            swapped=torch.nn.functional.silu(layer['up'])*layer['gate']
            self.assertGreater(float((swapped-layer['product']).abs().max().detach()),1e-5)

if __name__=='__main__':unittest.main(verbosity=2)
