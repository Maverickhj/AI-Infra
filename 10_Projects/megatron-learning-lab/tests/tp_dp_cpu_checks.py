import unittest
import math
import torch
from experiments.decoder_reference import TinyDecoder,sample_data
from experiments.tp_dp_reference import compare_tp,dp_reference,TensorParallelReference

class TPDPChecks(unittest.TestCase):
    def test_tp1_tp2_complete_model_outputs_and_all_parameter_gradients(self):
        for tp in (1,2):
            for sample in (0,1,2):
                for tied in (True,False):
                    with self.subTest(tp=tp,sample=sample,tied=tied):
                        r=compare_tp(tp,sample,tied)
                        self.assertLess(r['max_gradient_error'],1e-10)
                        self.assertLess(r['max_output_error'],1e-10)
                        self.assertEqual(len(r['rank_events']),9)

    def test_weight_shards_restore_gqa_ffn_row_and_vocab(self):
        # Independent hand cases: row-SUM and owner-only vocabulary CE.
        x=torch.tensor([2.,3.],requires_grad=True);w=torch.tensor([4.,5.],requires_grad=True)
        y=sum(x[r]*w[r] for r in range(2));self.assertEqual(float(y.detach()),23.)
        y.backward();torch.testing.assert_close(x.grad,torch.tensor([4.,5.]));torch.testing.assert_close(w.grad,torch.tensor([2.,3.]))
        head=torch.zeros((2,1),dtype=torch.float64,requires_grad=True)
        ref=TensorParallelReference({},tp=2)
        ref.vocab(torch.ones((1,1),dtype=torch.float64),head,torch.tensor([1]))
        self.assertAlmostEqual(float(ref.target_loss.detach()),math.log(2),places=14)
        ref.target_loss.sum().backward();torch.testing.assert_close(head.grad,torch.tensor([[.5],[-.5]],dtype=torch.float64))
        m=TinyDecoder();p=m.p
        for layer in (0,1):
            qkv=p[f'l{layer}_qkv']
            torch.testing.assert_close(torch.cat([qkv[:16],qkv[16:]]),qkv,atol=0,rtol=0)
            w=p[f'l{layer}_gate_up']
            shards=[torch.cat([w[r*6:(r+1)*6],w[12+r*6:12+(r+1)*6]]) for r in range(2)]
            restored=torch.cat([torch.cat([s[:6] for s in shards]),torch.cat([s[6:] for s in shards])])
            torch.testing.assert_close(restored,w,atol=0,rtol=0)
            self.assertFalse(torch.equal(torch.cat(shards),w))
            out=p[f'l{layer}_out']
            torch.testing.assert_close(torch.cat([out[:,:8],out[:,8:]],dim=1),out,atol=0,rtol=0)
        padded=torch.nn.functional.pad(p['embedding'],(0,0,0,1))
        self.assertEqual(tuple(padded.shape),(28,8))
        torch.testing.assert_close(torch.cat([padded[:14],padded[14:]])[:27],p['embedding'],atol=0,rtol=0)

    def test_faults_change_complete_loss_and_invalid_concat_is_rejected(self):
        m=TinyDecoder().eval();data=sample_data()
        good=m(*data,tp=2)['loss']
        for fault in ('omit_reduce','wrong_gate_pair','include_padded_vocab'):
            bad=m(*data,tp=2,fault=fault)['loss']
            self.assertGreater(float((good-bad).abs().detach()),1e-5,fault)
        with self.assertRaisesRegex(ValueError,'SUM'):m(*data,tp=2,fault='concat_row')

    def test_dp_accumulation_uses_global_valid_tokens_and_rejects_means_of_means(self):
        self.assertEqual((1+9)/(1+3),2.5)
        self.assertEqual((1/1+9/3)/2,2.)
        good=dp_reference();bad=dp_reference(True)
        self.assertEqual([r['microbatches'] for r in good['ranks']],[2,2])
        self.assertEqual([r['valid_tokens'] for r in good['ranks']],[4,18])
        self.assertEqual(good['global_count'],22)
        self.assertLess(good['max_gradient_error'],1e-10)
        self.assertGreater(bad['max_gradient_error'],1e-5)
        self.assertGreater(abs(good['global_loss']-good['mean_of_means']),1e-5)

    def test_rank_shapes_and_theoretical_ring_bytes_are_explicit(self):
        r=compare_tp(2,sample=1)
        self.assertEqual(r['communication']['payload_bytes'],11*8*8)
        self.assertEqual(r['communication']['ring_allreduce_sent_bytes'],704)
        self.assertIsNone(r['communication']['measured_time'])
        self.assertEqual(r['rank_events'][0]['ranks'][1]['weight_rows'],[16,32])
        row=r['rank_events'][1]
        torch.testing.assert_close(torch.tensor(row['global_token7']),torch.tensor(row['ranks'][0]['token7'])+torch.tensor(row['ranks'][1]['token7']))
        self.assertEqual(row['ranks'][1]['input_shape'],[11,8])
        self.assertEqual(r['rank_events'][3]['ranks'][1]['weight_shape'],[8,6])
        self.assertEqual(len(row['ranks'][1]['gradient_preview']),2)
        self.assertEqual(r['rank_events'][-1]['global_shape'],[11,27])
        self.assertEqual(len(r['rank_events'][-1]['global_token7']),27)
        vocab=r['rank_events'][-1]['ranks']
        self.assertEqual(vocab[0]['vocab_range'],[0,14]);self.assertEqual(vocab[1]['valid_range'],[14,27])
        self.assertEqual(vocab[1]['token7'][-1],float('-inf'))

    def test_illegal_dimensions_and_types_rejected(self):
        for tp in (0,3,4,1.5,True):
            with self.assertRaisesRegex(ValueError,'TP'):TensorParallelReference(TinyDecoder().p,tp=tp)

if __name__=='__main__':unittest.main(verbosity=2)
