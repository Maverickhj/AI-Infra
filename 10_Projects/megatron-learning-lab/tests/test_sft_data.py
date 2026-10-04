import copy
import json
import math
import unittest
from experiments.sft_data_reference import FIXTURE, calculate, encode


class SFTDataTests(unittest.TestCase):
    def setUp(self):
        self.f = json.loads(FIXTURE.read_text())

    def test_messages_preserved_and_authored_boundary(self):
        raw = [json.loads(s) for s in (FIXTURE.parents[1]/'samples.jsonl').read_text().splitlines()]
        for a,b in zip(self.f['samples'],raw):
            self.assertEqual(a['id'],b['id'])
            self.assertEqual([{k:m[k] for k in ('role','content')} for m in a['messages']],b['messages'])
        self.assertEqual(self.f['provenance'],'authored token fixture')

    def test_hand_shift_mask_and_end(self):
        r = calculate(self.f,[0])
        # assistant role marker predicts 5, then '.', then EOS. Marker itself is not supervised.
        self.assertEqual(r['ids'][:8],[1,2,6,7,8,9,5,3])
        self.assertEqual(r['targets'][7:10],[10,11,4])
        self.assertEqual(r['masks'][6:11],[0,1,1,1,0])
        self.assertEqual(r['targets'][-1],-100)
        self.assertEqual(r['count'],6)
        self.assertEqual(calculate(self.f,[0],mode='last_turn')['count'],3)
        full=calculate(self.f,[0],mode='full')
        self.assertEqual(full['count'],len(full['ids'])-1)

    def test_pack_loss_targets_and_unpacked_agree(self):
        for mode in ('assistant','last_turn','full'):
            for padding in (1,8):
                packed=calculate(self.f,[1,2],mode=mode,padding=padding)
                singles=[calculate(self.f,[i],mode=mode) for i in [1,2]]
                self.assertEqual(packed['count'],sum(s['count'] for s in singles))
                self.assertAlmostEqual(packed['sum'],sum(s['sum'] for s in singles),places=12)
                self.assertEqual([t for t,m in zip(packed['targets'],packed['masks']) if m],
                                 [t for s in singles for t,m in zip(s['targets'],s['masks']) if m])
                for i,row in enumerate(packed['attention']):
                    for j,allow in enumerate(row):
                        if allow: self.assertEqual(packed['segments'][i],packed['segments'][j])
                for i in packed['cuSeqlensPadded'][1:]:
                    self.assertEqual(packed['masks'][i-1],0)

    def test_counterexample_cross_attention_changes_loss(self):
        good=calculate(self.f,[1,2])
        bad=calculate(self.f,[1,2],leak=True)
        self.assertGreater(abs(good['sum']-bad['sum']),1e-4)

    def test_truncation_zero_supervision_and_empty_response_eos(self):
        cut=calculate(self.f,[0],mode='last_turn',limit=12)
        self.assertEqual(cut['count'],0); self.assertIsNone(cut['mean'])
        self.assertNotEqual(cut['ids'][-1],4) # no fabricated EOS
        f=copy.deepcopy(self.f)
        f['samples'][1]['messages'][-1].update(content='',pieces=[])
        r=calculate(f,[1])
        self.assertEqual(r['count'],1) # explicitly authored EOS policy; not official template
        self.assertEqual([t for t,m in zip(r['targets'],r['masks']) if m],[4])

    def test_hand_loss_and_global_normalization(self):
        self.assertAlmostEqual((-math.log(.5)-math.log(.25))/2,1.0397207708399179)
        a,b=calculate(self.f,[1]),calculate(self.f,[2])
        global_mean=(a['sum']+b['sum'])/(a['count']+b['count'])
        self.assertNotAlmostEqual(global_mean,(a['mean']+b['mean'])/2,places=5)

    def test_invalid_input_rejected(self):
        for kw in ({'padding':0},{'limit':1},{'mode':'unknown'}):
            with self.assertRaises(ValueError): calculate(self.f,[0],**kw)
        with self.assertRaises(ValueError): calculate(self.f,[0,0])
