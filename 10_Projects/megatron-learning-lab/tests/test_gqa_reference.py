"""Mathematical invariants of the independent CPU teaching reference."""
import copy
import json
import math
import unittest
from experiments.gqa_reference import FIXTURE, compute, rotate, softmax, validate_selection


class GqaReferenceTests(unittest.TestCase):
    def setUp(self):
        self.f = json.loads(FIXTURE.read_text())

    def test_causal_future_mutation(self):
        for variant in ('qwen3', 'qwen25'):
            before = compute(self.f, variant)
            for boundary in range(3):
                changed = copy.deepcopy(self.f)
                for row in changed['x'][boundary+1:]:
                    row[:] = [x+7 for x in row]
                after = compute(changed, variant)
                self.assertEqual(before['residual'][:boundary+1], after['residual'][:boundary+1])
                self.assertNotEqual(before['residual'][-1], after['residual'][-1])

    def test_group_mapping_softmax_and_dimensions(self):
        r = compute(self.f)
        for t in range(4):
            for h in range(4):
                g = validate_selection(h,t)
                self.assertEqual(g, [0,0,1,1][h])
                p = r['probabilities'][t][h]
                self.assertAlmostEqual(sum(p), 1, places=14)
                for j in range(4):
                    self.assertEqual(r['masked'][t][h][j] == -math.inf, j > t)
                    if j > t: self.assertEqual(p[j],0)
                    self.assertAlmostEqual(r['scores'][t][h][j], sum(a*b for a,b in zip(r['qrope'][t][h],r['krope'][j][g])), places=14)
                for i in range(4):
                    self.assertAlmostEqual(r['heads'][t][h][i], sum(p[j]*r['v'][j][g][i] for j in range(4)), places=14)
            self.assertEqual(len(r['merged'][t]), 16)
            self.assertEqual(len(r['projected'][t]), 8)
            self.assertEqual(r['residual'][t], [a+b for a,b in zip(self.f['x'][t],r['projected'][t])])

    def test_counterexamples_are_detectable(self):
        correct = compute(self.f)
        for fault in ('omit_scale','wrong_group'):
            wrong = compute(self.f, fault=fault)
            delta = max(abs(a-b) for x,y in zip(correct['residual'],wrong['residual']) for a,b in zip(x,y))
            self.assertGreater(delta, 1e-4)
        q25 = compute(self.f, 'qwen25')
        self.assertEqual(q25['q'], q25['qnorm'])
        self.assertNotEqual(correct['q'],correct['qnorm'])
        for t in range(4):
            for j in range(32):
                self.assertAlmostEqual(q25['mixed'][t][j]-correct['mixed'][t][j],self.f['qkv_bias'][j],places=14)

    def test_rope_split_half_matches_analytic_rotation(self):
        self.assertEqual(rotate([1,2,3,4],0,10000),[1,2,3,4])
        y = rotate([1,2,3,4],1,10000)
        self.assertAlmostEqual(y[0],math.cos(1)-3*math.sin(1))
        self.assertAlmostEqual(y[1],2*math.cos(.01)-4*math.sin(.01))
        self.assertAlmostEqual(sum(v*v for v in y),30,places=13)

    def test_invalid_indices_inputs_and_all_mask(self):
        for value in (-1,4,1.5,float('nan'),float('inf'),'',None,True):
            for args in ((value,0),(0,value)):
                with self.assertRaises(ValueError): validate_selection(*args)
        for value in (float('nan'),float('inf'),'',None):
            bad = copy.deepcopy(self.f); bad['x'][0][0] = value
            with self.assertRaises(ValueError): compute(bad)
        bad=copy.deepcopy(self.f); bad['x']=[]
        with self.assertRaises(ValueError): compute(bad)
        for row in ([],[-math.inf]*4,[math.nan,0],[math.inf,0]):
            with self.assertRaises(ValueError): softmax(row)
        self.assertEqual(softmax([10000,10000,-math.inf]),[.5,.5,0])
