import json
import math
import unittest
from experiments.sequence_reference import pipeline,partition,cp_reference

class SequenceReferenceTests(unittest.TestCase):
    def check_schedule(self,result,pp,m):
        events=result['events']
        self.assertEqual(len(events),2*pp*m)
        keys={(e['stage'],e['kind'],e['microbatch']):e for e in events}
        self.assertEqual(len(keys),2*pp*m)
        for s in range(pp):
            local=sorted([e for e in events if e['stage']==s],key=lambda e:e['start'])
            self.assertTrue(all(a['end']<=b['start'] for a,b in zip(local,local[1:])))
            for n in range(m):
                f,b=keys[(s,'F',n)],keys[(s,'B',n)]
                self.assertLessEqual(f['end'],b['start'])
                if s:self.assertLessEqual(keys[(s-1,'F',n)]['end'],f['start'])
                if s+1<pp:self.assertLessEqual(keys[(s+1,'B',n)]['end'],b['start'])
        self.assertEqual(sorted({layer for e in events for layer in e['layers']}),[0,1])
        self.assertTrue(all(l['start']<l['end']<=result['duration'] for l in result['lifetimes']))

    def test_all_schedules_and_mutation_rejection(self):
        for pp in (1,2):
            for m in range(1,9):self.check_schedule(pipeline(pp,m),pp,m)
        bad=pipeline();bad['events'].pop()
        with self.assertRaises(AssertionError):self.check_schedule(bad,2,4)
        bad=pipeline();next(e for e in bad['events'] if e['kind']=='B' and e['stage']==0)['start']=0
        with self.assertRaises(AssertionError):self.check_schedule(bad,2,4)

    def test_hand_schedule_and_activation_lifetimes(self):
        r=pipeline(2,2)
        self.assertEqual([(e['stage'],e['kind'],e['microbatch'],e['start']) for e in r['events']],[(0,'F',0,0),(0,'F',1,1),(1,'F',0,1),(1,'B',0,2),(0,'B',0,3),(1,'F',1,3),(1,'B',1,4),(0,'B',1,5)])
        self.assertEqual(r['duration'],6);self.assertEqual(r['peak_activations'],[2,1])
        self.assertEqual(r['idle_slots'],4)
        self.assertEqual(pipeline(1,2)['duration'],8)

    def test_zigzag_is_per_document_and_preserves_metadata(self):
        self.assertEqual(partition([0,8],2),[[0,1,6,7],[2,3,4,5]])
        self.assertEqual(partition([0,4,8],2),[[0,3,4,7],[1,2,5,6]])
        r=cp_reference('thd',8,2)
        self.assertEqual(r['cuSeqlens'],[0,11,34])
        self.assertEqual(r['cuSeqlensPadded'],[0,16,40])
        self.assertEqual(sorted(sum(r['shards'],[])),list(range(40)))
        self.assertEqual(r['positions'][16],0)
        self.assertEqual(sum(r['valid']),34)
        self.assertEqual(r['outputs'][11],[[0.]*4]*4)

    def test_full_context_and_packing_counterexamples(self):
        for layout in ('ordinary','thd'):
            a=cp_reference(layout,8,1);b=cp_reference(layout,8,2)
            self.assertEqual(a['outputs'],b['outputs'])
            bad=cp_reference(layout,8,2,'local_kv')
            self.assertGreater(max(abs(x-y) for t,u in zip(a['outputs'],bad['outputs']) for h,k in zip(t,u) for x,y in zip(h,k)),1e-4)
        leak=cp_reference('thd',8,2,'leak')
        good=cp_reference('thd',8,2)
        self.assertEqual(leak['outputs'][:16],good['outputs'][:16])
        self.assertNotEqual(leak['outputs'][16],good['outputs'][16])

    def test_invalid_parallel_and_padding_rejected(self):
        for pp,m in [(0,2),(3,2),(True,2),(2,0),(2,9),(2,1.5)]:
            with self.assertRaises(ValueError):pipeline(pp,m)
        for cp in (0,3,True):
            with self.assertRaises(ValueError):partition([0,8],cp)
        for boundaries in ([0,3],[0,4,7],[1,8],[0,4,4]):
            with self.assertRaises(ValueError):partition(boundaries,2)
        with self.assertRaisesRegex(ValueError,'2\\*CP'):cp_reference('thd',1,2)

if __name__=='__main__':unittest.main()
