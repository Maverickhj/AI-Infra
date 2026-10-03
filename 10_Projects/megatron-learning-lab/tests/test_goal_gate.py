"""只测试门槛工具，不把合成报告当作真实 GQA/浏览器结果。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tools.goal_gate import (ROOT, GOAL, digest, inside, snapshot, check_evidence,
                             browser_result, execute, verify_report)


def browser_fixture(**stats):
    return {'stats':{'expected':1,'unexpected':0,'flaky':0,'skipped':0,**stats},
            'errors':[], 'suites':[{'specs':[{'title':'[TEST] fixture only',
                       'tests':[{'status':'expected'}]}]}]}


class GoalGateTests(unittest.TestCase):
    def test_pending_scaffold_cannot_complete(self):
        contract = json.loads((ROOT/GOAL/'acceptance.json').read_text(encoding='utf-8'))
        # Copy the schema, not the live future evidence status.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/GOAL).mkdir(parents=True)
            (root/GOAL/'evidence.json').write_text(json.dumps({'goal_id':contract['goal_id'],
                'criteria':[{'id':c['id'],'status':'pending','summary':'','artifacts':[]} for c in contract['criteria']]}))
            self.assertTrue(check_evidence(root,contract))

    def test_browser_report_required(self):
        with self.assertRaises(ValueError):
            browser_result('all tests passed',[])

    def test_browser_json_with_npm_prefix(self):
        result=browser_result('> npm run test\n'+json.dumps(browser_fixture()),['[TEST]'])
        self.assertEqual(result['stats']['expected'],1)

    def test_browser_zero_skipped_flaky_or_failed_rejected(self):
        for field,value in [('expected',0),('unexpected',1),('skipped',1),('flaky',1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                browser_result(json.dumps(browser_fixture(**{field:value})),[])

    def test_missing_required_scenario_rejected(self):
        with self.assertRaises(ValueError):
            browser_result(json.dumps(browser_fixture()),['[G01-GQA]'])

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): inside(Path(d),'../secret')
            with self.assertRaises(ValueError): inside(Path(d),'/tmp/secret')

    def test_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            root=Path(d)
            try: (root/'link').symlink_to(Path(other),target_is_directory=True)
            except OSError: self.skipTest('this host cannot create symlinks')
            with self.assertRaises(ValueError): inside(root,'link/file')

    def test_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/GOAL).mkdir(parents=True); f=root/'result.txt'; f.write_text('fixture')
            c={'goal_id':'test','required_artifacts':['result.txt'],'criteria':[{'id':'a'}]}
            evidence={'goal_id':'test','criteria':[{'id':'a','status':'verified','summary':'fixture only',
                'artifacts':[{'path':'result.txt','sha256':digest(f)}]}]}
            (root/GOAL/'evidence.json').write_text(json.dumps(evidence))
            self.assertFalse(check_evidence(root,c))
            f.write_text('changed')
            self.assertTrue(check_evidence(root,c))

    def test_duplicate_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/GOAL).mkdir(parents=True)
            (root/GOAL/'evidence.json').write_text(json.dumps({'goal_id':'test','criteria':[{'id':'a'},{'id':'a'}]}))
            self.assertTrue(check_evidence(root,{'goal_id':'test','required_artifacts':[],'criteria':[{'id':'a'},{'id':'b'}]}))

    def test_source_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'web').mkdir(); (root/GOAL).mkdir(parents=True)
            for name in ('GOAL.md','acceptance.json'): (root/GOAL/name).write_text('fixture')
            c={'source_roots':['web'],'source_files':[]}
            (root/'web/main.ts').write_text('first')
            a=snapshot(root,c); (root/'web/main.ts').write_text('second')
            self.assertNotEqual(a,snapshot(root,c))

    def test_failed_process_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r=execute(root,[sys.executable,'-c','raise SystemExit(7)'],root/'log.txt',5)
            self.assertEqual(r['returncode'],7); self.assertEqual(r['status'],'failed')

    def test_timeout_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r=execute(root,[sys.executable,'-c','import time; time.sleep(10)'],root/'log.txt',0.1)
            self.assertEqual(r['status'],'blocked')

    def test_missing_program_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r=execute(root,['nonexistent-goal-test-program'],root/'log.txt',5)
            self.assertEqual(r['status'],'blocked')

    def test_baseline_is_not_goal_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); report=root/'report.json'
            report.write_text(json.dumps({'mode':'baseline','status':'baseline_passed'}))
            with self.assertRaises(ValueError): verify_report(root,report,{})


if __name__=='__main__':
    unittest.main()
