"""计划工具的合成 fixture 测试，不证明任何模型/浏览器/训练阶段已通过。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.program_plan import DIRECTORY, ROOT, local, read, select_next, sha, validate


def fixture():
    stages = [
        {'id': 'A', 'track': 'software', 'depends_on': [], 'requires_grant': None},
        {'id': 'B', 'track': 'software', 'depends_on': ['A'], 'requires_grant': None},
        {'id': 'R', 'track': 'runtime', 'depends_on': ['A'], 'requires_grant': 'gpu'},
        {'id': 'F', 'track': 'software', 'depends_on': ['A', 'B'], 'requires_grant': None},
    ]
    for row in stages:
        row.update(title=row['id'], required_checks=['test'])
    plan = {'program_id': 'test', 'stages': stages}
    state = {'program_id': 'test', 'stages': {s['id']: {'status': 'pending', 'report': None, 'report_sha256': None} for s in stages}}
    return plan, state


def receipt(root, state, key):
    log = root / f'{key}.log'
    log.write_text('synthetic fixture only, not an executed test\n')
    report = {'program_id': 'test', 'stage_id': key, 'status': 'passed', 'source_fingerprint': 'a' * 64,
              'checks': [{'id': 'test', 'status': 'passed', 'returncode': 0, 'command': ['fixture-only'],
                          'log': log.name, 'log_sha256': sha(log)}],
              'artifacts': [{'path': log.name, 'sha256': sha(log)}]}
    path = root / f'{key}.json'
    path.write_text(json.dumps(report))
    state['stages'][key].update(status='validated', report=path.name, report_sha256=sha(path))
    return path


class ProgramPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan, self.state = fixture()

    def next(self, resources=None):
        return select_next(self.root, self.plan, self.state, resources)

    def test_initial_stage(self):
        self.assertEqual(self.next()['stage'], 'A')

    def test_continue_instead_of_stop_after_one_stage(self):
        receipt(self.root, self.state, 'A')
        self.assertEqual(self.next()['stage'], 'B')

    def test_missing_gpu_does_not_block_release(self):
        for key in ('A', 'B'):
            receipt(self.root, self.state, key)
        self.assertEqual(self.next()['stage'], 'F')

    def test_authorized_runtime_selected(self):
        for key in ('A', 'B'):
            receipt(self.root, self.state, key)
        self.assertEqual(self.next({'grants': {'gpu': {'authorized': True, 'authorization_ref': 'test-only'}}})['stage'], 'R')

    def test_string_true_is_not_authorization(self):
        for key in ('A', 'B'):
            receipt(self.root, self.state, key)
        self.assertEqual(self.next({'grants': {'gpu': {'authorized': 'true', 'authorization_ref': 'test-only'}}})['stage'], 'F')

    def test_authorization_needs_source(self):
        for key in ('A', 'B'):
            receipt(self.root, self.state, key)
        self.assertEqual(self.next({'grants': {'gpu': {'authorized': True}}})['stage'], 'F')

    def test_missing_report_requires_revalidation(self):
        self.state['stages']['A']['status'] = 'validated'
        self.assertEqual(self.next()['action'], 'revalidate')
        self.assertEqual(self.next()['stage'], 'A')

    def test_changed_log_requires_revalidation(self):
        receipt(self.root, self.state, 'A')
        (self.root / 'A.log').write_text('changed')
        self.assertEqual(self.next()['stage'], 'A')

    def test_report_cannot_omit_tests(self):
        path = receipt(self.root, self.state, 'A')
        report = read(path)
        report['checks'] = []
        path.write_text(json.dumps(report))
        self.state['stages']['A']['report_sha256'] = sha(path)
        self.assertEqual(self.next()['stage'], 'A')

    def test_failed_test_not_validated(self):
        path = receipt(self.root, self.state, 'A')
        report = read(path)
        report['checks'][0]['returncode'] = 1
        path.write_text(json.dumps(report))
        self.state['stages']['A']['report_sha256'] = sha(path)
        self.assertEqual(self.next()['stage'], 'A')

    def test_dependency_not_bypassed(self):
        receipt(self.root, self.state, 'B')
        result = self.next()
        self.assertEqual(result['stage'], 'A')
        self.assertIn('B', result['integrity_issues'])

    def test_unrelated_stage_can_continue(self):
        receipt(self.root, self.state, 'A')
        self.state['stages']['B'].update(status='blocked_external', blocker='test dependency missing')
        result = self.next({'grants': {'gpu': {'authorized': True, 'authorization_ref': 'test-only'}}})
        self.assertEqual(result['stage'], 'R')

    def test_software_ready_is_not_all_done(self):
        for key in ('A', 'B', 'F'):
            receipt(self.root, self.state, key)
        self.assertEqual(self.next()['status'], 'software_reports_present_runtime_blocked')

    def test_all_reports_only_request_final_review(self):
        for key in ('A', 'B', 'R', 'F'):
            receipt(self.root, self.state, key)
        self.assertEqual(self.next()['action'], 'review_final')

    def test_cycles_rejected(self):
        self.plan['stages'][0]['depends_on'] = ['B']
        with self.assertRaises(ValueError): self.next()

    def test_unknown_dependency_rejected(self):
        self.plan['stages'][0]['depends_on'] = ['missing']
        with self.assertRaises(ValueError): self.next()

    def test_duplicate_stage_rejected(self):
        self.plan['stages'].append(copy.deepcopy(self.plan['stages'][0]))
        with self.assertRaises(ValueError): self.next()

    def test_path_escape_rejected(self):
        for path in ('../secret', '/tmp/secret', ''):
            with self.assertRaises(ValueError): local(self.root, path)

    def test_json_duplicate_key_rejected(self):
        file = self.root / 'bad.json'
        file.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError): read(file)

    def test_blocker_reason_required(self):
        self.state['stages']['A']['status'] = 'blocked_external'
        with self.assertRaises(ValueError): self.next()

    def test_live_plan_dag_only(self):
        plan = read(ROOT / DIRECTORY / 'plan.json')
        state = {'program_id': plan['program_id'], 'stages': {s['id']: {'status': 'pending'} for s in plan['stages']}}
        self.assertEqual(len(validate(plan, state)), 11)
        self.assertEqual(select_next(self.root, plan, state)['stage'], 'M00')


if __name__ == '__main__':
    unittest.main()
