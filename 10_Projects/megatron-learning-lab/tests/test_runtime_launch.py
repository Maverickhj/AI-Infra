"""Actual subprocess limits and explicitly synthetic launch receipts; no GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.runtime.launch import launch, supervised_process, verify_worker
from tests import test_runtime_plan as plan_tests

ROOT=Path(__file__).resolve().parents[1]


class RuntimeSupervisorTests(unittest.TestCase):
    def test_actual_child_exit_status_and_text_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/"out.log"
            result=supervised_process([sys.executable,"-S","-c","print('synthetic child');raise SystemExit(7)"],
                cwd=ROOT,env=dict(os.environ),log_path=log,max_seconds=3)
            self.assertEqual(result["returncode"],7)
            self.assertEqual(result["status"],"failed")
            self.assertIsNone(result["stop_reason"])
            self.assertEqual(log.read_text(),"synthetic child\n")

    def test_actual_timeout_and_output_bound_stop_only_owned_children(self):
        cases=[("import time;time.sleep(20)",.15,100,"wall_time_limit"),
               ("print('x'*10000)",3,100,"output_limit")]
        for code,seconds,limit,reason in cases:
            with tempfile.TemporaryDirectory() as folder:
                log=Path(folder)/"out.log"
                result=supervised_process([sys.executable,"-S","-c",code],cwd=ROOT,env=dict(os.environ),
                    log_path=log,max_seconds=seconds,max_output_bytes=limit)
                self.assertEqual(result["stop_reason"],reason)
                self.assertEqual(result["status"],"failed")
                self.assertLessEqual(log.stat().st_size,limit)
                self.assertLess(result["elapsed_wall_seconds"],6)

    def test_denied_grant_precedes_import_inspection_and_output_creation(self):
        sample=plan_tests.RuntimePlanTests()
        sample.setUp()
        self.addCleanup(sample.doCleanups)
        resource=sample.root/"resources.json"
        resource.write_text(json.dumps(dict(grants={"bridge_sft":dict(authorized=False)})))
        with patch("experiments.runtime.launch.inspect_plan",side_effect=AssertionError("must not inspect")):
            with self.assertRaisesRegex(ValueError,"not authorized"):
                launch(sample.path,resource)
        self.assertFalse(Path(sample.plan["execution"]["output_path"]).exists())
        self.assertNotIn("torch",sys.modules)

    def test_synthetic_supervisor_cannot_report_success_without_worker_result(self):
        sample=plan_tests.RuntimePlanTests()
        sample.setUp()
        self.addCleanup(sample.doCleanups)
        resource=sample.root/"resources.json";resource.write_text(json.dumps(sample.grant()))
        observed={}
        def synthetic_supervisor(argv,**kwargs):
            observed.update(argv=argv,env=kwargs["env"])
            return dict(status="passed",returncode=0,stop_reason=None)
        with patch("experiments.runtime.launch.supervised_process",synthetic_supervisor):
            result=launch(sample.path,resource)
        self.assertEqual(result["execution"],"failed")
        self.assertEqual(result["result"]["stop_reason"],"missing_or_invalid_result")
        self.assertEqual(observed["env"]["CUDA_VISIBLE_DEVICES"],"0")
        self.assertEqual(observed["env"]["HF_HUB_OFFLINE"],"1")
        self.assertNotIn("PYTHONPATH",observed["env"])
        self.assertIn("experiments.runtime.worker",observed["argv"])
        self.assertNotIn("torch",sys.modules)

    def test_actual_cli_refuses_ungranted_run_before_any_framework_import(self):
        sample=plan_tests.RuntimePlanTests()
        sample.setUp()
        self.addCleanup(sample.doCleanups)
        resource=sample.root/"resources.json"
        resource.write_text(json.dumps({"grants":{"bridge_sft":{"authorized":False}}}))
        child=subprocess.run([sys.executable,"-S","tools/runtime_cli.py","run","--plan",str(sample.path),
                              "--resources",str(resource)],cwd=ROOT,text=True,capture_output=True,timeout=5)
        self.assertEqual(child.returncode,2,child.stdout+child.stderr)
        self.assertIn("not authorized",json.loads(child.stderr)["error"])
        self.assertFalse(Path(sample.plan["execution"]["output_path"]).exists())
