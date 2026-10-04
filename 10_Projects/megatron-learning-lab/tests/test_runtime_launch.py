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


    def test_detached_double_fork_is_reaped_without_touching_sibling_or_parent_state(self):
        import ctypes
        import signal
        import time
        libc=ctypes.CDLL(None,use_errno=True)
        before=ctypes.c_int()
        self.assertEqual(libc.prctl(37,ctypes.byref(before),0,0,0),0)
        sibling=subprocess.Popen([sys.executable,"-S","-c","import time;time.sleep(30)"],
                                 start_new_session=True)
        self.addCleanup(lambda: sibling.poll() is None and sibling.kill())
        try:
            for early_exit in (True,False):
                with self.subTest(early_exit=early_exit),tempfile.TemporaryDirectory() as folder:
                    marker=Path(folder)/"detached.pid"
                    code=f"""import os,signal,time
from pathlib import Path
marker=Path({str(marker)!r})
if os.fork()==0:
 os.setsid()
 if os.fork(): os._exit(0)
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 marker.write_text(str(os.getpid()))
 time.sleep(30)
 os._exit(0)
limit=time.monotonic()+2
while not marker.exists() and time.monotonic()<limit: time.sleep(.01)
assert marker.exists()
{'raise SystemExit(0)' if early_exit else 'time.sleep(30)'}
"""
                    result=supervised_process([sys.executable,"-S","-c",code],
                        cwd=ROOT,env=dict(os.environ),log_path=Path(folder)/"out.log",
                        max_seconds=3 if early_exit else .3)
                    self.assertEqual(result["ownership"],"dedicated_linux_subreaper_pidfd")
                    self.assertEqual(result["descendant_cleanup"]["status"],"completed")
                    self.assertGreaterEqual(result["descendant_cleanup"]["descendants_seen"],2)
                    self.assertEqual(result["status"],"passed" if early_exit else "failed")
                    self.assertEqual(result["stop_reason"],None if early_exit else "wall_time_limit")
                    pid=int(marker.read_text())
                    with self.assertRaises(ProcessLookupError):os.kill(pid,0)
                    self.assertIsNone(sibling.poll())
                    self.assertLess(result["elapsed_wall_seconds"],6)
        finally:
            sibling.terminate();sibling.wait(timeout=5)
        after=ctypes.c_int()
        self.assertEqual(libc.prctl(37,ctypes.byref(after),0,0,0),0)
        self.assertEqual(after.value,before.value)

    def test_parent_exit_interrupts_guard_and_reaps_detached_runtime(self):
        import time
        with tempfile.TemporaryDirectory() as folder:
            marker=Path(folder)/"worker.pid"
            log=Path(folder)/"runtime.log"
            worker=f"""import os,signal,time
from pathlib import Path
os.setsid()
signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path({str(marker)!r}).write_text(str(os.getpid()))
time.sleep(30)
"""
            # The supervised leader forks a separately-sessioned worker, as a
            # framework launcher can. Only these known synthetic processes run.
            leader="import subprocess,sys,time;subprocess.Popen([sys.executable,'-S','-c',"+repr(worker)+"]);time.sleep(30)"
            code=("from experiments.runtime.launch import supervised_process;import os,sys;"
                  "supervised_process([sys.executable,'-S','-c',"+repr(leader)+"],cwd="+repr(str(ROOT))+
                  ",env=dict(os.environ),log_path="+repr(str(log))+",max_seconds=20)")
            parent=subprocess.Popen([sys.executable,"-S","-c",code],cwd=ROOT,
                                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                deadline=time.monotonic()+5
                while not marker.exists() and parent.poll() is None and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertTrue(marker.exists())
                pid=int(marker.read_text())
                parent.terminate();parent.wait(timeout=5)
                deadline=time.monotonic()+6
                while Path(f"/proc/{pid}").exists() and time.monotonic()<deadline:
                    time.sleep(.02)
                with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            finally:
                if parent.poll() is None:
                    parent.kill()
                parent.communicate(timeout=10)
