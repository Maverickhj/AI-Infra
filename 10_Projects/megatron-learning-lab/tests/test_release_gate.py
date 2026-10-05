"""Synthetic gate-control tests; never evidence of a real release or runtime."""
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import release_gate


class ReleaseGateTests(unittest.TestCase):
    def test_runtime_blockers_never_become_full_completion(self):
        validated={"status":"validated"}
        blocked={"status":"blocked_external","blocker":"synthetic missing resource"}
        self.assertEqual(release_gate.release_status({"R01":validated,"R02":blocked}),
                         "software_ready_runtime_blocked")
        self.assertEqual(release_gate.release_status({"R01":validated,"R02":validated}),
                         "full_project_ready_for_review")
        for bad in ({"status":"pending"},{"status":"failed"},{"status":"blocked_external","blocker":""}):
            with self.subTest(status=bad),self.assertRaises(ValueError):
                release_gate.release_status({"R01":validated,"R02":bad})

    def test_missing_or_incomplete_baseline_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"report.json"
            for data in ({}, {"status":"baseline_passed","mode":"baseline","commands":[]},
                         {"status":"ready_for_review","mode":"final","commands":[]}):
                p.write_text(json.dumps(data))
                with self.subTest(data=data),self.assertRaises(ValueError):
                    release_gate.validate_baseline(p)

    def test_regression_stops_on_failure_and_retains_actual_returncode(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/"fresh"
            with patch.object(release_gate,"execute",return_value={
                "status":"failed","returncode":17,"command":["synthetic-only"],
                "log":"synthetic.log","log_sha256":"0"*64}) as execute:
                self.assertEqual(release_gate.regression(folder),1)
            self.assertEqual(execute.call_count,1)
            report=json.loads((folder/"report.json").read_text())
            self.assertEqual(report["status"],"failed")
            self.assertEqual(report["checks"][0]["returncode"],17)

    def test_regression_refuses_to_overwrite_prior_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(release_gate,"execute") as execute:
                with self.assertRaises(FileExistsError):
                    release_gate.regression(Path(directory))
                execute.assert_not_called()


    def test_dependency_versions_match_the_normal_cpu_interpreter(self):
        # Independent query checks the installed selection, including venv/site
        # precedence; the release driver itself runs with -S for stdlib checks.
        env=dict(os.environ);env.pop("PYTHONPATH",None)
        query="import importlib.metadata as m,json,sys;print(json.dumps({'prefix':sys.prefix,'versions':{k:m.version(k) for k in ('numpy','transformers','torch')}}))"
        expected=json.loads(subprocess.check_output([sys.executable,"-c",query],text=True,env=env))
        actual=release_gate.dependencies()
        self.assertEqual(actual["python_prefix"],expected["prefix"])
        for name,version in expected["versions"].items():
            self.assertEqual(actual["python_packages"][name],version,name)


    def test_real_quiet_command_keeps_its_zero_output_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(release_gate,"ROOT",root):
                command=release_gate.execute(root,[sys.executable,"-S","-c","pass"],root/"quiet.log",10)
                command["id"]="quiet"
                self.assertEqual(command["returncode"],0)
                evidence=release_gate.command_evidence(command,root)
                self.assertGreater(evidence.stat().st_size,0)
                self.assertEqual((root/"quiet.log").read_bytes(),b"")
                receipt=json.loads(evidence.read_text())
                self.assertEqual(receipt["execution"],command)
                self.assertEqual(receipt["output_bytes"],0)
                (root/"quiet.log").write_text("changed")
                with self.assertRaisesRegex(ValueError,"hash changed"):
                    release_gate.command_evidence(command,root)
