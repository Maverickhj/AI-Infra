"""Keep stdlib metadata discovery independent from the Torch environment."""
import os
from pathlib import Path
import subprocess
import unittest


class RuntimeAdapterProcessTests(unittest.TestCase):
    def test_actual_cpu_adapter_contracts(self):
        env={k:v for k,v in os.environ.items() if k!="PYTHONPATH"}
        env["CUDA_VISIBLE_DEVICES"]=""
        result=subprocess.run(["python","-m","tests.runtime_adapter_cpu_checks"],cwd=Path(__file__).resolve().parents[1],
                              env=env,text=True,capture_output=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("Ran 11 tests",result.stderr)

    def test_configuration_only_contracts(self):
        env={k:v for k,v in os.environ.items() if k!="PYTHONPATH"}
        env["CUDA_VISIBLE_DEVICES"]=""
        result=subprocess.run(["python","-m","tests.runtime_config_checks"],cwd=Path(__file__).resolve().parents[1],
                              env=env,text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("Ran 7 tests",result.stderr)

    def test_nemo_loss_and_worker_cpu_contracts(self):
        env={k:v for k,v in os.environ.items() if k!="PYTHONPATH"}
        env["CUDA_VISIBLE_DEVICES"]=""
        result=subprocess.run(["python","-m","tests.runtime_nemo_capture_checks"],
                              cwd=Path(__file__).resolve().parents[1], env=env,
                              text=True,capture_output=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("Ran 9 tests",result.stderr)
