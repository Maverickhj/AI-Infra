import os
from pathlib import Path
import subprocess
import unittest

class TPDPSubprocessTests(unittest.TestCase):
    def test_actual_full_model_tp_dp_suite(self):
        env=dict(os.environ);env.pop('PYTHONPATH',None);env['CUDA_VISIBLE_DEVICES']=''
        p=subprocess.run(['python','-m','tests.tp_dp_cpu_checks'],cwd=Path(__file__).resolve().parents[1],env=env,text=True,capture_output=True,timeout=180)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertIn('Ran 6 tests',p.stderr)
        self.assertNotIn('skipped',p.stderr)
