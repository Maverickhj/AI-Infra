"""Run the real float64 CPU checks without importing Torch into stdlib discovery."""
import os
from pathlib import Path
import subprocess
import unittest

class MoEMLASubprocessTests(unittest.TestCase):
    def test_actual_complete_family_decoder_suite(self):
        env=dict(os.environ);env.pop('PYTHONPATH',None);env['CUDA_VISIBLE_DEVICES']=''
        result=subprocess.run(['python','-m','tests.moe_mla_cpu_checks'],cwd=Path(__file__).resolve().parents[1],
                              env=env,text=True,capture_output=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Ran 8 tests',result.stderr)
        self.assertNotIn('skipped',result.stderr)
