"""Keep the stdlib discovery path; execute CPU autograd in the installed Torch interpreter."""
import os
from pathlib import Path
import subprocess
import unittest

class DecoderSubprocessTests(unittest.TestCase):
    def test_real_cpu_autograd_and_resume_suite(self):
        env=dict(os.environ);env.pop('PYTHONPATH',None);env['CUDA_VISIBLE_DEVICES']=''
        result=subprocess.run(['python','-m','tests.decoder_cpu_checks'],cwd=Path(__file__).resolve().parents[1],
                              env=env,text=True,capture_output=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Ran 7 tests',result.stderr)
        self.assertNotIn('skipped',result.stderr)
