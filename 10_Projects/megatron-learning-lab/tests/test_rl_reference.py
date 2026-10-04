"""Execute actual CPU autograd RL checks from stdlib discovery."""
import os
from pathlib import Path
import subprocess
import unittest
class RLReferenceSubprocessTests(unittest.TestCase):
    def test_actual_grpo_ppo_loss_gradient_update_contract_suite(self):
        env=dict(os.environ);env.pop('PYTHONPATH',None);env['CUDA_VISIBLE_DEVICES']=''
        r=subprocess.run(['python','-m','tests.rl_cpu_checks'],cwd=Path(__file__).resolve().parents[1],env=env,text=True,capture_output=True,timeout=180)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn('Ran 8 tests',r.stderr)
        self.assertNotIn('skipped',r.stderr)
