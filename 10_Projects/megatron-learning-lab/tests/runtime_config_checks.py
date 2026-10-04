"""Configuration-only checks; no torch, launcher, model, or Ray imports."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from experiments.runtime.read_only_config import resolve_nemo_config

ROOT=Path(__file__).resolve().parents[1]
class ReadOnlyConfigChecks(unittest.TestCase):
    def setUp(self):
        import hashlib
        archive=ROOT/"tests/fixtures/nemo-config/manifest.json"
        manifest=json.loads(archive.read_text())
        self.assertEqual(manifest["format"],"verbatim_source_archive_v1")
        self.assertEqual(manifest["commit"],"81aa43dda4765b0429cf31dab44441e4e4383911")
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source_root=Path(temporary.name)
        for entry in manifest["files"]:
            raw=entry["content"].encode("utf-8")
            self.assertEqual(hashlib.sha256(raw).hexdigest(),entry["sha256"])
            self.assertEqual(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest(),entry["git_blob_sha"])
            path=self.source_root/entry["path"]
            self.assertTrue(path.resolve().is_relative_to(self.source_root))
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)

    def test_inheritance_override_marker_and_pure_math_follow_hand_example(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"base.yaml").write_text("steps: 2\nmodel:\n  width: 8\n  old: true\n")
            (root/"second.yaml").write_text("steps: 3\nlabel: second\n")
            (root/"child.yaml").write_text("defaults: [base.yaml, second.yaml]\nmodel:\n  _override_: true\n  width: 4\nmetrics:\n  total: ${mul:${steps}, ${model.width}}\n")
            got=resolve_nemo_config(root/"child.yaml",source_root=root,overrides=["steps=5"])
            self.assertEqual(got["config"],dict(steps=5,model=dict(width=4),label="second",metrics=dict(total=20)))
            self.assertEqual(len(got["files"]),3)
            self.assertEqual(len(got["config_sha256"]),64)

    def test_resolver_cannot_read_environment_or_evaluate_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/"x.yaml"
            for content in ("x: ${oc.env:HOME}\n","x: ${eval:'1+1'}\n","x: !!python/object/apply:os.system ['false']\n","x: ${${name}}\n"):
                p.write_text(content)
                with self.assertRaises(ValueError):resolve_nemo_config(p,source_root=root)
            p.write_text("x: 1\n")
            with self.assertRaises(ValueError):
                resolve_nemo_config(p,source_root=root,overrides=["x=${oc.env:HOME}"])

    def test_missing_override_cycle_escape_duplicate_and_nonfinite_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/"x.yaml";p.write_text("x: 1\n")
            with self.assertRaises(ValueError):resolve_nemo_config(p,source_root=root,overrides=["missing=2"])
            for content in ("defaults: x.yaml\nx: 1\n","defaults: ../outside.yaml\nx: 1\n","x: 1\nx: 2\n","x: .nan\n"):
                p.write_text(content)
                with self.assertRaises(ValueError):resolve_nemo_config(p,source_root=root)

    def test_actual_pinned_configs_match_inspected_official_helper(self):
        import hashlib
        from omegaconf import OmegaConf
        root=self.source_root
        path=root/"nemo_rl/utils/config.py"
        raw=path.read_bytes()
        blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        self.assertEqual(blob,"a911e9289addd14a7a04ec1067a47cc765c3bd2d")
        spec=importlib.util.spec_from_file_location("inspected_pinned_config_helper",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.register_omegaconf_resolvers()
        for algo in ("grpo","ppo"):
            config=root/f"examples/configs/{algo}_math_1B.yaml"
            overrides=[f"{algo}.max_num_steps=2","policy.max_total_sequence_length=128"]
            got=resolve_nemo_config(config,source_root=root,overrides=overrides)
            expected=OmegaConf.to_container(module.parse_hydra_overrides(module.load_config(config),overrides),resolve=True)
            self.assertEqual(got["config"],expected)
            self.assertEqual(got["config"][algo]["max_num_steps"],2)
        self.assertNotIn("torch",sys.modules);self.assertNotIn("ray",sys.modules)
        self.assertFalse(any(k.startswith("nemo_rl") for k in sys.modules))

    def test_top_level_parent_reference_is_rejected_like_current_upstream(self):
        # The inspected helper indexes every root override during merge, before
        # inherited keys exist. Do not claim raw upstream compatibility for this
        # branch or silently return a guessed configuration.
        from omegaconf.errors import InterpolationKeyError
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"base.yaml").write_text("steps: 2\n")
            child=root/"child.yaml"
            child.write_text("defaults: base.yaml\ntotal: ${steps}\n")
            with self.assertRaises(InterpolationKeyError):
                resolve_nemo_config(child,source_root=root)
            helper=self.source_root/"nemo_rl/utils/config.py"
            spec=importlib.util.spec_from_file_location("inspected_parent_reference_helper",helper)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with self.assertRaises(InterpolationKeyError):
                module.load_config(child)

    def test_freeze_cli_binds_exact_bytes_and_refuses_to_overwrite(self):
        import hashlib
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root/"example.yaml"
            config.write_text("steps: 2\n")
            output = root/"frozen.json"
            command = [sys.executable, "tools/runtime_cli.py", "freeze-nemo",
                       "--root", str(root), "--config", str(config),
                       "--override", "steps=3", "--output", str(output)]
            env = {k:v for k,v in os.environ.items() if k != "PYTHONPATH"}
            env["CUDA_VISIBLE_DEVICES"] = ""
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            response = json.loads(result.stdout)
            raw = output.read_bytes()
            self.assertEqual(json.loads(raw), {"steps": 3})
            self.assertEqual(response["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(response["bytes"], len(raw))
            self.assertEqual(response["execution"], "not_run")
            again = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(again.returncode, 2)
            self.assertEqual(output.read_bytes(), raw)

    def test_no_launcher_import_during_read_only_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"run_grpo.py").write_text("raise RuntimeError('launcher must not execute')\n")
            config=root/"resolved.yaml";config.write_text("steps: 2\n")
            self.assertEqual(resolve_nemo_config(config,source_root=root)["config"]["steps"],2)
            self.assertNotIn("torch",sys.modules);self.assertNotIn("ray",sys.modules)


if __name__=="__main__":unittest.main(verbosity=2)
