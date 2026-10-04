"""Pinned official launcher function bodies with explicitly synthetic dependencies.

These tests prove delegation and cleanup only. They import no NeMo/Torch/Ray
runtime and do not claim a production setup, tokenizer, rollout or optimizer.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.runtime.nemo_entry import OwnedRay, invoke_official
from tests import test_runtime_nemo_plan as plans

ROOT = Path(__file__).resolve().parents[1]


class SyntheticRay:
    def __init__(self):
        self.live = False
        self.init_calls = []
        self.shutdown_calls = 0
        self.gpus = 1

    def is_initialized(self):
        return self.live

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.live = True

    def cluster_resources(self):
        return {"GPU": self.gpus, "CPU": 4}

    def nodes(self):
        return [{"Alive": True}]

    def shutdown(self):
        self.shutdown_calls += 1
        self.live = False


class SyntheticMaster:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._raw = kwargs

    def model_dump(self):
        return copy.deepcopy(self._raw)


class SyntheticObserver:
    def __init__(self):
        self.configs = []
        self.attached = []
        self.closed = False

    def configure(self, actual, sha):
        self.configs.append((actual, sha))

    def attach(self, result, launcher):
        self.attached.append(result)
        return result

    def finish(self):
        return {"status": "synthetic_launcher_completed"}

    def close(self):
        self.closed = True


class NeMoLauncherDelegationTests(unittest.TestCase):
    def setUp(self):
        self.profile = plans.NeMoBoundPlanTests()
        self.profile.setUp()
        self.addCleanup(self.profile.doCleanups)

    def launcher(self, algorithm, *, fail_setup=False):
        archive = json.loads((ROOT/"tests/fixtures/nemo-config/manifest.json").read_text())
        entry = next(x for x in archive["files"] if x["path"] == f"examples/run_{algorithm}.py")
        raw = entry["content"].encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
        self.assertEqual(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest(), entry["git_blob_sha"])
        parsed = ast.parse(raw)
        functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("parse_args", "main", "_select_trainer")]
        self.assertEqual({x.name for x in functions},
                         {"parse_args", "main", "_select_trainer"} if algorithm == "grpo" else {"parse_args", "main"})
        module = ModuleType("synthetic_dependencies_for_pinned_"+algorithm)
        calls, policy, generation, loss, value_loss = [], object(), object(), object(), object()

        def setup(config, tokenizer, dataset, val, **kwargs):
            calls.append(("setup", config, tokenizer, dataset, val, kwargs))
            if fail_setup:
                raise RuntimeError("synthetic setup failure")
            if algorithm == "grpo":
                return (policy, generation, None, "cluster", "loader", None, loss,
                        "logger", "checkpoint", {"total_steps": 0}, config, {}, {})
            return (policy, generation, "critic", "cluster", "loader", None, loss, value_loss,
                    "logger", "checkpoint", {"total_steps": 0}, config)

        def train(*args):
            calls.append(("train", args))

        def response_data(tokenizer, data, env):
            # This is the real helper's inspected single-dataset normalization.
            data["train"] = [data["train"]]
            return "dataset", None, {"math": "synthetic env"}, {}

        module.__dict__.update(argparse=argparse, os=os, pprint=SimpleNamespace(pprint=lambda _: None),
            MasterConfig=SyntheticMaster, OmegaConf=SimpleNamespace(to_container=lambda value, **_: value),
            register_omegaconf_resolvers=lambda: None,
            load_config=lambda path: json.loads(Path(path).read_text()),
            parse_hydra_overrides=lambda *_: self.fail("frozen config must have no overrides"),
            get_next_experiment_dir=lambda path: path+"/run_1",
            init_ray=lambda: self.fail("official auto-attach initializer must be replaced"),
            get_tokenizer=lambda _: "synthetic tokenizer",
            configure_generation_config=lambda config, tokenizer, **kwargs: config,
            setup_response_data=response_data, setup=setup, grpo_train=train, ppo_train=train)
        # Execute only the archived function bodies; all imports/dependencies are
        # supplied above as synthetic objects. No production module is imported.
        exec(compile(ast.Module(body=functions, type_ignores=[]), entry["path"], "exec"), module.__dict__)
        return module, calls, policy, generation, loss, value_loss

    def invoke(self, algorithm, **kwargs):
        plan, _ = self.profile.bound(algorithm)
        output = Path(plan["execution"]["output_path"]); output.mkdir()
        launcher, calls, policy, generation, loss, value_loss = self.launcher(algorithm, **kwargs)
        observer, ray = SyntheticObserver(), SyntheticRay()
        owned = OwnedRay(ray, plan)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "RAY_ADDRESS": "",
                                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            result = invoke_official(launcher, plan, output, observer, owned)
        return result, calls, policy, generation, loss, value_loss, observer, ray

    def test_official_grpo_and_ppo_main_delegate_once_with_original_arguments(self):
        for algorithm in ("grpo", "ppo"):
            got, calls, policy, generation, loss, value_loss, observer, ray = self.invoke(algorithm)
            self.assertEqual(got["status"], "synthetic_launcher_completed")
            self.assertEqual([row[0] for row in calls], ["setup", "train"])
            args = calls[1][1]
            self.assertEqual(len(args), 12 if algorithm == "grpo" else 14)
            self.assertIs(args[0], policy); self.assertIs(args[1], generation)
            self.assertIs(args[5 if algorithm == "grpo" else 6], loss)
            if algorithm == "ppo":
                self.assertIs(args[7], value_loss)
            self.assertEqual(len(observer.configs), 1)
            self.assertTrue(observer.closed)
            self.assertEqual(ray.init_calls[0]["address"], "local")
            self.assertEqual(ray.init_calls[0]["num_gpus"], 1)
            self.assertFalse(ray.init_calls[0]["include_dashboard"])
            self.assertEqual(ray.shutdown_calls, 1)
            self.assertFalse(Path(ray.init_calls[0]["_temp_dir"]).exists())

    def test_setup_failure_restores_official_globals_argv_and_local_ray(self):
        plan, _ = self.profile.bound("grpo")
        output = Path(plan["execution"]["output_path"]); output.mkdir()
        module, *_ = self.launcher("grpo", fail_setup=True)
        before = (module.setup, module.init_ray, module.MasterConfig, sys.argv)
        ray, observer = SyntheticRay(), SyntheticObserver()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "RAY_ADDRESS": ""}):
            with self.assertRaisesRegex(RuntimeError, "synthetic setup failure"):
                invoke_official(module, plan, output, observer, OwnedRay(ray, plan))
        self.assertEqual((module.setup, module.init_ray, module.MasterConfig, sys.argv), before)
        self.assertTrue(observer.closed)
        self.assertEqual(ray.shutdown_calls, 1)
        self.assertFalse(ray.live)

    def test_unbound_configuration_fails_before_ray_tokenizer_or_setup(self):
        plan, config = self.profile.bound("grpo")
        config["cluster"]["num_nodes"] = 2
        Path(plan["training"]["nemo_config"]).write_text(json.dumps(config))
        output = Path(plan["execution"]["output_path"]); output.mkdir()
        module, calls, *_ = self.launcher("grpo")
        ray = SyntheticRay()
        with self.assertRaisesRegex(ValueError, "cluster.num_nodes"):
            invoke_official(module, plan, output, SyntheticObserver(), OwnedRay(ray, plan))
        self.assertEqual(calls, [])
        self.assertEqual(ray.init_calls, [])
        self.assertEqual(ray.shutdown_calls, 0)

    def test_existing_cluster_and_changed_device_are_not_touched(self):
        plan, _ = self.profile.bound("grpo")
        ray = SyntheticRay(); ray.live = True
        with self.assertRaisesRegex(ValueError, "existing"):
            OwnedRay(ray, plan).start()
        self.assertTrue(ray.live)
        self.assertEqual(ray.shutdown_calls, 0)
        ray.live = False
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with self.assertRaisesRegex(ValueError, "device"):
                OwnedRay(ray, plan).start()
        self.assertEqual(ray.init_calls, [])

    def test_unfinished_production_entry_rejects_before_imports_or_outputs(self):
        from experiments.runtime.nemo_entry import run
        output = self.profile.root/"must-not-exist"
        with self.assertRaisesRegex(ValueError, "wiring is not complete"):
            run({}, output)
        self.assertFalse(output.exists())
        self.assertFalse(any(name.split(".")[0] in ("torch", "ray", "nemo_rl")
                             for name in sys.modules))

    def test_resource_mismatch_cleans_only_new_owned_context(self):
        plan, _ = self.profile.bound("grpo")
        ray = SyntheticRay(); ray.gpus = 2
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "RAY_ADDRESS": ""}):
            with self.assertRaisesRegex(ValueError, "GPU resources"):
                OwnedRay(ray, plan).start()
        self.assertEqual(ray.shutdown_calls, 1)
        self.assertFalse(ray.live)
        self.assertFalse(Path(ray.init_calls[0]["_temp_dir"]).exists())
