"""Synthetic constructor/registry contracts; no Ray or NeMo import."""
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from experiments.runtime.nemo_extension import (
    OFFICIAL_WORKERS, VALUE_WORKER, WORKER, WorkerExtensionScope,
)


class SyntheticPolicy:
    def __init__(self, config, *, worker_extension_cls_fqn=None):
        self.config = config
        self.worker_extension = worker_extension_cls_fqn


def synthetic_grpo_setup(config, *, policy_factory=None):
    return policy_factory(config), config


class NeMoExtensionTests(unittest.TestCase):
    def registry(self):
        return {key: sys.executable for key in (*OFFICIAL_WORKERS, VALUE_WORKER)}

    def test_grpo_factory_preserves_constructor_input_and_registers_importable_name(self):
        registry = self.registry()
        before = registry.copy()
        scope = WorkerExtensionScope(SyntheticPolicy, registry, algorithm="grpo")
        config = object()
        with patch.dict(os.environ, {"NEMO_RL_PY_EXECUTABLES_SYSTEM": "1"}):
            scope.open()
            self.assertEqual(registry[WORKER], sys.executable)
            result, original = scope.setup(synthetic_grpo_setup, config)
            self.assertIsInstance(result, SyntheticPolicy)
            self.assertIs(result.config, config)
            self.assertIs(original, config)
            self.assertEqual(result.worker_extension, WORKER)
            # This is a module/class name for the isolated initializer, not a
            # driver-only substitution of an already imported Ray actor class.
            module, name = WORKER.rsplit(".", 1)
            self.assertTrue((Path(__file__).resolve().parents[1]/(module.replace(".", "/")+".py")).is_file())
            self.assertEqual(name, "ObservedMegatronPolicyWorker")
            scope.close()
        self.assertEqual(registry, before)

    def test_ppo_constructor_binding_restores_on_success_and_setup_failure(self):
        for fail in (False, True):
            module = ModuleType("synthetic_ppo_setup")
            module.Policy = SyntheticPolicy
            exec("def setup(config):\n policy = Policy(config)\n"
                 + (" raise RuntimeError('synthetic setup failure')\n" if fail else " return policy\n"),
                 module.__dict__)
            registry = self.registry()
            scope = WorkerExtensionScope(SyntheticPolicy, registry, algorithm="ppo")
            with patch.dict(os.environ, {"NEMO_RL_PY_EXECUTABLES_SYSTEM": "1"}):
                scope.open()
                try:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "setup failure"):
                            scope.setup(module.setup, {})
                    else:
                        self.assertIsInstance(scope.setup(module.setup, {}), SyntheticPolicy)
                    self.assertIs(module.Policy, SyntheticPolicy)
                finally:
                    scope.close()
            self.assertNotIn(WORKER, registry)

    def test_unapproved_worker_environment_fails_before_registry_mutation(self):
        for key in (*OFFICIAL_WORKERS, VALUE_WORKER):
            registry = self.registry()
            registry[key] = "uv run --group unwanted"
            before = registry.copy()
            scope = WorkerExtensionScope(SyntheticPolicy, registry, algorithm="ppo")
            with patch.dict(os.environ, {"NEMO_RL_PY_EXECUTABLES_SYSTEM": "1"}):
                with self.assertRaisesRegex(ValueError, "unapproved Python"):
                    scope.open()
            self.assertEqual(registry, before)

    def test_duplicate_extension_and_missing_environment_choice_fail_closed(self):
        registry = self.registry()
        registry[WORKER] = "/existing/owner/python"
        scope = WorkerExtensionScope(SyntheticPolicy, registry, algorithm="grpo")
        with self.assertRaisesRegex(ValueError, "already registered"):
            scope.open()
        self.assertEqual(registry[WORKER], "/existing/owner/python")
        del registry[WORKER]
        with patch.dict(os.environ, {"NEMO_RL_PY_EXECUTABLES_SYSTEM": "0"}):
            with self.assertRaisesRegex(ValueError, "current-environment"):
                scope.open()
        self.assertNotIn(WORKER, registry)

    def test_unknown_interfaces_do_not_guess_or_import_a_runtime(self):
        with self.assertRaisesRegex(ValueError, "lacks"):
            WorkerExtensionScope(lambda config: config, self.registry(), algorithm="grpo")
        scope = WorkerExtensionScope(SyntheticPolicy, self.registry(), algorithm="grpo")
        with patch.dict(os.environ, {"NEMO_RL_PY_EXECUTABLES_SYSTEM": "1"}):
            scope.open()
            try:
                with self.assertRaisesRegex(ValueError, "conflicting"):
                    scope.policy_factory({}, worker_extension_cls_fqn="unknown.Worker")
                with self.assertRaisesRegex(ValueError, "lacks"):
                    scope.setup(lambda: None)
            finally:
                scope.close()
        self.assertFalse(any(name.split(".")[0] in ("torch", "ray", "nemo_rl") for name in sys.modules))
