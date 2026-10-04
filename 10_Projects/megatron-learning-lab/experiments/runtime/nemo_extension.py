"""Use the official policy worker extension point in the current Python environment.

The Ray initializer imports worker classes by name in a separate process. Patching
a worker class in the driver cannot install a worker extension. No dependency
resolver, installer, or alternate training implementation is invoked here.
"""
from __future__ import annotations
import inspect
import os
from pathlib import Path
import sys
from .contracts import require

WORKER = "experiments.runtime.nemo_worker.ObservedMegatronPolicyWorker"
OFFICIAL_WORKERS = (
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
    "nemo_rl.environments.math_environment.MathEnvironment",
)
VALUE_WORKER = "nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker"


class WorkerExtensionScope:
    """Temporarily register a named worker and delegate construction to Policy."""
    def __init__(self, policy_class, registry, *, algorithm):
        require(algorithm in ("grpo", "ppo"), "unknown worker extension profile")
        require("worker_extension_cls_fqn" in inspect.signature(policy_class).parameters,
                "official Policy lacks its worker extension interface")
        self.policy_class, self.registry, self.algorithm = policy_class, registry, algorithm
        self.active = False

    def open(self):
        require(not self.active and WORKER not in self.registry,
                "worker extension is already registered")
        require(os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM") == "1",
                "explicit current-environment worker execution is required")
        expected = Path(sys.executable).resolve()
        workers = OFFICIAL_WORKERS + ((VALUE_WORKER,) if self.algorithm == "ppo" else ())
        for name in workers:
            value = self.registry.get(name)
            require(isinstance(value, str) and Path(value).is_absolute()
                    and Path(value).resolve() == expected,
                    "worker would use an unapproved Python environment: " + name)
        self.registry[WORKER] = sys.executable
        self.active = True

    def policy_factory(self, *args, **kwargs):
        require(self.active, "worker extension scope is not active")
        require("worker_extension_cls_fqn" not in kwargs,
                "conflicting policy worker extension")
        return self.policy_class(*args, **kwargs, worker_extension_cls_fqn=WORKER)

    def setup(self, original, *args, **kwargs):
        require(self.active, "worker extension scope is not active")
        if self.algorithm == "grpo":
            require("policy_factory" in inspect.signature(original).parameters,
                    "GRPO setup lacks the inspected policy factory interface")
            require("policy_factory" not in kwargs, "conflicting setup policy factory")
            return original(*args, **kwargs, policy_factory=self.policy_factory)
        # The inspected PPO setup has no factory parameter. Replace only its
        # driver-side Policy constructor, restoring it even if threaded setup fails.
        namespace = original.__globals__
        require(namespace.get("Policy") is self.policy_class,
                "unknown PPO Policy constructor binding")
        namespace["Policy"] = self.policy_factory
        try:
            return original(*args, **kwargs)
        finally:
            namespace["Policy"] = self.policy_class

    def close(self):
        if self.active:
            require(self.registry.get(WORKER) == sys.executable,
                    "worker registry changed during the owned extension scope")
            del self.registry[WORKER]
            self.active = False
