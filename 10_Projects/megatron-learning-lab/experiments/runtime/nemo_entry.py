"""Delegate to the inspected official synchronous NeMo launcher.

The injectable delegation boundary is tested with synthetic dependencies.
Production run remains disabled until observer and supervisor wiring is complete.
This module implements no RL trainer and never auto-attaches to a cluster.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

from .capture import effective_config
from .contracts import require
from .nemo_config import validate_bound_config
from .plan import ROOT


class OwnedRay:
    """One explicit local Ray instance, with shutdown even if setup/train fails."""
    def __init__(self, ray, plan):
        self.ray, self.plan = ray, plan
        self.started = False
        self.scratch = None

    def start(self, log_dir=None):
        require(not self.started and not self.ray.is_initialized(),
                "refusing an existing or duplicate Ray session")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(self.plan["execution"]["devices"]),
                "Ray device environment differs from approved plan")
        require(not os.environ.get("RAY_ADDRESS"), "external Ray address is forbidden")
        self.scratch = tempfile.TemporaryDirectory(prefix="lab-ray-")
        env = {key: os.environ[key] for key in
               ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE", "WANDB_DISABLED", "OMP_NUM_THREADS")
               if key in os.environ}
        # Ray workers must be able to import this small observation module and
        # the explicitly inspected local NeMo checkout. No user PYTHONPATH is reused.
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + self.plan["sources"]["nemo_rl"]
        try:
            self.ray.init(address="local", num_gpus=len(self.plan["execution"]["devices"]),
                num_cpus=4, include_dashboard=False, log_to_driver=True,
                namespace="megatron-lab-" + self.plan["run_id"], _node_ip_address="127.0.0.1",
                _temp_dir=self.scratch.name, runtime_env={"env_vars": env})
            self.started = True
            resources = self.ray.cluster_resources()
            require(resources.get("GPU") == len(self.plan["execution"]["devices"]),
                    "Ray GPU resources differ from the authorized count")
            require(len([node for node in self.ray.nodes() if node.get("Alive")]) == 1,
                    "the capture profile requires one local Ray node")
        except BaseException:
            self.close()
            raise

    def close(self):
        # shutdown only touches the local context created above; never ray stop.
        if self.scratch is not None:
            try:
                self.ray.shutdown()
            finally:
                self.started = False
                self.scratch.cleanup()
                self.scratch = None


def runtime_config(config, plan):
    """Allow only known launcher normalizations before rechecking resource limits."""
    actual = config.model_dump()
    audit = copy.deepcopy(actual)
    train = audit["data"]["train"]
    if isinstance(train, list):
        require(len(train) == 1, "runtime expanded the dataset selection")
        audit["data"]["train"] = train[0]
    base = Path(plan["execution"]["output_path"])/"logs"
    require(Path(audit["logger"]["log_dir"]).resolve().is_relative_to(base.resolve()),
            "runtime logger escaped approved output")
    audit["logger"]["log_dir"] = str(base)
    validate_bound_config(audit, plan)
    return actual


def invoke_official(launcher, plan, output, observer, owned_ray):
    """Small injectable boundary; tests use explicitly synthetic dependencies."""
    originals = {name: getattr(launcher, name) for name in
                 ("MasterConfig", "init_ray", "setup")}
    argv = sys.argv
    configured = []

    def master(**kwargs):
        instance = originals["MasterConfig"](**kwargs)
        # This happens in the official main before Ray, tokenizer, or data setup.
        validate_bound_config(instance.model_dump(), plan)
        configured.append(instance)
        return instance

    def setup(*args, **kwargs):
        require(len(configured) == 1 and args and args[0] is configured[0],
                "unknown official configuration/setup handoff")
        actual = runtime_config(args[0], plan)
        encoded, sha = effective_config(actual)
        (output/"effective-config.json").write_text(json.dumps(encoded, ensure_ascii=False, indent=2)+"\n")
        observer.configure(actual, sha)
        result = originals["setup"](*args, **kwargs)
        return observer.attach(result, launcher)

    launcher.MasterConfig, launcher.init_ray, launcher.setup = master, owned_ray.start, setup
    sys.argv = [str(Path(plan["sources"]["nemo_rl"])/"examples"/("run_"+plan["profile"][3:]+".py")),
                "--config", plan["training"]["nemo_config"]]
    try:
        launcher.main()
        require(len(configured) == 1 and owned_ray.started, "official launcher did not complete setup")
        return observer.finish()
    finally:
        sys.argv = argv
        for key, value in originals.items():
            setattr(launcher, key, value)
        try:
            observer.close()
        finally:
            owned_ray.close()


def run(plan, output):
    """Keep the unfinished production path closed, even when called directly."""
    require(False, "NeMo runtime observer and supervisor wiring is not complete yet")
