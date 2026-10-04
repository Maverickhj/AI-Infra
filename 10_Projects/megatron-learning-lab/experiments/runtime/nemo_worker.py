"""Ray wrapper imported only by an authorized NeMo worker runtime."""
import inspect
import os
from pathlib import Path
import ray
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from .contracts import require
from .nemo_loading import CheckpointLoadScope
from .nemo_worker_observation import TrainObservationMixin


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class ObservedMegatronPolicyWorker(TrainObservationMixin, MegatronPolicyWorkerImpl):
    """Keep official initialization/training; observe actual conversion and load."""
    def __init__(self, *args, **kwargs):
        bound = inspect.signature(MegatronPolicyWorkerImpl.__init__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        config = bound.arguments["config"]
        require(bound.arguments["init_optimizer"] is True, "observed worker must own the actor optimizer")
        require(config.get("pretrained_checkpoint") is None, "initial actor requires the planned HF snapshot")
        output = Path(os.environ.get("MEGATRON_LAB_OUTPUT_PATH", ""))
        root = Path(os.environ.get("NRL_MEGATRON_CHECKPOINT_DIR", ""))
        require(output.is_absolute() and output.is_dir() and root.is_absolute()
                and root.resolve() == (output/"model-import").resolve(),
                "worker HF conversion cache is not bound to the run output")
        actor_cache = root/"actor"
        actor_cache.mkdir(parents=True, exist_ok=False)
        from nemo_rl.models.megatron import setup
        previous = os.environ["NRL_MEGATRON_CHECKPOINT_DIR"]
        os.environ["NRL_MEGATRON_CHECKPOINT_DIR"] = str(actor_cache)
        try:
            with CheckpointLoadScope(setup, snapshot=config["model_name"],
                    cache_root=actor_cache, reference_model=bound.arguments["init_reference_model"]) as loading:
                super().__init__(*args, **kwargs)
                self._lab_loading = loading.complete(self.megatron_cfg)
        finally:
            os.environ["NRL_MEGATRON_CHECKPOINT_DIR"] = previous
