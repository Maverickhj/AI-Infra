"""Ray wrapper imported only by an authorized NeMo worker runtime."""
import ray
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from .nemo_worker_observation import TrainObservationMixin


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class ObservedMegatronPolicyWorker(TrainObservationMixin, MegatronPolicyWorkerImpl):
    """Delegate all training to the official implementation, observing one scalar."""
