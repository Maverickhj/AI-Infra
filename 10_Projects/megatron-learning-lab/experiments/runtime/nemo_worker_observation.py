"""Observation mixin for the official Megatron policy worker's single train call."""
from pathlib import Path
import json
from .adapters import LossTap
from .capture import ParameterSlice, callable_source
from .contracts import require
from .nemo_capture import LossRecordSink


def unwrap_training_model(model):
    from megatron.core.utils import unwrap_model
    return unwrap_model(model)


class TrainObservationMixin:
    def train(self, data, loss_fn, *args, **kwargs):
        require(isinstance(loss_fn, LossTap) and isinstance(loss_fn.sink, LossRecordSink),
                "observed worker requires the bounded loss capture sink")
        sink = loss_fn.sink
        if sink.role != "actor":
            return super().train(data, loss_fn, *args, **kwargs)
        require(not kwargs.get("eval_mode", False), "capture expects a real optimizer step")
        require(not args, "use explicit keyword training options")
        spec = sink.parameter
        require(isinstance(spec, dict) and set(spec) == {"name", "index"},
                "explicit actor parameter capture missing")
        model = unwrap_training_model(self.model)
        parameters = dict(model.named_parameters())
        require(spec["name"] in parameters, "selected actor parameter is absent")
        parameter = parameters[spec["name"]]
        index = tuple(spec["index"])
        # The inspected NeMo worker uses the same explicit Megatron master-group
        # interface. Unknown optimizer implementations must supply another adapter.
        from .bridge_entry import optimizer_parameter
        main = optimizer_parameter(self.optimizer, parameter)
        path = Path(sink.directory).parent/"actor-update.json"
        require(not path.exists(), "actor update capture already exists")
        before = float(main.detach()[index].item())
        observer = ParameterSlice(parameter, index)
        try:
            observer.start()
            result = super().train(data, loss_fn, **kwargs)
            selected = observer.finish(optimizer_executed=True)
            require(selected["backward_calls"] > 0 and main.grad is not None,
                    "actor backward/optimizer evidence is missing")
            after, gradient = float(main.detach()[index].item()), float(main.grad.detach()[index].item())
            require(gradient != 0 and after != before, "selected actor master parameter did not update")
            record = dict(parameter=dict(name=spec["name"], index=spec["index"], **selected),
                optimizer_parameter=dict(before=before, after=after, gradient=gradient,
                    dtype=str(main.dtype), changed=True),
                delegated_train=callable_source(super().train, "nemo_official_worker_train"),
                note="one selected scalar; model rounding and optimizer master state are separate")
            with path.open("x") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            return result
        finally:
            observer.close()
