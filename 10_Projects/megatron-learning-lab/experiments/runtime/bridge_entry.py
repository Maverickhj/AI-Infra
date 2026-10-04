"""Thin Bridge entry: official provider, weight import, finetune, checkpoint/export.

No production trainer or optimizer is reimplemented. Configuration construction
can be probed separately with CUDA/network blocked; training requires the worker.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .capture import BridgeForwardTap, ParameterSlice, TraceCollector, callable_source, effective_config
from .contracts import require
from .hf_entry import snapshot_identity
from .plan import read_document
from .token_data import canonical_chat


def build_config(plan, *, hf_config=None):
    from transformers import AutoConfig
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.training.config import (
        CheckpointConfig, ConfigContainer, DistributedDataParallelConfig, LoggerConfig,
        OptimizerConfig, RNGConfig, SchedulerConfig, TokenizerConfig, TrainingConfig,
        ValidationConfig,
    )
    from .bridge_dataset import CanonicalDatasetProvider
    path = plan["model"]["snapshot"]
    if hf_config is None:
        hf_config = AutoConfig.from_pretrained(path, trust_remote_code=False, local_files_only=True)
    require(hf_config.model_type in ("qwen3", "qwen2"), "unknown Bridge family adapter")
    actual_source = callable_source(AutoBridge.from_hf_config, "AutoBridge.from_hf_config")
    require(Path(actual_source["path"]).is_relative_to(Path(plan["sources"]["bridge"]).resolve()),
            "loaded Bridge differs from inspected source root")
    bridge = AutoBridge.from_hf_config(hf_config)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.perform_initialization = False
    journal = []
    if plan["training"]["resume_from"] is None:
        def load_actual_weights(models):
            # Calling the official conversion here produces an observable
            # completion event. Config-only provider creation is not loading.
            bridge.load_hf_weights(models, hf_path=path)
            journal.append(dict(action="AutoBridge.load_hf_weights", status="completed",
                                source=path, completed_at=datetime.now(timezone.utc).isoformat()))
            return models
        provider.register_pre_wrap_hook(load_actual_weights)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.sequence_parallel = False
    provider.seq_length = plan["training"]["sequence_length"]
    provider.calculate_per_token_loss = True
    provider.gradient_accumulation_fusion = False
    provider.hidden_dropout = 0.0
    provider.attention_dropout = 0.0
    output = Path(plan["execution"]["output_path"])
    train = plan["training"]
    resume = train["resume_from"]
    cfg = ConfigContainer(
        model=provider,
        train=TrainingConfig(train_iters=train["steps"], global_batch_size=1, micro_batch_size=1),
        optimizer=OptimizerConfig(optimizer="adam", lr=train["learning_rate"], min_lr=0.0,
                                  use_distributed_optimizer=False, weight_decay=0.0),
        ddp=DistributedDataParallelConfig(use_distributed_optimizer=False,
            overlap_grad_reduce=False, overlap_param_gather=False, grad_reduce_in_fp32=True),
        scheduler=SchedulerConfig(lr_decay_style="constant", lr_warmup_iters=0,
                                  lr_decay_iters=train["steps"]),
        dataset=CanonicalDatasetProvider(seq_length=train["sequence_length"],
            canonical_path=str(output/"canonical-input.json"), vocab_size=hf_config.vocab_size,
            dataloader_type="single", num_workers=0, persistent_workers=False,
            trust_remote_code=False),
        logger=LoggerConfig(log_interval=1, tensorboard_dir=None),
        tokenizer=TokenizerConfig(tokenizer_type="HuggingFaceTokenizer",
                                  tokenizer_model=plan["tokenizer"]["snapshot"]),
        checkpoint=CheckpointConfig(save=str(output/"checkpoints"),
            load=resume or str(output/"checkpoints"), pretrained_checkpoint=None,
            save_interval=1, save_optim=True, save_rng=True, load_optim=True, load_rng=True,
            exit_on_missing_checkpoint=resume is not None, async_save=False),
        validation=ValidationConfig(eval_iters=0, eval_interval=0),
        rng=RNGConfig(seed=train["seed"]),
        mixed_precision="bf16_mixed" if train["dtype"] == "bfloat16" else None,
    )
    return cfg, bridge, journal


def optimizer_parameter(optimizer, parameter):
    """Explicit Float16Optimizer/ChainedOptimizer mapping, not name guessing."""
    opts = getattr(optimizer, "chained_optimizers", None)
    opts = opts if opts is not None else [optimizer]
    found = []
    for opt in opts:
        model_groups = getattr(opt, "float16_groups", [])
        main_groups = getattr(opt, "fp32_from_float16_groups", [])
        require(len(model_groups) == len(main_groups), "unknown optimizer master group mapping")
        for model_group, main_group in zip(model_groups, main_groups):
            require(len(model_group) == len(main_group), "unknown optimizer master parameter mapping")
            found.extend(main for model, main in zip(model_group, main_group) if model is parameter)
        if any(p is parameter for p in opt.get_parameters()):
            found.append(parameter)
    require(len(found) == 1, "selected optimizer parameter mapping missing or ambiguous")
    require(tuple(found[0].shape) == tuple(parameter.shape), "master parameter shape differs")
    return found[0]


def run(plan, output):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.training.checkpointing import checkpoint_exists
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "authorized one-GPU profile is not visible to Bridge")
    torch.cuda.set_device(0)
    started = datetime.now(timezone.utc).isoformat()
    resume = plan["training"]["resume_from"]
    if resume is not None:
        require(checkpoint_exists(resume), "resume checkpoint is absent or incomplete")
    hf_config = AutoConfig.from_pretrained(plan["model"]["snapshot"],
                                          trust_remote_code=False, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer"]["snapshot"],
        use_fast=True, trust_remote_code=False, local_files_only=True)
    data, tokenization = canonical_chat(tokenizer, read_document(plan["data"]["path"]),
        mask_mode=plan["data"]["mask_mode"],
        forward_sequence_length=plan["training"]["sequence_length"], vocab_size=hf_config.vocab_size)
    (output/"canonical-input.json").write_text(json.dumps(data, separators=(",", ":")))
    (output/"tokenization.json").write_text(json.dumps(tokenization, ensure_ascii=False, indent=2)+"\n")
    weights = snapshot_identity(plan["model"]["snapshot"])
    cfg, bridge, journal = build_config(plan, hf_config=hf_config)
    collector = TraceCollector(output/"traces", max_records=plan["capture"]["max_records"])

    class Observer(Callback):
        def __init__(self):
            self.pending = None
            self.completed = 0
            self.activation = {}
            self.hook = None
            self.parameter_observer = None
            self.checkpoints = []
            self.exported = False

        def on_train_start(self, context):
            require(torch.distributed.is_initialized() and torch.distributed.get_world_size() == 1,
                    "Bridge runtime process group differs from the one-rank plan")
            self.start_step = int(context.state.train_state.step)
            require(self.start_step < plan["training"]["steps"], "resume has no remaining requested iterations")
            require((resume is None and self.start_step == 0 and len(journal) == 1)
                    or (resume is not None and self.start_step > 0),
                    "real HF weight loading or checkpoint restore did not complete")
            models = unwrap_model(context.model)
            require(len(models) == 1, "unexpected virtual pipeline chunks")
            self.model = models[0]
            name = plan["capture"]["parameter_name"]
            parameters = dict(self.model.named_parameters())
            require(name in parameters, "explicit Bridge parameter is absent: " + name)
            self.parameter = parameters[name]
            self.index = tuple(plan["capture"]["parameter_index"])
            self.parameter_observer = ParameterSlice(self.parameter, self.index)
            self.main_parameter = optimizer_parameter(context.optimizer, self.parameter)
            norm = self.model.get_submodule("decoder.final_layernorm")
            active = [j for j, mask in enumerate(data["loss_mask"][0][:-1]) if mask][:2]
            def observe(module, args, result):
                require(isinstance(result, torch.Tensor) and result.ndim == 3
                        and tuple(result.shape[:2]) == (plan["training"]["sequence_length"], 1),
                        "unknown Bridge final-norm SBH mapping")
                self.activation = dict(module="decoder.final_layernorm", layout="SBH",
                    shape=list(result.shape), canonical_indices=[[0, j, 0] for j in active],
                    values=[float(result.detach()[j, 0, 0].item()) for j in active])
            self.hook = norm.register_forward_hook(observe)
            self.actual_config, self.config_hash = effective_config(context.state.cfg)
            self.config_text = json.dumps(self.actual_config, ensure_ascii=False,
                                          separators=(",", ":"), allow_nan=False)
            (output/"effective-config.json").write_text(self.config_text)
            self.sources = [callable_source(forward_step, "bridge_forward"),
                            callable_source(finetune, "bridge_finetune"),
                            callable_source(bridge.load_hf_weights, "bridge_weight_import"),
                            callable_source(bridge.save_hf_pretrained, "bridge_export"),
                            callable_source(norm.forward, "bridge_final_norm"),
                            callable_source(type(context.state.cfg).to_dict, "resolved_config"),
                            callable_source(type(context.optimizer).step, "bridge_optimizer_step")]

        def on_train_step_start(self, context):
            require(self.pending is None, "previous forward capture was not consumed")
            self.activation = {}
            self.parameter_observer.start()
            self.main_before = float(self.main_parameter.detach()[self.index].item())

        def sink(self, actual_data, measurements, state, model):
            require(actual_data == data, "Bridge consumed a different canonical batch")
            require(self.pending is None, "unknown multi-microbatch/rerun capture branch")
            self.pending = measurements

        def on_train_step_end(self, context):
            require(not context.skipped_iter and self.pending is not None and self.activation,
                    "step skipped or forward/activation evidence missing")
            measured = self.pending
            self.pending = None
            selected = self.parameter_observer.finish(optimizer_executed=True)
            require(selected["backward_calls"] > 0, "parameter backward hook did not execute")
            grad = self.main_parameter.grad
            require(grad is not None, "unknown optimizer gradient lifecycle")
            main_after = float(self.main_parameter.detach()[self.index].item())
            main_grad = float(grad.detach()[self.index].item())
            require(main_grad != 0 and main_after != self.main_before,
                    "selected optimizer parameter did not receive a nonzero update")
            measured["parameter_slice"] = dict(name=plan["capture"]["parameter_name"],
                index=list(self.index), **selected)
            measured["optimizer_parameter_slice"] = dict(before=self.main_before, after=main_after,
                gradient=main_grad, dtype=str(self.main_parameter.dtype), changed=main_after != self.main_before,
                mapping="Float16Optimizer master parameter or exact fp32 optimizer parameter")
            measured["activation_slice"] = self.activation
            measured["step_before"] = int(context.state.train_state.step)
            measured["step_after"] = measured["step_before"] + 1
            group = dict(tp=parallel_state.get_tensor_model_parallel_world_size(),
                         pp=parallel_state.get_pipeline_model_parallel_world_size(),
                         dp=parallel_state.get_data_parallel_world_size(),
                         cp=parallel_state.get_context_parallel_world_size(),
                         ep=parallel_state.get_expert_model_parallel_world_size(),
                         world_size=torch.distributed.get_world_size(), groups_origin="runtime")
            require(all(group[k] == 1 for k in ("tp", "pp", "dp", "cp", "ep", "world_size")),
                    "runtime parallel groups differ from plan")
            manifest = dict(adapter="bridge_bsh_v1", layout="BS", backend="megatron-bridge",
                dtype=plan["training"]["dtype"], evidence_kind="bridge_runtime",
                model=dict(id=plan["model"]["id"], revision=plan["model"]["revision"],
                    weights_origin="training_checkpoint" if resume else "hf_checkpoint",
                    architecture_origin="hf_config", initial_hf_files=weights),
                tokenizer=dict(id=plan["tokenizer"]["id"], revision=plan["tokenizer"]["revision"],
                    chat_template_sha256=tokenization["chat_template_sha256"]),
                parallel=group, software=dict(python=platform.python_version(), torch=torch.__version__,
                    bridge=importlib.metadata.version("megatron-bridge"),
                    core=importlib.metadata.version("megatron-core")),
                runtime_sources=self.sources,
                execution=dict(status="executed", synthetic=False, command=list(sys.argv),
                    started_at=started, completed_at=datetime.now(timezone.utc).isoformat()),
                capture=dict(scope="selected_slices", timing="not_measured",
                    unobserved=["full activations", "full logits", "all parameter gradients"]),
                limitations=["This trace records one real training step, not full R01 acceptance.",
                    "FP32 optimizer master updates and rounded model weights are separate measurements.",
                    "Resume equivalence and exported HF numerics require independent comparison."])
            collector.sft(plan["run_id"] + "-step-" + str(measured["step_after"]), manifest,
                dict(vocab_size=hf_config.vocab_size, mask_mode=plan["data"]["mask_mode"],
                     effective_config_json=self.config_text, effective_config_sha256=self.config_hash,
                     tolerances=plan["tolerances"]), data, measured,
                provenance="observed_bridge")
            self.completed += 1

        def on_checkpoint_save(self, context):
            self.checkpoints.append(int(context.state.train_state.step))

        def on_train_end(self, context):
            require(int(context.state.train_state.step) == plan["training"]["steps"],
                    "training stopped before requested total iterations")
            require(self.completed == plan["training"]["steps"] - self.start_step,
                    "training capture count differs from actual iterations")
            require(checkpoint_exists(output/"checkpoints"), "official checkpoint was not saved")
            bridge.save_hf_pretrained(context.model, output/"hf-export",
                source_path=plan["model"]["snapshot"], show_progress=False, strict=True)
            self.exported = True

        def close(self):
            if self.hook is not None:
                self.hook.remove()
            if self.parameter_observer is not None:
                self.parameter_observer.close()

    observer = Observer()
    tap = BridgeForwardTap(forward_step, observer.sink, vocab_size=hf_config.vocab_size)
    try:
        finetune(cfg, tap, callbacks=[observer])
        require(observer.exported and observer.completed > 0, "training/export did not finish")
        exported = snapshot_identity(output/"hf-export")
        result = dict(status="completed_bridge_entry", steps=observer.completed,
            total_iterations=plan["training"]["steps"], start_iteration=observer.start_step,
            weight_load_events=journal, resumed_from=resume, traces=collector.records,
            checkpoints=str(output/"checkpoints"), checkpoint_events=observer.checkpoints,
            hf_export=dict(path=str(output/"hf-export"), files=exported),
            limitation="Real trace/save/export collection; independent R01 equivalence checks remain required")
        (output/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        return dict(steps=observer.completed, trace_count=len(collector.records))
    finally:
        observer.close()
