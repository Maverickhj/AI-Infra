"""Isolated configuration construction with model/GPU/network entry points blocked."""
from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import socket
import sys

from .capture import callable_source, effective_config
from .contracts import require
from .plan import digest, read_document, validate_plan


def probe_bridge(plan_path):
    """Run only in a disposable CLI process: guards intentionally remain active."""
    plan = validate_plan(read_document(plan_path))
    require(plan["profile"] == "bridge_sft", "Bridge configuration probe requires bridge_sft")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    blocked = []
    def forbidden(*args, **kwargs):
        raise AssertionError("configuration-only probe forbids network/model/training initialization")
    def cuda_unavailable(*args, **kwargs):
        blocked.append("CUDA initialization blocked before calling real initializer")
        raise RuntimeError("configuration-only probe: CUDA unavailable")
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        torch.cuda.init = cuda_unavailable
        torch.cuda._lazy_init = cuda_unavailable
        torch.distributed.init_process_group = forbidden
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge
        from megatron.bridge.models.model_provider import ModelProviderMixin
        from transformers import AutoModelForCausalLM
        # No model can be constructed or weights loaded through the inspected
        # entry paths, even if a later config-builder edit calls one by mistake.
        source = callable_source(AutoBridge.from_hf_config, "AutoBridge.from_hf_config")
        source_root = Path(plan["sources"]["bridge"]).resolve()
        require(Path(source["path"]).is_relative_to(source_root), "loaded Bridge differs from inspected source root")
        ModelProviderMixin.provide_distributed_model = forbidden
        AutoModelForCausalLM.from_pretrained = forbidden
        AutoBridge.load_hf_weights = forbidden
        from .bridge_entry import build_config
        cfg, _, events = build_config(plan)
        actual, sha = effective_config(cfg)
        require(events == [] and not torch.cuda.is_initialized(), "configuration probe crossed its boundary")
    return dict(status="configuration_only", plan_sha256=digest(plan),
                phase="constructed_before_framework_finalize",
                constructed_config_json=json.dumps(actual,ensure_ascii=False,separators=(",",":"),allow_nan=False),
                constructed_config_sha256=sha, runtime_effective_config="not_run",
                source=source, blocked_cuda_initialization_attempts=len(blocked),
                model_constructed=False, weights_loaded=False, cuda_initialized=False,
                limitation="Finalized runtime values are captured at official on_train_start; this is not runtime compatibility")
