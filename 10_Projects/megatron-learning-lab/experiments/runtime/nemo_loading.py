"""Observe official HF conversion and checkpoint-load completion in one worker.

This scope changes no model construction, tensors, optimizer, or load result.
The production wrapper supplies a fresh, run-owned conversion directory.
"""
from __future__ import annotations
import json
from pathlib import Path
from .capture import callable_source, effective_config
from .contracts import require


class CheckpointLoadScope:
    def __init__(self, setup_module, *, snapshot, cache_root, reference_model):
        self.module = setup_module
        self.snapshot, self.cache_root = Path(snapshot).resolve(), Path(cache_root).resolve()
        self.reference_model = reference_model
        require(type(reference_model) is bool, "reference-model choice must be explicit")
        self.original_import = setup_module.import_model_from_hf_name
        self.original_load = setup_module.load_checkpoint
        self.imports, self.loads = [], []
        self.active = False

    def open(self):
        require(not self.active, "checkpoint observer already active")
        require(self.module.import_model_from_hf_name is self.original_import
                and self.module.load_checkpoint is self.original_load,
                "checkpoint functions changed before observation")
        self.module.import_model_from_hf_name = self.import_model
        self.module.load_checkpoint = self.load_checkpoint
        self.active = True
        return self

    def import_model(self, hf_model_name, pretrained_path, *args, **kwargs):
        require(self.active and not self.imports, "unexpected repeated HF conversion")
        source, destination = Path(hf_model_name).resolve(), Path(pretrained_path).resolve()
        require(source == self.snapshot, "HF conversion source differs from the planned checkpoint")
        require(destination.is_relative_to(self.cache_root) and destination != self.cache_root,
                "HF conversion escaped the run-owned cache")
        require(not destination.exists(), "HF conversion must start in a fresh run-owned directory")
        located = callable_source(self.original_import, "nemo_hf_conversion")
        result = self.original_import(hf_model_name, pretrained_path, *args, **kwargs)
        require(destination.is_dir(), "official conversion did not create its checkpoint directory")
        self.imports.append(dict(source_snapshot=str(source), converted_checkpoint=str(destination),
                                 completed=True, source=located))
        return result

    def load_checkpoint(self, state, model, optimizer, scheduler, *args, **kwargs):
        require(self.active and len(self.imports) == 1, "checkpoint loading preceded observed HF conversion")
        checkpoint = state.cfg.checkpoint
        require(Path(checkpoint.pretrained_checkpoint).resolve()
                == Path(self.imports[0]["converted_checkpoint"]),
                "loaded checkpoint differs from observed HF conversion")
        require(checkpoint.load is None, "unexpected resume checkpoint in initial RL profile")
        require(kwargs.get("skip_load_to_model_and_opt", False) is False,
                "checkpoint call skipped loading model weights")
        role = "reference" if optimizer is None else "actor"
        require(role not in {row["role"] for row in self.loads}, "duplicate checkpoint load role")
        require(role != "reference" or self.reference_model, "unexpected reference-model allocation")
        located = callable_source(self.original_load, "nemo_checkpoint_load")
        result = self.original_load(state, model, optimizer, scheduler, *args, **kwargs)
        self.loads.append(dict(role=role, completed=True,
                               converted_checkpoint=self.imports[0]["converted_checkpoint"], source=located))
        return result

    def complete(self, actual_config):
        roles = {"actor", "reference"} if self.reference_model else {"actor"}
        require(len(self.imports) == 1 and {row["role"] for row in self.loads} == roles,
                "actual model checkpoint loading is incomplete")
        config, sha = effective_config(actual_config)
        return dict(status="official_conversion_and_load_completed",
                    imports=list(self.imports), loads=list(self.loads),
                    effective_config_json=json.dumps(config,ensure_ascii=False,separators=(",",":"),allow_nan=False),
                    effective_config_sha256=sha,
                    limitation="call completion and source paths; numerical HF alignment belongs to R01/R02")

    def close(self):
        if self.active:
            self.module.import_model_from_hf_name = self.original_import
            self.module.load_checkpoint = self.original_load
            self.active = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
