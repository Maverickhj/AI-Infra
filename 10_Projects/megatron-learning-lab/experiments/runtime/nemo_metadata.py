"""Runtime identity from the actual local tokenizer, checkpoint bytes and modules."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .capture import callable_source
from .contracts import integer, require
from .hf_entry import sha_file, snapshot_identity
from .plan import read_document



def inspect_metadata(plan):
    """Read bounded local metadata before the official trust_remote_code loader."""
    models = {}
    for role in ("model",) + (("critic",) if plan["profile"] == "rl_ppo" else ()):
        config = read_document(Path(plan[role]["snapshot"])/"config.json")
        require(config.get("model_type") in ("qwen2", "qwen3"),
                "unknown local NeMo model family: " + role)
        require(not config.get("auto_map"), "remote-code mapping is outside the local NeMo profile: " + role)
        require(integer(config.get("vocab_size")) and config["vocab_size"] > 1,
                "invalid local NeMo vocabulary: " + role)
        models[role] = config
    tokenizer = read_document(Path(plan["tokenizer"]["snapshot"])/"tokenizer_config.json")
    require(not tokenizer.get("auto_map"), "remote-code tokenizer mapping is outside the local NeMo profile")
    require(tokenizer.get("tokenizer_class") in ("Qwen2Tokenizer", "Qwen2TokenizerFast"),
            "unknown local NeMo tokenizer class")
    template = Path(plan["tokenizer"]["snapshot"])/"chat_template.jinja"
    require(template.is_file() or isinstance(tokenizer.get("chat_template"), str)
            and bool(tokenizer["chat_template"]), "local NeMo chat template is missing")
    if "critic" in models:
        require(models["critic"]["vocab_size"] == models["model"]["vocab_size"],
                "critic vocabulary differs from actor")
    return models


def tokenizer_identity(snapshot):
    root = Path(snapshot)
    names = ["tokenizer_config.json", "tokenizer.json"]
    if (root/"chat_template.jinja").is_file():
        names.append("chat_template.jinja")
    return [dict(name=name, bytes=(root/name).stat().st_size, sha256=sha_file(root/name))
            for name in names]

def manifest(plan, tokenizer, *, sources, frozen):
    config=inspect_metadata(plan)["model"]
    require(getattr(tokenizer,"is_fast",False) is True, "actual tokenizer is not fast")
    require(Path(tokenizer.name_or_path).resolve()==Path(plan["tokenizer"]["snapshot"]).resolve(),
            "actual tokenizer source differs from plan")
    template=tokenizer.get_chat_template()
    require(isinstance(template,str) and template, "actual tokenizer chat template is missing")
    software={"python":platform.python_version()}
    for package in ("torch","ray","transformers","megatron-core","megatron-bridge","nemo-rl","vllm"):
        try:software[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:software[package]="distribution metadata unavailable"
    result=dict(adapter="nemo_full_token_v1",dtype=plan["training"]["dtype"],layout="BS",
        backend="nemo-rl",evidence_kind="rl_runtime",
        model=dict(id=plan["model"]["id"],revision=plan["model"]["revision"],
                   weights_origin="hf_checkpoint",architecture_origin="hf_config",
                   loaded_files=snapshot_identity(plan["model"]["snapshot"])),
        tokenizer=dict(id=plan["tokenizer"]["id"],revision=plan["tokenizer"]["revision"],
                       chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                       loaded_files=tokenizer_identity(plan["tokenizer"]["snapshot"])),
        software=software,runtime_sources=list(sources)+[callable_source(tokenizer.apply_chat_template,"nemo_chat_template")],
        execution=dict(status="executed",synthetic=False,command=list(sys.argv)),
        capture=dict(scope="selected_slices",timing="not_measured",
                     unobserved=["full logits","full activations","end-to-end refit weight hash"]),
        limitations=["Single-rank synchronous profile; no production performance measurement.",
                     "Source paths and call completion are not numerical checkpoint equivalence.",
                     "Refit completion has no end-to-end weight hash acknowledgment."],
        frozen_config=frozen)
    if plan["profile"]=="rl_ppo":
        result["critic"]=dict(**plan["critic"],loaded_files=snapshot_identity(plan["critic"]["snapshot"]),
                              head="official regression value head; separate critic resource grant")
    return result,config["vocab_size"]
