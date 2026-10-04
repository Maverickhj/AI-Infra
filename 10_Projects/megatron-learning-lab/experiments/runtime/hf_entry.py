"""Teacher-forced HF reference using a local complete checkpoint and tokenizer.

Imported only by an authorized worker. No custom model, random-weight fallback,
remote code, tokenizer-template rewrite, or training loop is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

from .adapters import hf_sft_slice
from .capture import TraceCollector, callable_source, effective_config
from .contracts import require
from .plan import read_document
from .token_data import canonical_chat


def sha_file(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def snapshot_identity(snapshot):
    root=Path(snapshot)
    index=root/"model.safetensors.index.json"
    names={"config.json"}
    if index.is_file():
        mapping=json.loads(index.read_text())["weight_map"]
        names.update(mapping.values())
        names.add(index.name)
    else:
        names.add("model.safetensors")
    result=[]
    for name in sorted(names):
        require(Path(name).name == name,"weight manifest path must be a local basename")
        path=root/name
        require(path.is_file(),"checkpoint file missing: "+name)
        result.append(dict(name=name,bytes=path.stat().st_size,sha256=sha_file(path)))
    return result


def run(plan, output):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available() and torch.cuda.device_count()==1,
            "authorized one-GPU profile is not visible to HF worker")
    torch.cuda.set_device(0)
    torch.manual_seed(plan["training"]["seed"])
    started=datetime.now(timezone.utc).isoformat()
    snapshot=plan["model"]["snapshot"]
    tokenizer=AutoTokenizer.from_pretrained(plan["tokenizer"]["snapshot"],
        use_fast=True,trust_remote_code=False,local_files_only=True)
    config=AutoConfig.from_pretrained(snapshot,trust_remote_code=False,local_files_only=True)
    require(config.model_type in ("qwen3","qwen2"),"unknown HF family adapter")
    data,tokenization=canonical_chat(tokenizer,read_document(plan["data"]["path"]),
        mask_mode=plan["data"]["mask_mode"],forward_sequence_length=plan["training"]["sequence_length"],
        vocab_size=config.vocab_size)
    weight_files=snapshot_identity(snapshot)
    dtype=getattr(torch,plan["training"]["dtype"])
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,dtype=dtype,attn_implementation="eager")
    model.to(device="cuda:0")
    model.eval()
    parameter_name=plan["capture"]["parameter_name"]
    parameters=dict(model.named_parameters())
    require(parameter_name in parameters,"explicit HF parameter name is absent: "+parameter_name)
    parameter=parameters[parameter_name]
    index=tuple(plan["capture"]["parameter_index"])
    require(len(index)==parameter.ndim and all(0<=j<n for j,n in zip(index,parameter.shape)),
            "HF parameter slice is out of range")
    parameter_value=float(parameter.detach()[index].item())
    # Known Qwen HF final norm output is BSH. Only a few scalar coordinates
    # are copied; hidden_states for the whole decoder are never retained.
    norm=model.get_submodule("model.norm")
    active=[j for j,mask in enumerate(data["loss_mask"][0]) if mask][:2]
    activation={}
    def observe(module,args,result):
        require(isinstance(result,torch.Tensor) and result.ndim==3
                and tuple(result.shape[:2])==(1,len(data["input_ids"][0])),
                "unknown HF final-norm layout")
        activation.update(module="model.norm",layout="BSH",shape=list(result.shape),
                          canonical_indices=[[0,j,0] for j in active],
                          values=[float(result.detach()[0,j,0].item()) for j in active])
    handle=norm.register_forward_hook(observe)
    ids=torch.tensor(data["input_ids"],dtype=torch.long,device="cuda:0")
    positions=torch.tensor(data["position_ids"],dtype=torch.long,device="cuda:0")
    # Match Bridge causal attention: right padding is ignored by the loss,
    # while diagnostic padded positions still attend their causal context.
    attention=torch.ones_like(ids)
    try:
        with torch.inference_mode():
            result=model(input_ids=ids,position_ids=positions,attention_mask=attention,
                         use_cache=False,return_dict=True)
            measured=hf_sft_slice(result,data,{"vocab_size":config.vocab_size})
    finally:
        handle.remove()
    require(activation,"HF final-norm hook was not called")
    measured["activation_slice"]=activation
    measured["parameter_slice"]=dict(name=parameter_name,index=list(index),before=parameter_value,
                                     after=parameter_value,gradient=None,optimizer_executed=False)
    actual_config,config_hash=effective_config(config)
    completed=datetime.now(timezone.utc).isoformat()
    manifest=dict(adapter="hf_next_token_v1",dtype=plan["training"]["dtype"],layout="BS",
        backend="transformers",evidence_kind="hf_runtime",
        model=dict(id=plan["model"]["id"],revision=plan["model"]["revision"],
                   weights_origin="hf_checkpoint",architecture_origin="hf_config",
                   loaded_files=weight_files),
        tokenizer=dict(id=plan["tokenizer"]["id"],revision=plan["tokenizer"]["revision"],
                       chat_template_sha256=tokenization["chat_template_sha256"]),
        parallel=dict(tp=1,pp=1,dp=1,cp=1,ep=1,world_size=1,groups_origin="in_memory_reference"),
        software=dict(python=platform.python_version(),torch=torch.__version__,
                      transformers=importlib.metadata.version("transformers")),
        runtime_sources=[callable_source(model.forward,"hf_forward"),
                         callable_source(norm.forward,"hf_final_norm"),
                         callable_source(tokenizer.apply_chat_template,"hf_chat_template"),
                         callable_source(hf_sft_slice,"adapter")],
        execution=dict(status="executed",synthetic=False,command=list(sys.argv),
                       started_at=started,completed_at=completed),
        capture=dict(scope="selected_slices",timing="not_measured",
                     unobserved=["full activations","full logits","gradients","optimizer update"]),
        limitations=["Teacher-forced HF reference; no optimizer step.",
                     "Local snapshot revision is supplied metadata; loaded file hashes record actual bytes.",
                     "No Bridge or rollout-engine compatibility conclusion."])
    collector=TraceCollector(output/"traces",max_records=plan["capture"]["max_records"])
    trace=collector.sft(plan["run_id"],manifest,
        dict(vocab_size=config.vocab_size,mask_mode=plan["data"]["mask_mode"],
             effective_config_json=json.dumps(actual_config,ensure_ascii=False,separators=(",",":"),allow_nan=False),
             effective_config_sha256=config_hash,
             tolerances=plan["tolerances"]),data,measured,provenance="reference")
    (output/"tokenization.json").write_text(json.dumps(tokenization,ensure_ascii=False,indent=2)+"\n")
    (output/"result.json").write_text(json.dumps(dict(status="completed_reference",trace=trace,
        optimizer_steps=0,forward_evaluations=1),ensure_ascii=False,indent=2)+"\n")
    return trace
