"""Small evidence collectors which delegate to real framework entry points.

Only a caller that has already passed resource/profile checks should connect
these hooks to a runtime. The collector validates provenance but cannot attest
that a caller's manifest is truthful.
"""
from __future__ import annotations
import hashlib
from functools import partial
import inspect
import json
from pathlib import Path
from .contracts import MAX_BYTES, read_trace, require
from .adapters import bridge_sft_slice


def callable_source(fn, component):
    """Locate an actually loaded Python callable; do not import a guessed path."""
    while isinstance(fn, partial):
        fn = fn.func
    if not inspect.isfunction(fn) and not inspect.ismethod(fn):
        fn = fn.__call__
    fn = inspect.unwrap(fn)
    path = inspect.getsourcefile(fn)
    require(path is not None, "runtime callable has no Python source; explicit native mapping required")
    lines, line = inspect.getsourcelines(fn)
    resolved = Path(path).resolve()
    return dict(component=component, path=str(resolved), symbol=fn.__qualname__,
                sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
                line=line, end_line=line+len(lines)-1,
                evidence="loaded_python_callable", module=fn.__module__)


def effective_config(config):
    """Serialize actual config values without repr addresses or executing targets."""
    import dataclasses
    import math
    from enum import Enum
    def walk(value, depth=0):
        require(depth <= 32, "config nesting exceeds limit")
        if type(value) is float and not math.isfinite(value):
            require(not math.isnan(value), "NaN is not an effective config value")
            # Explicit metadata tags preserve upstream unbounded thresholds.
            # This path never serializes measurements, which must remain finite.
            return {"float_sentinel": "+inf" if value > 0 else "-inf"}
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return dict(enum_type=type(value).__module__+"."+type(value).__qualname__,
                        name=value.name, value=walk(value.value,depth+1))
        if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
            return str(value)
        if inspect.isfunction(value) or inspect.ismethod(value):
            return dict(callable=callable_source(value,"effective_config_callable"))
        if inspect.isclass(value):
            return dict(class_name=value.__module__+"."+value.__qualname__)
        if isinstance(value,(list,tuple)):
            return [walk(x,depth+1) for x in value]
        if isinstance(value,dict):
            require(all(isinstance(k,str) for k in value), "non-string effective config key")
            return {k:walk(v,depth+1) for k,v in value.items()}
        if dataclasses.is_dataclass(value):
            return {f.name:walk(getattr(value,f.name),depth+1)
                    for f in dataclasses.fields(value) if not f.name.startswith("_")}
        raise ValueError("unsupported effective config value: "+type(value).__name__)
    # Use the framework's real resolved container when available, never a
    # handwritten approximation of all of its defaults.
    raw=config.to_dict() if callable(getattr(config,"to_dict",None)) else config
    result=walk(raw)
    text=json.dumps(result,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    require(len(text.encode("utf-8"))<=MAX_BYTES,"effective config exceeds capture limit")
    return result, hashlib.sha256(text.encode("utf-8")).hexdigest()


class TraceCollector:
    """Write at most a small declared number of validated traces, never overwrite."""
    def __init__(self, output_dir, *, max_records=16):
        require(type(max_records) is int and 1<=max_records<=64,"invalid capture record limit")
        self.output_dir=Path(output_dir)
        self.max_records=max_records
        self.records=[]

    def write(self, trace):
        require(len(self.records)<self.max_records,"capture record limit exceeded")
        text=json.dumps(trace,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
        read_trace(text)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        # File names do not come from untrusted run_id or source paths.
        path=self.output_dir/f"trace-{len(self.records):04d}.json"
        with path.open("x",encoding="utf-8") as stream:
            stream.write(text)
        record=dict(path=str(path),sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    run_id=trace["run_id"],provenance=trace["provenance"],trust="imported_claim")
        self.records.append(record)
        return record

    def sft(self, run_id, manifest, config, data, measurements, *, provenance):
        require(provenance in ("reference","observed_bridge"),"invalid SFT capture provenance")
        trace=make_trace(run_id,"sft",provenance,manifest,config,data,measurements)
        return self.write(trace)


def make_trace(run_id, task, provenance, manifest, config, data, measurements):
    """One constructor for supplied runtime identities and actual collected slices."""
    from .contracts import SCHEMA, digest_text
    cfg=json.dumps(config,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    inp=json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    trace=dict(schema=SCHEMA,schema_version=1,run_id=run_id,task=task,provenance=provenance,
               manifest=manifest,config_json=cfg,config_sha256=digest_text(cfg),
               input_json=inp,input_sha256=digest_text(inp),
               measurements=measurements)
    read_trace(json.dumps(trace,ensure_ascii=False,allow_nan=False))
    return trace


class BridgeForwardTap:
    """Wrap official gpt_step for the explicit one-rank, unpacked capture profile."""
    def __init__(self, delegate, sink, *, vocab_size, mapping="bridge_bsh_v1"):
        self.delegate,self.sink=delegate,sink
        self.vocab_size,self.mapping=vocab_size,mapping

    def __call__(self, state, data_iterator, model, return_schedule_plan=False):
        require(not return_schedule_plan,"schedule-plan capture requires explicit adapter")
        seen=[]
        class CaptureIterator:
            def __iter__(self):return self
            def __next__(self):
                batch=next(data_iterator)
                require(not seen,"forward consumed multiple batches; unknown mapping")
                seen.append(batch)
                return batch
        result=self.delegate(state,CaptureIterator(),model,return_schedule_plan=False)
        require(isinstance(result,tuple) and len(result)==2,"unknown Bridge forward return")
        require(len(seen)==1,"forward did not consume its expected batch")
        data,measurements=bridge_sft_slice(seen[0],result[0],vocab_size=self.vocab_size,
                                          mapping=self.mapping,output_kind="token_loss")
        self.sink(data,measurements,state,model)
        return result


class ParameterSlice:
    """Observe one scalar's backward contributions and before/after update values."""
    def __init__(self, parameter, index):
        import torch
        require(isinstance(parameter,torch.Tensor) and parameter.requires_grad,"parameter must be trainable")
        require(len(index)==parameter.ndim and all(type(v)is int and 0<=v<n for v,n in zip(index,parameter.shape)),
                "invalid parameter slice")
        self.parameter,self.index=parameter,tuple(index)
        self.gradient=0.0
        self.backward_calls=0
        self.before=None
        self.handle=parameter.register_hook(self._hook)

    def _hook(self, gradient):
        self.gradient+=float(gradient.detach()[self.index].item())
        self.backward_calls+=1
        # Returning None preserves the original gradient object/value.

    def start(self):
        self.before=float(self.parameter.detach()[self.index].item())
        self.gradient=0.0
        self.backward_calls=0

    def finish(self, *, optimizer_executed):
        require(self.before is not None,"parameter capture was not started")
        after=float(self.parameter.detach()[self.index].item())
        return dict(before=self.before,gradient=self.gradient,after=after,
                    backward_calls=self.backward_calls,optimizer_executed=bool(optimizer_executed),
                    changed=after!=self.before,
                    limitation="one parameter scalar; hook sum before any later optimizer scaling")

    def close(self):
        self.handle.remove()
