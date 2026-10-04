"""Resolve inspected NeMo-style YAML inheritance without importing its launcher.

Only dictionary references and pure mul/div/max resolvers are supported. No
environment resolver, arbitrary callable, Python tag, network, model, or cluster
is invoked. Hydra/OmegaConf are configuration dependencies, not training imports.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from .contracts import MAX_BYTES, finite_tree, require


def _resolve_nemo_config(path, *, source_root, overrides=()):
    from hydra._internal.config_loader_impl import ConfigLoaderImpl
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import DictConfig, ListConfig, OmegaConf
    root=Path(source_root).resolve()
    sources=[]
    seen=set()

    def guarded_interpolations(value):
        if isinstance(value,str):
            # Explicitly reject computed key names, relative refs, and resolvers
            # which can access environment or perform work. Keep this narrow.
            require(not re.search(r"\$\{(?![A-Za-z_][\w.]*(?:[:}]))",value),
                    "unknown interpolation syntax")
            for resolver in re.findall(r"\$\{([A-Za-z_][\w.]*):",value):
                require(resolver in ("mul","div","max"),"unsupported config resolver: "+resolver)
            return re.sub(r"\$\{(mul|div|max):",r"${lab_trace_\1:",value)
        if isinstance(value,list):
            return [guarded_interpolations(v) for v in value]
        if isinstance(value,dict):
            return {k:guarded_interpolations(v) for k,v in value.items()}
        return value

    def merge(base, override):
        # Match the inspected official helper: _override_ applies to immediate
        # dictionary sections in each inherited file.
        for key in list(override.keys()):
            section=override[key]
            if isinstance(section,DictConfig) and section.get("_override_",False):
                section.pop("_override_")
                if key in base:base.pop(key)
        return OmegaConf.merge(base,override)

    def load(filename, stack=()):
        p=Path(filename).resolve()
        require(p.is_relative_to(root),"config inheritance escapes source root")
        require(p not in stack and len(stack)<16,"cyclic/deep config inheritance")
        raw=p.read_bytes()
        require(len(raw)<=MAX_BYTES,"config exceeds 1 MiB")
        config=OmegaConf.create(raw.decode("utf-8"))
        require(isinstance(config,DictConfig),"config must be a dictionary")
        # Reject unsafe resolvers before any access that might resolve them.
        unparsed=OmegaConf.to_container(config,resolve=False)
        finite_tree(unparsed)
        config=OmegaConf.create(guarded_interpolations(unparsed))
        if p not in seen:
            sources.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
            seen.add(p)
        if "defaults" in config:
            defaults=config.pop("defaults")
            if isinstance(defaults,str):defaults=[defaults]
            require(isinstance(defaults,(list,ListConfig)) and all(isinstance(v,str) and "${" not in v for v in defaults),
                    "unknown inheritance defaults mapping")
            base=OmegaConf.create({})
            for parent in defaults:
                base=merge(base,load(p.parent/parent,stack+(p,)))
            config=merge(base,config)
        return config

    # Private names isolate this pure arithmetic from previously registered
    # upstream resolvers with the same short names.
    for name,fn in (("mul",lambda a,b:a*b),("div",lambda a,b:a/b),("max",lambda a,b:max(a,b))):
        OmegaConf.register_new_resolver("lab_trace_"+name,fn,replace=True)
    cfg=load(path)
    safe_overrides=[guarded_interpolations(value) for value in overrides]
    OmegaConf.set_struct(cfg,True)
    parser=OverridesParser.create()
    parsed=parser.parse_overrides(overrides=safe_overrides)
    ConfigLoaderImpl._apply_overrides_to_config(overrides=parsed,cfg=cfg)
    actual=OmegaConf.to_container(cfg,resolve=True)
    finite_tree(actual)
    encoded=json.dumps(actual,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    require(len(encoded.encode("utf-8"))<=MAX_BYTES,"resolved config exceeds capture limit")
    return dict(config=actual,config_json=encoded,config_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                files=sources,scope="resolved configuration values; runtime behavior not checked")


def resolve_nemo_config(path, *, source_root, overrides=()):
    try:
        return _resolve_nemo_config(path,source_root=source_root,overrides=overrides)
    except (ValueError,OSError):
        raise
    except Exception as exc:
        raise ValueError("config resolution failed ("+type(exc).__name__+"): "+str(exc)[:500]) from exc
