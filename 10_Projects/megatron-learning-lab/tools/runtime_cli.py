#!/usr/bin/env python3
"""Inspect bounded runtime data, resolve configs, or execute an explicitly granted plan."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.runtime.contracts import read_trace
from experiments.runtime.source_probe import inspect_bridge, inspect_nemo_runtime
from experiments.runtime.plan import dry_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    validate=sub.add_parser("validate-trace",help="Validate bounded UTF-8 trace data without running it")
    validate.add_argument("path",type=Path)
    bridge=sub.add_parser("probe-bridge",help="AST-only inspection; does not prove runtime compatibility")
    bridge.add_argument("--root",type=Path,required=True)
    bridge.add_argument("--recipe",default="qwen3_600m_sft_config")
    nemo=sub.add_parser("probe-nemo",help="Inspect a provided source checkout without importing NeMo RL")
    nemo.add_argument("--root",type=Path,required=True)
    nemo.add_argument("--algorithm",choices=("grpo","ppo"),required=True)
    dry=sub.add_parser("dry-run",help="Read-only plan/source/resource inspection; never initializes a runtime")
    dry.add_argument("--plan",type=Path,required=True)
    dry.add_argument("--resources",type=Path)
    run=sub.add_parser("run",help="Requires an exact-plan-bound, explicit resource authorization manifest")
    run.add_argument("--plan",type=Path,required=True)
    run.add_argument("--resources",type=Path,required=True)
    resolve=sub.add_parser("resolve-nemo",help="Read-only YAML/Hydra resolution; no NeMo launcher import")
    resolve.add_argument("--root",type=Path,required=True)
    resolve.add_argument("--config",type=Path,required=True)
    resolve.add_argument("--override",action="append",default=[])
    freeze=sub.add_parser("freeze-nemo",help="Freeze resolved configuration bytes before requesting runtime resources")
    freeze.add_argument("--root",type=Path,required=True)
    freeze.add_argument("--config",type=Path,required=True)
    freeze.add_argument("--override",action="append",default=[])
    freeze.add_argument("--output",type=Path,required=True)
    inspect=sub.add_parser("inspect-nemo-config",help="Check a frozen RL configuration against its bounded plan")
    inspect.add_argument("--plan",type=Path,required=True)
    constructed=sub.add_parser("resolve-bridge",help="Isolated config construction; blocks network, weights and GPU initialization")
    constructed.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args()
    code=0
    if args.command=="validate-trace":
        if args.path.stat().st_size>1_048_576:raise ValueError("trace exceeds 1 MiB")
        result=read_trace(args.path.read_text(encoding="utf-8"))
        result={k:v for k,v in result.items() if k not in ("config","data")}
    elif args.command=="probe-bridge":
        result=inspect_bridge(args.root,args.recipe)
    elif args.command=="probe-nemo":
        result=inspect_nemo_runtime(args.root,args.algorithm)
    elif args.command=="dry-run":
        result=dry_run(args.plan,args.resources)
    elif args.command=="freeze-nemo":
        from experiments.runtime.nemo_config import freeze_config
        result=freeze_config(args.config,source_root=args.root,overrides=args.override,output=args.output)
    elif args.command=="inspect-nemo-config":
        from experiments.runtime.nemo_config import read_bound_config
        from experiments.runtime.plan import read_document,validate_plan
        config,source,mapping=read_bound_config(validate_plan(read_document(args.plan)))
        result=dict(status="configuration_bound",source=source,mapping=mapping,execution="not_run")
    elif args.command=="resolve-bridge":
        from experiments.runtime.config_probe import probe_bridge
        result=probe_bridge(args.plan)
    elif args.command=="resolve-nemo":
        from experiments.runtime.read_only_config import resolve_nemo_config
        result=resolve_nemo_config(args.config,source_root=args.root,overrides=args.override)
    else:
        from experiments.runtime.launch import launch
        result=launch(args.plan,args.resources)
        code=0 if result["execution"]=="passed" else 1
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return code


if __name__=="__main__":
    try:raise SystemExit(main())
    except (ValueError,OSError,UnicodeError,ImportError) as exc:
        print(json.dumps({"status":"rejected","error":str(exc)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
