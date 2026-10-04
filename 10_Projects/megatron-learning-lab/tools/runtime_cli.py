#!/usr/bin/env python3
"""Read-only G08 entry points. No trace JSON or inspected launcher is executed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.runtime.contracts import read_trace
from experiments.runtime.source_probe import inspect_bridge, inspect_nemo_cli


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
    args=parser.parse_args()
    if args.command=="validate-trace":
        if args.path.stat().st_size>1_048_576:raise ValueError("trace exceeds 1 MiB")
        result=read_trace(args.path.read_text(encoding="utf-8"))
        result={k:v for k,v in result.items() if k not in ("config","data")}
    elif args.command=="probe-bridge":
        result=inspect_bridge(args.root,args.recipe)
    else:
        result=inspect_nemo_cli(args.root,args.algorithm)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    try:raise SystemExit(main())
    except (ValueError,OSError,UnicodeError) as exc:
        print(json.dumps({"status":"rejected","error":str(exc)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
