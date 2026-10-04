"""Private, resource-rechecked child entry. JSON never selects code to execute."""
from __future__ import annotations
import argparse
import json
import signal
import sys
from .launch import verify_worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--resources",required=True)
    parser.add_argument("--receipt",required=True)
    args=parser.parse_args()
    plan,output=verify_worker(args.plan,args.resources,args.receipt)
    def stop(signum,frame):
        raise KeyboardInterrupt("runtime supervisor requested shutdown")
    signal.signal(signal.SIGTERM,stop)
    if plan["profile"]=="hf_reference":
        from .hf_entry import run
    elif plan["profile"]=="bridge_sft":
        from .bridge_entry import run
    elif plan["profile"] in ("rl_grpo","rl_ppo"):
        from .nemo_entry import run
    else:
        raise ValueError("unknown worker profile")
    result=run(plan,output)
    print(json.dumps({"status":"completed","result":result},ensure_ascii=False))
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError,OSError) as exc:
        print(json.dumps({"status":"rejected","error":str(exc)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
