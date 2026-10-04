"""Produce small *executed CPU reference* traces for the read-only importer.

Nothing runs on import; --write is explicit. These are authored model/trajectory
fixtures, not an observed Bridge/HF/NeMo RL training run.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from .contracts import SCHEMA, digest_text, read_trace
from .source_probe import anchor

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "content/fixtures/runtime-reference.json"


def common_manifest(torch_version, task, started, completed):
    source = ROOT / "experiments" / ("decoder_reference.py" if task == "sft" else "rl_reference.py")
    located = anchor(source, "TinyDecoder.forward" if task == "sft" else "reference_run")
    located["component"] = "authored_" + task
    fixture = ROOT / "content/fixtures/sft-data.json"
    return dict(adapter="authored_cpu_v1", dtype="float64", layout="BS", backend="torch-cpu",
                runtime_profile="authored_"+task, source_lane="authored_cpu",
                evidence_kind="authored_cpu",
                model=dict(id="authored-two-layer-decoder", revision="authored-v1", weights_origin="authored",
                           architecture_origin="authored_scaled"),
                tokenizer=dict(id="authored-visible-vocabulary", revision="fixture-v1",
                               chat_template_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                               note="hash identifies authored serialization fixture, not an HF chat template"),
                parallel=dict(tp=1, pp=1, dp=1, cp=1, ep=1, world_size=1, groups_origin="in_memory_reference"),
                software=dict(python=sys.version.split()[0], torch=str(torch_version),
                              cuda="not_applicable", transformer_engine="not_applicable"),
                runtime_sources=[located],
                capture=dict(scope="selected_slices", timing="not_measured", unobserved=["GPU", "production framework", "rollout"]),
                execution=dict(status="executed", synthetic=False, device="cpu", started_at=started, completed_at=completed),
                limitations=["authored weights/tokens; not pretrained", "not observed_bridge or observed_rl"])


def envelope(run_id, task, manifest, config, data, measurements):
    config_text = json.dumps(config, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    input_text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    result = dict(schema=SCHEMA, schema_version=1, run_id=run_id, task=task, provenance="reference",
                  manifest=manifest, config_json=config_text, config_sha256=digest_text(config_text),
                  input_json=input_text, input_sha256=digest_text(input_text), measurements=measurements)
    read_trace(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


def generate():
    import torch
    from experiments.decoder_reference import TinyDecoder, sample_data
    from experiments.rl_reference import reference_run, plain_trace, FIXTURE, state_hash
    now = lambda: datetime.now(timezone.utc).isoformat()
    started = now()
    model = TinyDecoder().eval()
    ids, labels, mask = sample_data()
    with torch.no_grad():
        result = model(ids, labels, mask)
    count = int(mask.sum())
    lp = -result["token_loss"]
    data = dict(input_ids=[ids.tolist()], labels=[labels.tolist()], loss_mask=[mask.to(torch.int64).tolist()],
                alignment="next_token", position_ids=[list(range(len(ids)))], document_ids=[[0]*len(ids)],
                sample_ids=["arithmetic-multiturn"])
    config = {k:v for k,v in model.fixture.items() if k != "parameters"}
    config.update(vocab_size=27, mode="eval", active_dropout=0, tied_head=True, supervision="assistant",
                  operation="teacher_forced_forward", weights_fixture_sha256=hashlib.sha256((ROOT/"content/fixtures/decoder-reference.json").read_bytes()).hexdigest())
    sft = envelope("authored-sft-cpu-v1", "sft", common_manifest(torch.__version__, "sft", started, now()), config, data,
                   dict(token_logprobs=[lp.tolist()], token_count=count,
                        loss_sum=float((result["token_loss"]*mask).sum()), loss_mean=float(result["loss"]),
                        update=None, update_reason="this exported slice is forward-only; G03 has separate update/resume evidence"))
    started = now()
    r = plain_trace(reference_run("ppo", "token", True, False))
    fixture = json.loads(FIXTURE.read_text())
    config = {**fixture["config"], "vocab_size":27, "algorithm":"ppo", "loss_variant":"clipped_pg_k3",
              "offpolicy_correction":False, "reduction":"token", "ratio_clip":fixture["config"]["clip"],
              "force_on_policy":False, "kl_sampling":"non_is_score_gradient",
              "backbone_frozen":True, "fixture_sha256":hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}
    data = dict(input_ids=r["ids"], response_mask=[[int(v) for v in row] for row in r["mask"]],
                sample_mask=[1]*4, alignment="action_position", trajectory_ids=[x["id"] for x in fixture["trajectories"]],
                prompt_ids=[x["group"] for x in fixture["trajectories"]], group_ids=[x["group"] for x in fixture["trajectories"]],
                policy_versions=dict(generation=0, previous=0, current=1, after=2))
    measurements = {k:r[k] for k in ("generation_logprobs", "previous_logprobs", "current_logprobs",
                                     "reference_logprobs", "advantages", "rewards")}
    measurements.update({k:r["terms"][k] for k in ("ratio", "pg_token", "actor_loss", "kl_loss", "policy_loss")})
    measurements.update({k:r["value"][k] for k in ("values", "old_values", "returns", "value_loss")})
    measurements.update(refit=dict(status="not_run", reason="no generation engine called"),
                        update=dict(parameter="head[10,0]", before=fixture["heads"]["current"][10][0],
                                    gradient=r["head_gradient"][10][0], after=r["head_after"][10][0],
                                    weights_sha256_after=state_hash(r["head_after"]),
                                    optimizer="torch.optim.SGD", optimizer_executed=True),
                        actions_origin="authored_fixed")
    rl = envelope("authored-ppo-cpu-v1", "rl", common_manifest(torch.__version__, "rl", started, now()), config, data, measurements)
    return {"schema_version":1, "note":"Actual authored CPU references; imported metadata is not runtime attestation", "traces":[sft,rl]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = generate()
    if args.write:
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status":"passed", "traces":[{"run_id":t["run_id"],"task":t["task"],"provenance":t["provenance"]} for t in result["traces"]],
                      "written":str(OUTPUT) if args.write else None}, indent=2))
