#!/usr/bin/env python3
"""从已核对的模型字段生成整模教学数据，不加载权重或执行模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def positive_int(model: dict[str, Any], key: str) -> int:
    value = model.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{model.get('id')}: {key} 必须是正整数")
    return value


def derive_case(model: dict[str, Any]) -> dict[str, Any]:
    """返回语义步骤；shape 是 TP=PP=CP=1 的逻辑布局，不是运行时 trace。"""
    hidden = positive_int(model, "hidden_size")
    layers = positive_int(model, "layers")
    heads = positive_int(model, "heads")
    vocab = positive_int(model, "vocab_size")
    experts = model.get("routed_experts", 0)
    first_dense = model.get("first_dense_layers", 0 if experts else layers)
    if type(first_dense) is not int or not 0 <= first_dense <= layers:
        raise ValueError("first_dense_layers 超出层数范围")
    if experts and not 0 < model.get("experts_per_token", 0) <= experts:
        raise ValueError("experts_per_token 必须在专家数量范围内")

    if model["attention"] == "gqa":
        dim = positive_int(model, "head_dim")
        kv_heads = positive_int(model, "kv_heads")
        if heads % kv_heads:
            raise ValueError("GQA 的 query head 数必须是 KV head 数的整数倍")
        attention = {
            "type": "gqa", "q_projection_width": heads * dim,
            "kv_projection_width_each": kv_heads * dim,
            "fused_qkv_width": (heads + 2 * kv_heads) * dim,
            "q_shape": ["B", "S", heads, dim],
            "k_v_shape": ["B", "S", kv_heads, dim],
            "query_heads_per_kv": heads // kv_heads,
            "qk_norm": model["qk_norm"], "qkv_bias": model["qkv_bias"],
            "output_projection_weight_shape": [hidden, heads * dim],
            "source_ids": list(dict.fromkeys(model["sources"] + ["C-ATTN", "B-QKV"])),
            "steps": ["pre_attention_norm", "qkv_projection_and_grouped_split"]
                     + (["per_head_qk_norm"] if model["qk_norm"] else [])
                     + ["apply_rope", "causal_gqa", "attention_output_projection"],
        }
    elif model["attention"] == "mla":
        kv_rank = positive_int(model, "kv_lora_rank")
        no_pe = positive_int(model, "qk_nope_head_dim")
        rope = positive_int(model, "qk_rope_head_dim")
        v_dim = positive_int(model, "v_head_dim")
        q_rank = model.get("q_lora_rank")
        if q_rank is not None and (type(q_rank) is not int or q_rank <= 0):
            raise ValueError("q_lora_rank 必须为 null 或正整数")
        attention = {
            "type": "mla", "q_lora_rank": q_rank,
            "q_path": "direct_projection" if q_rank is None else "down_norm_up",
            "kv_lora_rank": kv_rank, "kv_down_combined_width": kv_rank + rope,
            "expanded_q_head_dim": no_pe + rope,
            "expanded_k_head_dim": no_pe + rope, "expanded_v_head_dim": v_dim,
            "expanded_q_shape": ["B", "S", heads, no_pe + rope],
            "expanded_v_shape": ["B", "S", heads, v_dim],
            "output_projection_weight_shape": [hidden, heads * v_dim],
            "cache_note": "仅为 MLA 训练语义图；未测量推理 KV cache、absorption 或显存。",
            "source_ids": list(dict.fromkeys(model["sources"] + ["C-MLA", "B-DSMAP"])),
            "steps": ["pre_attention_norm"]
                     + (["q_direct_projection"] if q_rank is None else ["q_down_projection", "q_latent_norm", "q_up_projection"])
                     + ["kv_down_projection", "split_kv_latent_and_positional_key", "kv_latent_norm",
                        "kv_up_projection", "decoupled_rope", "causal_attention", "attention_output_projection"],
        }
    else:
        raise ValueError(f"未实现的 attention: {model['attention']}")

    layer_layout = []
    for index in range(layers):
        is_moe = experts > 0 and index >= first_dense
        ffn = model.get("expert_ffn_hidden_size", model["ffn_hidden_size"]) if is_moe else model["ffn_hidden_size"]
        layer_layout.append({
            "layer_index": index, "attention": model["attention"],
            "feed_forward": "moe" if is_moe else "dense_swiglu",
            "hidden_shape": ["B", "S", hidden],
            "semantic_sequence": ["pre_attention_norm", "attention_subgraph", "attention_residual_add",
                                  "pre_ffn_norm", "moe_subgraph" if is_moe else "dense_swiglu_subgraph",
                                  "ffn_residual_add"],
            "ffn_source_ids": ["C-MOE"] if is_moe else model["sources"],
            "shared_branch_note": "shared expert 是并行语义分支，实际 overlap 取决于配置与后端" if is_moe and model.get("shared_experts", 0) else None,
            "branch_ffn_width": ffn, "branch_fused_gate_up_width": 2 * ffn,
            "routed_experts": experts if is_moe else 0,
            "active_routed_experts": model.get("experts_per_token", 0) if is_moe else 0,
            "shared_experts": model.get("shared_experts", 0) if is_moe else 0,
            "shared_intermediate_width": model.get("shared_experts", 0) * ffn if is_moe else 0,
        })

    return {
        "id": model["id"], "hf_id": model["hf_id"],
        "provenance": "derived", "execution_status": "not_run",
        "layout_scope": "整模逻辑形状，以 [B,S,...] 展示；不声称实际 Megatron storage/layout 相同",
        "parallelism": {"tp": 1, "pp": 1, "cp": 1, "ep": 1, "sp": False},
        "runtime_source_lane": None,
        "learning_source_lane": "bridge_sft",
        "attention": attention,
        "pipeline": [
            {"id": "input", "label": "样本、tokenizer、roles 与单次 label shift", "source_ids": ["B-SFTDATA"], "shape": ["B", "S"]},
            {"id": "embedding", "label": "词嵌入", "source_ids": ["C-GPT"], "shape": ["B", "S", hidden]},
            {"id": "decoder", "label": f"完整 {layers} 层 decoder，含 attention/残差/FFN", "source_ids": ["C-GPT"], "shape": ["B", "S", hidden]},
            {"id": "final_norm", "label": "最后的 RMSNorm", "source_ids": ["C-BLOCK"] + model["sources"], "shape": ["B", "S", hidden]},
            {"id": "lm_head", "label": "词表投影；推导形状，不物化大矩阵", "source_ids": ["C-GPT"], "shape": ["B", "S", vocab]},
            {"id": "loss", "label": "模型 token loss → mask sum/count → 调度归一化", "source_ids": ["C-GPT", "B-LOSS", "B-STEP"]},
            {"id": "backward", "label": "独立解释反向路径，非 forward 动画反放", "source_ids": ["B-TRAIN"]},
            {"id": "update", "label": "优化器更新、保存与恢复", "source_ids": ["B-TRAIN", "B-PRETRAIN"], "source_boundary": "train_step 和完整 save/resume 子链仍须在执行阶段展开"},
        ],
        "layers": layer_layout,
        "optional_mtp": {"configured_layers": model.get("mtp_layers_in_hf_config", 0), "included_in_base_walkthrough": False,
                         "note": "主图仅标准 next-token decoder；MTP 是显式附加分支，不宣称配置中的 MTP 已执行。"},
    }


def build(root: Path = ROOT) -> dict[str, Any]:
    models = json.loads((root / "content/models.json").read_text(encoding="utf-8"))["models"]
    return {"schema_version": 1, "generator": "tools/build_case_data.py", "contains_measured_results": False,
            "cases": [derive_case(model) for model in models]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.root / "content/generated/model-cases.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build(args.root)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {len(result['cases'])} 个整模派生档案：{output}")


if __name__ == "__main__":
    main()
