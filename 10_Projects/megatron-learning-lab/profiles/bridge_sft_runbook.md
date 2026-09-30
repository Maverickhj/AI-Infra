---
type: runbook
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# Bridge-SFT 执行入口与环境门槛

## 环境

优先复用当前可用训练环境，先按 [最小兼容策略](minimum-compatibility.md) 验收 `bridge_sft` 所需接口与语义，不要求逐包匹配参考 Bridge/Core/TE 或统一 Python3.12。实际安装包本身的 Python/ABI 约束仍须满足。记录所用组件版本和 import 路径；只修复必要依赖。只读网页不依赖 GPU 训练环境。

## 固定参考版的启动器模板

`Megatron-Bridge/scripts/training/run_recipe.py` 支持 `--recipe`、`--mode sft`、`--pretrained_checkpoint`、`--save_dir`、`--save_interval`，以及 `-ms/-mb/-gb/-sl/-tp/-pp/-cp`。`-sl`首先更新dataset，再由runner同步model.seq_length。不要只改model.seq_length导致dataset不同步。

以下仅是参考 Bridge v0.6.2 的命令骨架。当前 runner 的签名/参数不同应使用已核验的等价入口或薄适配器，先输出有效配置，不能照抄失败后重装整套环境。**本包没有执行这个GPU命令**。要求PRETRAINED_CKPT已经是由选定版本准备并验证的checkpoint，数据已可访问：

```bash
cd "$MEGATRON_BRIDGE_ROOT"
: "${PRETRAINED_CKPT:?请指定已验证的预训练 checkpoint}"
: "${RUN_DIR:?请指定新的输出目录}"

"${TRAIN_PYTHON:-python}" -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe qwen3_600m_sft_1gpu_h100_bf16_config \
  --mode sft --dataset squad \
  --pretrained_checkpoint "$PRETRAINED_CKPT" \
  --save_dir "$RUN_DIR/checkpoints" --save_interval 1 \
  -ms 2 -mb 1 -gb 2 -sl 512 -tp 1 -pp 1 -cp 1 \
  model.sequence_parallel=false \
  dataset.enable_offline_packing=false
```

`TRAIN_PYTHON` 指向已通过 profile 检查的 Python 解释器；不让启动命令隐式同步上游依赖。

这是参考版官方 SQuAD preset 的入口示例。`default_squad_config` 使用prompt-completion预处理，不是多轮chat。lab的assistant mask课程需建立独立的ChatSFTPreprocessingConfig与自有少量conversation样本配置，不能把上述SQuAD跑通当作chat mask验证完成。

H100 recipe 名称不保证其他 GPU 兼容。执行前检查最终配置、资源、checkpoint与data；使用网络受限环境时先完成cache/模型快照准备。不要自动发起权重/数据大下载。

## 采集限制

保存选定层、选定token的小切片/统计；不自动dump全词表logits和全部activation；不要逐层synchronize。数值采集和profiling分开，trace记录采集范围、未观测的fused中间态及run manifest。
