---
type: runbook
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# 资源与外部阻塞

## 默认可连续执行的范围

M00/G02–G09 是本项目的本地软件、课程、CPU 参考和测试工作。复用当前环境与公开固定源码缓存；必要 CPU 依赖遵守现有审批。代码能编写不代表对应模型已执行，test doubles 只能证明接入合约。

R01/R02 的真实 GPU/rollout 运行需要单独明确授权。本次“提交阶段计划”没有默示授权购买算力、远程启动集群、改驱动、下载巨量权重或访问私有数据。不要自动扫描用户 home/SSH 配置寻找资源。

## 一次性资源清单

用户可在当前 Codex 会话明确提供，或写入被现有 runs 忽略规则覆盖的 `runs/program-v1/resources.json`。清单不存秘钥；私有地址、模型路径、原始输出不提交到公共仓库。以下是未授权模板：

```json
{
  "grants": {
    "bridge_sft": {"authorized": false, "authorization_ref": ""},
    "rl_training": {"authorized": false, "authorization_ref": ""}
  },
  "execution": {
    "location": null, "devices": [], "max_gpu_count": 0,
    "max_wall_seconds": 0, "max_steps": 0,
    "model_snapshot": null, "tokenizer_snapshot": null,
    "data_path": null, "output_path": null,
    "allow_model_download": false,
    "allow_remote_jobs": false
  }
}
```

实际运行前必须明确设备归属/数量、时限、步数、权重与数据、输出目录、backend、是否允许联网。建议最小 smoke：短序列128–512、SFT两步、RL两次同步迭代，具体资源上限由用户确认；建议不是默认已授权预算。

planner 中 grant 只是帮助选任务，不是安全审批；agent 不能自己把 authorized 改为 true，也不能因为文件声称授权就绕过客户端审批。R01/R02 已有授权且资源就绪时可顺序执行，不必再次逐阶段询问。

## 选择下一工作项

缺 G03 的 CPU 依赖时，可继续不依赖其完成的 G05/G06 等工作；G04 依赖 G03，不能绕过。缺 R01 GPU 权限时，先记录 blocker，继续 G09 整合和未完成的软件任务。R02 需要 R01 的验证结果，不能用 G08 的 test doubles 代替。

每个 blocker 写：具体能力/文件、尝试过的命令和错误、哪些阶段受影响、可独立继续什么、最小解阻条件。对同一个失败，在无新假设/无环境变化时最多三次尝试；不要反复下载、重新安装所有依赖或偷偷换算法。

资源后来到位，恢复同一总 Goal，重验授权/环境和输入后处理 R01/R02；任何集成代码变化都将 G09 标为 revalidate，重新跑整体验收。用户撤销授权时立即停止对应外部执行，保留已取得证据。
