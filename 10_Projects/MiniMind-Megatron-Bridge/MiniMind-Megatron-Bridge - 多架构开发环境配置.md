---
type: runbook
status: draft
created: 2026-08-09
updated: 2026-08-10
domains:
  - multimodal-training
  - megatron
  - distributed-training
  - development-environment
aliases:
  - MiniMind Megatron Bridge 开发容器
  - Megatron Bridge Windows DGX Spark 环境
source:
  - https://github.com/NVIDIA-NeMo/Megatron-Bridge
  - https://docs.nvidia.com/nemo/megatron-bridge/latest/
  - https://docs.nvidia.com/dgx/dgx-spark/system-overview.html
  - https://github.com/jingyaogong/minimind-v
ai_generated: true
reviewed: false
---

# MiniMind-Megatron-Bridge 多架构开发环境配置

> [!warning]
> 本文包含 AI 生成或改写的实质内容，尚未完成人工核验。Windows 侧 Dev Container 已完成实际启动、GPU、Transformer Engine、Bridge/MCore workspace import smoke test；DGX Spark 侧配置仍需在实机上验证。

- **关联项目**：[[_Project - MiniMind Megatron Bridge|MiniMind-V Megatron Bridge 接入]]
- **适用范围**：Windows 11 + Docker Desktop/WSL2 + RTX 4070 Ti SUPER，以及 NVIDIA DGX Spark ARM64
- **目标**：使用同一套 Dev Container 定义完成模型开发、权重转换、单元测试、数值对齐和单卡 SFT smoke test，并自动识别当前运行环境

## 设计结论

采用“一套容器定义、一个多架构镜像、启动时生成硬件画像”的方案：

```text
同一个 devcontainer.json
同一个 compose.yaml
同一个 NeMo 26.06 多架构镜像
           │
           ▼
容器启动时运行 detect_runtime.py
           │
           ├── windows_wsl2_ada_16g
           ├── dgx_spark_arm64_uma
           ├── nvidia_x86_64_generic
           ├── nvidia_arm64_generic
           └── cpu_only
```

自动探测只负责选择资源和性能参数，不得改变以下模型语义：

- 模型结构和参数 shape；
- tokenizer、processor 和图像占位符；
- 数据样本与样本顺序；
- attention mask、position ids、labels 和 loss mask；
- HF ↔ Megatron 权重映射；
- parity 模式的精度、随机种子和容差。

## 已锁定的基础版本

### NeMo 镜像

```text
image:
  nvcr.io/nvidia/nemo:26.06

multi-architecture index digest:
  sha256:912033288c982a8c4af05df46a1d670c34350f1427c758f5da9c485bdec57264

linux/amd64 manifest:
  sha256:c6c64a07935c2765a8f42811b4baefa675aaaa324c9ed9e1901332816bb89bae

linux/arm64 manifest:
  sha256:6ecd3aaf8f8fe8dfa2c2320d22df989d9a0613f97b0f42ca551e869d037258cf
```

Compose 中使用多架构 index digest，不设置固定 `platform:`。Docker 应根据当前宿主机自动选择 `amd64` 或 `arm64` manifest。

### 镜像内软件基线

以下版本已在本地 `amd64` 镜像中读取：

| 组件 | 版本或 commit |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | `2.12.0a0+0291f960b6.nv26.04.48445190` |
| CUDA runtime | 13.2 |
| Transformer Engine | 2.16.0 |
| Transformers | 5.8.1 |
| Megatron Bridge | `0.5.1+cdbcb3073` |
| Megatron Core | `0.18.2+458c8d0ec` |
| Megatron-Bridge commit | `cdbcb307353740e49cc61d2e825be779a071689a` |
| Megatron-LM commit | `458c8d0ecafdf6d9e36771600d62ade27f2a67b7` |

首个开发基线应先 checkout 到以上 Megatron-Bridge commit，并初始化其固定的 Megatron-LM 子模块。完成 P0 验证后，再单独评估是否升级到更新的 `main`。

```bash
git clone https://github.com/<your-account>/Megatron-Bridge.git
cd Megatron-Bridge
git checkout cdbcb307353740e49cc61d2e825be779a071689a
git submodule update --init --recursive

git rev-parse HEAD
git -C 3rdparty/Megatron-LM rev-parse HEAD
```

预期输出分别为上述 Bridge 和 Megatron-LM commit。

### 实际开发分支基线（2026-08-10）

镜像内的 `cdbcb307...` 是 NeMo 26.06 构建时包含的 Bridge 版本，不是 2026-08-10 的最新仓库 tag。本次开发环境按项目要求从 NVIDIA 上游刷新 tags，并从最新正式 tag `v0.5.1` 创建开发分支：

```powershell
git fetch https://github.com/NVIDIA-NeMo/Megatron-Bridge.git --tags
git switch -c codex/dev-v0.5.1-minimind v0.5.1
git rev-parse HEAD
git ls-tree HEAD 3rdparty/Megatron-LM
```

实际锁定结果：

```text
Megatron-Bridge tag:    v0.5.1
Megatron-Bridge commit: 5cb3444c43f7499cf3872b2d46870cf8bc2e00ce
Megatron-LM commit:     458c8d0ecafdf6d9e36771600d62ade27f2a67b7
development branch:     codex/dev-v0.5.1-minimind
```

仓库 tag、镜像 digest 和子模块 commit 分别锁定，不能用镜像内 Python distribution 的 commit 替代当前 workspace commit。运行时必须同时记录两者。

## 已验证与待验证环境

### Windows 调试端

2026-08-09 已在实际 Docker Desktop/WSL2 环境验证：

```text
CPU architecture: x86_64
kernel: 6.6.87.2-microsoft-standard-WSL2
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
compute capability: 8.9
GPU memory: 15.99 GiB
PyTorch CUDA: available
Transformer Engine BF16 forward: passed
Transformer Engine BF16 backward: passed
input gradient: finite
```

该环境适合承担：

- Python、Bridge 和 Provider 开发；
- CPU 单元测试；
- HF ↔ Megatron 单卡转换；
- 固定输入 forward/backward；
- 小 batch、短序列 SFT smoke test；
- 与 DGX Spark 的跨架构数值对比。

### DGX Spark 调试端

根据 NVIDIA 官方资料，DGX Spark 使用 ARM64 CPU、GB10 Grace Blackwell 和 128 GB UMA 统一内存。以下项目仍需实机验证：

- [ ] `nvcr.io/nvidia/nemo:26.06` ARM64 manifest 能正常启动；
- [ ] 当前 DGX OS 驱动与镜像内 CUDA 13.2 runtime 兼容；
- [ ] PyTorch CUDA matmul 通过；
- [ ] Transformer Engine BF16 forward/backward 通过；
- [ ] `torch.cuda.get_device_name()` 和设备树信息足以自动识别 DGX Spark；
- [ ] `torch.cuda.get_device_properties().total_memory` 在 UMA 下的含义和可用上限；
- [ ] DataLoader 多进程和 pinned memory 的合理配置。

> [!important]
> DGX Spark 官方当前版本表列出的 GPU driver 与 NeMo 26.06 镜像构建环境并不完全相同。镜像包含 CUDA compatibility 组件，但是否兼容必须以实机 GPU smoke test 为准，不能只比较版本号推断。

## 目录与数据布局

两台机器分别保留独立 checkout，通过 Git commit 同步源码，不共享可写工作目录：

```text
Megatron-Bridge/
├── .devcontainer/
│   ├── devcontainer.json
│   ├── compose.yaml
│   ├── detect_runtime.py
│   ├── verify_env.py
│   └── generated/
│       └── runtime.json       # 不提交 Git
├── scripts/
│   └── run_detected.sh
├── src/
├── tests/
└── 3rdparty/Megatron-LM/
```

容器内统一使用：

```text
/opt/Megatron-Bridge    工作区源码
/opt/vendor             MiniMind-V 等外部参考源码
/data/hf                Hugging Face cache
/data/datasets          数据集
/data/checkpoints       checkpoint
/data/runs              实验输出
/data/cache             其他工具缓存
```

默认使用 Docker named volume，避免 Windows 与 Linux 宿主机路径差异。需要从宿主机直接读取大数据集时，可以增加不提交 Git 的 `compose.override.yaml`，将 `/data` 替换为本机 bind mount。

Windows 侧如需频繁执行 pytest 或扫描大量源码，优先把 Git checkout 放在 WSL2 ext4 文件系统中，而不是 Windows NTFS 路径中，以减少跨文件系统 I/O 开销。

## `.devcontainer/compose.yaml`

```yaml
name: minimind-megatron-bridge

services:
  dev:
    image: nvcr.io/nvidia/nemo:26.06@sha256:912033288c982a8c4af05df46a1d670c34350f1427c758f5da9c485bdec57264
    pull_policy: missing

    working_dir: /opt/Megatron-Bridge
    command: ["bash", "-lc", "sleep infinity"]
    init: true
    stdin_open: true
    tty: true

    shm_size: 8gb

    ulimits:
      memlock: -1
      stack: 67108864

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    environment:
      PYTHONPATH: /opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM
      HF_HOME: /data/hf
      XDG_CACHE_HOME: /data/cache
      UV_CACHE_DIR: /data/cache/uv
      UV_PROJECT_ENVIRONMENT: /opt/venv
      TOKENIZERS_PARALLELISM: "false"
      PYTHONUNBUFFERED: "1"
      MINIMIND_NEMO_IMAGE: nvcr.io/nvidia/nemo:26.06@sha256:912033288c982a8c4af05df46a1d670c34350f1427c758f5da9c485bdec57264
      MINIMIND_BRIDGE_SOURCE_TAG: v0.5.1

    volumes:
      - ..:/opt/Megatron-Bridge
      - hf-cache:/data/hf
      - dataset-cache:/data/datasets
      - checkpoint-cache:/data/checkpoints
      - run-cache:/data/runs
      - generic-cache:/data/cache
      - vendor-cache:/opt/vendor

volumes:
  hf-cache:
  dataset-cache:
  checkpoint-cache:
  run-cache:
  generic-cache:
  vendor-cache:
```

`gpus: all` 需要较新的 Compose 版本。本次 Windows 环境为 Docker Compose `v2.29.7`，实际配置使用 `deploy.resources.reservations.devices` 请求全部 NVIDIA GPU；该写法同时适用于当前 Docker Desktop 和 DGX Spark Compose。

默认不设置以下权限：

- `privileged: true`；
- `pid: host`；
- Docker socket 挂载。

只有在 Nsight、底层 debugger 或容器内 Docker 确有需要时，才通过独立 override 显式增加权限。

## `.devcontainer/devcontainer.json`

```jsonc
{
  "name": "MiniMind Megatron Bridge",
  "dockerComposeFile": "compose.yaml",
  "service": "dev",
  "workspaceFolder": "/opt/Megatron-Bridge",
  "shutdownAction": "stopCompose",
  "remoteUser": "root",

  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.debugpy",
        "charliermarsh.ruff",
        "ms-azuretools.vscode-docker"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/opt/venv/bin/python",
        "python.testing.pytestEnabled": true,
        "python.testing.pytestArgs": [
          "tests"
        ]
      }
    }
  },

  "postCreateCommand": "git submodule update --init --recursive && if grep -Eqi '(microsoft|wsl)' /proc/sys/kernel/osrelease; then git -C 3rdparty/Megatron-LM config core.filemode false && git config submodule.3rdparty/Megatron-LM.ignore dirty; fi && chmod +x scripts/run_detected.sh",
  "postStartCommand": "uv run --no-sync python .devcontainer/detect_runtime.py --output .devcontainer/generated/runtime.json && uv run --no-sync python .devcontainer/verify_env.py --runtime .devcontainer/generated/runtime.json --output .devcontainer/generated/verification.json",
  "postAttachCommand": "uv run --no-sync python -m json.tool .devcontainer/generated/runtime.json"
}
```

`postCreateCommand` 只在首次创建时初始化子模块。Windows bind mount 中由 Linux Git checkout 的 symlink/filemode 会被 Windows Git 误报为 dirty，因此仅在 WSL2 kernel 下设置本地 `core.filemode=false` 和 `submodule...ignore=dirty`；commit 漂移仍会被报告，DGX Spark 不应用该修正。`postStartCommand` 在每次容器启动时重新生成运行时画像、执行环境闸门，并写出独立的验证报告。

## `.devcontainer/detect_runtime.py`

```python
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch


WORKSPACE = Path("/opt/Megatron-Bridge")


def read_optional(path: str) -> str:
    file = Path(path)
    if not file.exists():
        return ""
    try:
        return file.read_text(errors="ignore").replace("\x00", "").strip()
    except OSError:
        return ""


def git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def system_memory_gib() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / 1024**3, 2)
    except (AttributeError, OSError, ValueError):
        return None


def detect_runtime() -> dict[str, Any]:
    machine = platform.machine().lower()
    kernel = platform.release().lower()

    gpu_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if gpu_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else ""
    gpu_props = torch.cuda.get_device_properties(0) if gpu_available else None
    capability = torch.cuda.get_device_capability(0) if gpu_available else (0, 0)

    hardware_identity = " ".join(
        part
        for part in [
            gpu_name,
            read_optional("/proc/device-tree/model"),
            read_optional("/sys/class/dmi/id/product_name"),
            read_optional("/sys/devices/virtual/dmi/id/product_name"),
        ]
        if part
    ).lower()

    is_x86_64 = machine in {"x86_64", "amd64"}
    is_arm64 = machine in {"aarch64", "arm64"}
    is_wsl2 = is_x86_64 and ("microsoft" in kernel or "wsl" in kernel)
    is_dgx_spark = is_arm64 and any(
        marker in hardware_identity
        for marker in ("dgx spark", "gb10", "grace blackwell")
    )

    override = os.environ.get("MINIMIND_RUNTIME_PROFILE")

    if override:
        profile = override
        detection_source = "environment_override"
    elif is_wsl2 and "4070 ti super" in gpu_name.lower():
        profile = "windows_wsl2_ada_16g"
        detection_source = "automatic"
    elif is_dgx_spark:
        profile = "dgx_spark_arm64_uma"
        detection_source = "automatic"
    elif gpu_available and is_x86_64:
        profile = "nvidia_x86_64_generic"
        detection_source = "automatic_fallback"
    elif gpu_available and is_arm64:
        profile = "nvidia_arm64_generic"
        detection_source = "automatic_fallback"
    elif gpu_available:
        profile = "nvidia_generic"
        detection_source = "automatic_fallback"
    else:
        profile = "cpu_only"
        detection_source = "automatic_fallback"

    memory_model = "unified" if profile == "dgx_spark_arm64_uma" else "discrete"
    compute_capability = f"{capability[0]}.{capability[1]}" if gpu_available else None

    return {
        "schema_version": 1,
        "profile": profile,
        "detection_source": detection_source,
        "machine": machine,
        "kernel": kernel,
        "hardware_identity": hardware_identity,
        "gpu_available": gpu_available,
        "device_count": device_count,
        "gpu_name": gpu_name or None,
        "compute_capability": compute_capability,
        "gpu_memory_gib": (
            round(gpu_props.total_memory / 1024**3, 2)
            if gpu_props is not None
            else None
        ),
        "system_memory_gib": system_memory_gib(),
        "memory_model": memory_model,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "torch_cuda_arch_list": compute_capability,
        "megatron_bridge_commit": git_revision(WORKSPACE),
        "megatron_lm_commit": git_revision(WORKSPACE / "3rdparty/Megatron-LM"),
        "distributed_defaults": {
            "nproc_per_node": 1,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / ".devcontainer/generated/runtime.json",
    )
    args = parser.parse_args()

    runtime = detect_runtime()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(runtime, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

判断 DGX Spark 时不能只检查 `aarch64`，否则普通 ARM64 NVIDIA 服务器会被错误归类。若实机设备字符串未包含 `DGX Spark`、`GB10` 或 `Grace Blackwell`，探测器应返回 `nvidia_arm64_generic` 并发出提示，再根据实测信息补充稳定的设备标识。

## `.devcontainer/verify_env.py`

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformer_engine
import transformer_engine.pytorch as te
import transformers
from megatron import bridge
from megatron import core


WORKSPACE = Path("/opt/Megatron-Bridge").resolve()


def is_workspace_import(module_file: str | None) -> bool:
    if not module_file:
        return False
    try:
        return Path(module_file).resolve().is_relative_to(WORKSPACE)
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()

    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the development container")

    if not is_workspace_import(bridge.__file__):
        raise RuntimeError(f"megatron.bridge is not imported from workspace: {bridge.__file__}")

    if not is_workspace_import(core.__file__):
        raise RuntimeError(f"megatron.core is not imported from workspace: {core.__file__}")

    torch.manual_seed(1)

    matrix = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    matmul = matrix @ matrix

    layer = te.Linear(
        128,
        128,
        params_dtype=torch.bfloat16,
    ).cuda()
    inputs = torch.randn(
        4,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    outputs = layer(inputs)
    loss = outputs.float().square().mean()
    loss.backward()

    if inputs.grad is None or not torch.isfinite(inputs.grad).all():
        raise RuntimeError("Transformer Engine backward produced invalid gradients")

    report = {
        "runtime_profile": runtime["profile"],
        "gpu": torch.cuda.get_device_name(0),
        "matmul_finite": bool(torch.isfinite(matmul).all()),
        "te_forward_shape": list(outputs.shape),
        "te_forward_dtype": str(outputs.dtype),
        "te_loss": loss.item(),
        "te_input_grad_finite": bool(torch.isfinite(inputs.grad).all()),
        "torch_version": torch.__version__,
        "transformer_engine_version": transformer_engine.__version__,
        "transformers_version": transformers.__version__,
        "bridge_import": bridge.__file__,
        "core_import": core.__file__,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

该脚本同时验证 GPU 计算、Transformer Engine BF16 前后向和实际 Python import path，避免源码已挂载但运行时仍加载镜像内旧包。

## `scripts/run_detected.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

workspace=/opt/Megatron-Bridge
runtime_json="${workspace}/.devcontainer/generated/runtime.json"

python "${workspace}/.devcontainer/detect_runtime.py" \
  --output "${runtime_json}"

runtime_profile="$(${PYTHON:-python} -c \
  "import json; print(json.load(open('${runtime_json}'))['profile'])")"

cuda_arch="$(${PYTHON:-python} -c \
  "import json; print(json.load(open('${runtime_json}')).get('torch_cuda_arch_list') or '')")"

export MINIMIND_RUNTIME_PROFILE="${runtime_profile}"

if [[ -n "${cuda_arch}" ]]; then
  export TORCH_CUDA_ARCH_LIST="${cuda_arch}"
fi

exec "$@"
```

为该文件增加可执行权限：

```bash
chmod +x scripts/run_detected.sh
```

所有会编译 CUDA extension 或启动训练的命令通过该包装器执行：

```bash
scripts/run_detected.sh \
  python examples/models/minimind_v/train.py \
  --runtime-mode parity
```

这样可以按实际 GPU compute capability 设置 `TORCH_CUDA_ARCH_LIST`。不得把 Windows 的 `8.9` 或尚未实测的 DGX Spark capability 写死在 Compose 中。

## 配置模式

### `parity`：跨架构功能和数值对齐

Windows 与 DGX Spark 使用完全一致的语义配置：

```yaml
runtime_mode: parity
seed: 1234
precision: bf16
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
context_parallel_size: 1
micro_batch_size: 1
sequence_length: 256
fp8: false
torch_compile: false
deterministic: true
```

该模式用于：

- HF 与 Megatron 中间张量比较；
- HF → Megatron → HF round-trip；
- Windows 与 ARM64 logits、loss 和梯度比较；
- checkpoint save/load 回归；
- 单 batch overfit。

### `native`：按硬件选择资源参数

硬件参数应作为保守起点，而不是未经验证的最佳配置：

| 参数                     | Windows 4070 Ti SUPER | DGX Spark   |
| ---------------------- | --------------------- | ----------- |
| 内存模型                   | 16 GiB 独立显存           | 128 GB UMA  |
| precision              | BF16                  | BF16        |
| TP / PP / CP           | 1 / 1 / 1             | 1 / 1 / 1   |
| micro batch 初值         | 1                     | 1，实测后增加     |
| DataLoader workers 初值  | 2                     | 4           |
| gradient checkpointing | 开                     | 先开，实测后评估    |
| FP8                    | 默认关                   | 默认关，单独验证后开启 |
| sequence length hint   | 256–512               | 512–768     |

自动识别只应修改 micro batch、worker 数、重计算和其他资源参数。Recipe 必须允许命令行或配置文件覆盖所有自动默认值，并在日志中保存最终 resolved config。

## MiniMind-V 依赖隔离

当前存在明确的 Transformers 版本边界：

- NeMo 26.06 镜像内为 Transformers 5.8.1；
- 当前 Megatron-Bridge 要求 Transformers 5.x；
- MiniMind-V 上游 `requirements.txt` 固定 Transformers 4.57.6。

不得在 `/opt/venv` 中直接执行：

```bash
pip install -r /opt/vendor/minimind-v/requirements.txt
```

推荐保留两个逻辑环境：

```text
/opt/venv
  Megatron-Bridge、Megatron Core、Transformer Engine 和训练环境

/data/venvs/minimind-hf
  原始 MiniMind-V HF 基线环境，仅用于生成对齐数据
```

首选长期方案是将 MiniMind-V 的 HF 契约适配到 Transformers 5.x。在完成适配前，原始 HF 基线环境不得用于运行 Megatron-Bridge。

## 双机并行调测约定

### 源码同步

- Windows 和 DGX Spark 分别使用独立 Git checkout；
- 每轮对比记录完整 Git SHA；
- 不通过共享可写目录同步未提交源码；
- 不比较不同 commit、不同 checkpoint revision 或不同固定样本产生的结果。

### 产物路径

```text
/data/runs/<runtime-profile>/<git-sha>/<run-id>/
├── runtime.json
├── resolved_config.yaml
├── environment.json
├── sample_ids.json
├── metrics.jsonl
└── tensors/
    ├── vision_output.pt
    ├── projector_output.pt
    ├── fused_embeddings.pt
    ├── decoder_output.pt
    └── logits.pt
```

checkpoint、数据集和完整日志不提交到 AI-Infra Vault 或 Megatron-Bridge Git 仓库。

### 数值比较

- parity 模式必须使用相同固定输入、seed、dtype 和配置；
- BF16 使用事先记录的 `atol`、`rtol`；
- 逐层定位首次超出容差的位置，不只比较生成文本；
- 跨架构 reduction 顺序不同可能产生小误差，不能默认要求 bitwise equality；
- 比较报告必须记录两侧 `runtime.json`；
- 性能模式结果不得与 parity 模式的数值结论混在一起。

## 首次启动

### Windows / Docker Desktop

确认 Docker Desktop 使用 WSL2 backend，并允许容器访问 NVIDIA GPU：

```powershell
docker run --rm --gpus all `
  nvcr.io/nvidia/nemo:26.06@sha256:912033288c982a8c4af05df46a1d670c34350f1427c758f5da9c485bdec57264 `
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

预期当前机器输出 `True` 和 `NVIDIA GeForce RTX 4070 Ti SUPER`。

### DGX Spark

先更新到受支持的 DGX OS 版本，再验证容器 runtime：

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/nemo:26.06@sha256:912033288c982a8c4af05df46a1d670c34350f1427c758f5da9c485bdec57264 \
  python -c 'import platform, torch; print(platform.machine()); print(torch.cuda.is_available()); print(torch.cuda.get_device_name())'
```

期望至少满足：

```text
platform.machine() == aarch64
torch.cuda.is_available() == True
```

随后通过 VS Code Remote SSH 打开 DGX Spark 上的 checkout，并执行 `Reopen in Container`。

## 2026-08-10 Windows 本地实施记录

### 生成的仓库文件

```text
.devcontainer/
├── compose.yaml
├── devcontainer.json
├── detect_runtime.py
├── verify_env.py
├── version-lock.json
└── generated/
    ├── runtime.json          # Git 忽略
    └── verification.json     # Git 忽略
scripts/
└── run_detected.sh
```

`.gitignore` 必须包含 `.devcontainer/generated/`，只提交探测和验证逻辑，不提交单机运行画像。

### 启动、挂载与生命周期

先做 Compose 和 Dev Containers schema 校验，再实际启动：

```powershell
docker compose -f .devcontainer/compose.yaml config --quiet
docker compose -f .devcontainer/compose.yaml up -d --wait
docker compose -f .devcontainer/compose.yaml ps
```

实际启动的 service 为 `dev`，容器名为 `minimind-megatron-bridge-dev-1`。`.devcontainer/compose.yaml` 中的 `..` 解析为仓库根目录，并 bind mount 到 `/opt/Megatron-Bridge`。HF cache、数据集、checkpoint、运行结果、通用 cache 和 vendor 源码使用 Compose named volume；执行普通 `stop` 或 `down` 不删除这些卷，只有 `down -v` 会删除。

VS Code 的 `Reopen in Container` 会自动执行 `postCreateCommand` 和 `postStartCommand`。手工 Compose 验证时，应显式执行相同生命周期命令：

```powershell
docker compose -f .devcontainer/compose.yaml exec -T dev `
  bash -lc 'git submodule update --init --recursive && if grep -Eqi "(microsoft|wsl)" /proc/sys/kernel/osrelease; then git -C 3rdparty/Megatron-LM config core.filemode false && git config submodule.3rdparty/Megatron-LM.ignore dirty; fi && chmod +x scripts/run_detected.sh'

docker compose -f .devcontainer/compose.yaml exec -T dev `
  bash -lc 'uv run --no-sync python .devcontainer/detect_runtime.py --output .devcontainer/generated/runtime.json && uv run --no-sync python .devcontainer/verify_env.py --runtime .devcontainer/generated/runtime.json --output .devcontainer/generated/verification.json'
```

Windows NTFS bind mount 上首次 clone 和 checkout Megatron-LM 用时明显较长，主要耗时发生在 `git index-pack` 和大量小文件展开。若需要频繁运行 Git、pre-commit 或 pytest，应将 checkout 放到 WSL2 ext4 文件系统。

### 实测运行画像

```text
profile:                  windows_wsl2_ada_16g
detection_source:         automatic
machine:                  x86_64
kernel:                   6.6.87.2-microsoft-standard-wsl2
GPU:                      NVIDIA GeForce RTX 4070 Ti SUPER
compute capability:       8.9
GPU memory:               15.99 GiB
NVIDIA driver:            610.88
PyTorch:                  2.12.0a0+0291f960b6.nv26.04.48445190
CUDA runtime:             13.2
Transformer Engine:       2.16.0+b9d690e0
Transformers:             5.8.1
Megatron-Bridge commit:   5cb3444c43f7499cf3872b2d46870cf8bc2e00ce
Megatron-LM commit:       458c8d0ecafdf6d9e36771600d62ade27f2a67b7
```

### 实测环境闸门

```text
CUDA FP32 matmul:                 passed, finite output
Transformer Engine BF16 forward: passed, output shape [4, 128]
Transformer Engine BF16 backward: passed
input gradient:                   finite
megatron.bridge import:           /opt/Megatron-Bridge/src/megatron/bridge/__init__.py
megatron.core import:             /opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/__init__.py
```

Compose 解析、Dev Containers CLI `read-configuration`、Python bytecode compile、Ruff 和 `bash -n scripts/run_detected.sh` 均通过。NeMo 镜像导入期间会报告 `triton_kernels.matmul_ogs` 缺失、NIXL unavailable 和 TVM duplicate-field 警告；这些信息没有使当前 P0 gate 失败，但使用相应优化路径前必须单独验证。

仓库 pre-commit 在 Windows NTFS bind mount 中两次超过 180 秒，停留在容器内 `git diff --ignore-submodules` 阶段。此项记录为文件系统性能限制，而不是 lint 失败；完整 pre-commit 应在 WSL2 ext4 或原生 Linux checkout 中补跑。

## 验收清单

| 验证项 | Windows 4070 Ti SUPER | DGX Spark |
| --- | --- | --- |
| 自动架构识别 | 已验证 `x86_64/WSL2` | 待实机验证 |
| Docker manifest | `amd64` | `arm64` 待确认 |
| PyTorch CUDA | 已通过 | 待验证 |
| TE BF16 forward/backward | 已通过 | 待验证 |
| Bridge import 来自 workspace | 已通过，来自 `/opt/Megatron-Bridge/src` | 待实机验证 |
| MCore import 来自 workspace | 已通过，来自固定子模块 | 待实机验证 |
| 固定输入 HF 基线 | 待实现 | 复用同一基线 |
| HF ↔ Megatron round-trip | 待实现 | 待实现 |
| 单 batch backward | 待实现 | 待实现 |
| SFT smoke | 小 batch | 较大资源配置 |
| 跨架构数值对比 | 生成基线 | 与 Windows 对比 |

## 故障处理

### 自动识别为 generic profile

检查：

```bash
uname -m
cat /proc/sys/kernel/osrelease
cat /proc/device-tree/model 2>/dev/null || true
cat /sys/class/dmi/id/product_name 2>/dev/null || true
python -c 'import torch; print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())'
```

保留这些输出，并据此补充探测规则。临时验证可使用：

```bash
export MINIMIND_RUNTIME_PROFILE=dgx_spark_arm64_uma
```

手动 override 只能作为探测失败时的临时措施，运行记录中必须保留 `detection_source=environment_override`。

### `megatron.bridge` 指向镜像内代码

检查：

```bash
python -c 'import megatron.bridge; print(megatron.bridge.__file__)'
python -c 'import megatron.core; print(megatron.core.__file__)'
```

两者都应位于 `/opt/Megatron-Bridge`。若不是，检查 workspace mount、`PYTHONPATH` 和子模块初始化状态。不要通过复制镜像内源码修复。

### Windows 文件访问缓慢

若源码位于 `C:\...` 并出现 pytest、Git 或小文件扫描缓慢，应将 checkout 移到 WSL2 Linux 文件系统，并从 WSL/VS Code 打开。模型、数据和 checkpoint 继续使用 Docker named volume。

### DGX Spark 容器无法使用 CUDA

依次检查：

```bash
nvidia-smi
nvidia-ctk --version
docker info
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

若基础 CUDA 容器通过但 NeMo 26.06 失败，应保留完整错误、DGX OS/driver 版本和镜像 digest，再判断是驱动兼容还是 ARM64 镜像问题。不要在未确认 DGX OS 支持边界前手工替换系统驱动。

## 安全与维护边界

- 不把 Hugging Face、WandB、NGC token 写入 Compose、Git 或本文；
- 不挂载宿主机 Docker socket；
- 不默认启用 `privileged`；
- 不把 checkpoint、数据集、完整 tensor dump 或大日志放入 AI-Infra Vault；
- 镜像、Bridge、MCore、MiniMind-V 和 HF checkpoint revision 需要成组锁定；
- 升级任一基础版本后，应重新执行环境、转换、数值对齐和短训练检查；
- `runtime.json` 与 `resolved_config.yaml` 是每次实验的必需证据。

## 参考资料

- [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [Megatron Bridge Contributor Guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md)
- [Adding New Models](https://docs.nvidia.com/nemo/megatron-bridge/nightly/adding-new-models.html)
- [Using Recipes](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html)
- [DGX Spark System Overview](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html)
- [DGX Spark Container Runtime](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
- [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v)

以上 URL 与版本信息访问或核验日期为 2026-08-09。MiniMind-V commit、目标 HF checkpoint revision、DGX Spark 实机运行结果和最终数值容差仍待锁定。
