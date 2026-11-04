# BrickGPT 官方基线复现检查清单

核查日期：2026-11-04。官方源码基准为 `9c53c7cb65f9de0df89562cc74b2d2e270fbcab4`，
对应标签 `upstream-baseline-9c53c7c`；课程框架核查至 `53aae0d`。
依据是本仓库 README、`pyproject.toml`、`uv.lock`、官方推理/渲染源码及文末官方资料。

**当前仅完成源码与资料核查，服务器验收尚未开展。以下 Bash 命令都是服务器恢复后的操作，
本轮未安装依赖、下载模型权重或执行 GPU 推理。** 按步骤检查预期结果；失败时保留日志并排查，
不要直接跳到批量实验。所有命令从课程仓库根目录执行，路径变量按学校实际目录设置。

## 1. 前置条件与验收范围

| 项目 | 仅 TXT/LDraw 的真实生成 | 官方 `infer` 输出 TXT/LDraw/PNG | 说明 |
| --- | --- | --- | --- |
| Python、项目依赖、模型文件 | 必需 | 必需 | 使用官方锁文件与 Python 3.11 |
| Hugging Face 基座访问授权 | 在线首次加载必需 | 在线首次加载必需 | 基座为 gated Llama；已合法完整缓存时可离线加载 |
| HF 读权限 token 与网络 | 首次下载/鉴权需要 | 首次下载/鉴权需要 | GitHub 登录不等于 HF 模型授权 |
| `gurobipy` Python 包 | 当前官方导入链必需 | 当前官方导入链必需 | 即使 `use_gurobi=False` 也不能直接省掉这个包 |
| 有效、适配服务器的 Gurobi 许可证 | 完整物理检查必需 | 完整物理检查必需 | `False` 可做连通性诊断，但不是物理稳定性基线 |
| `bpy` Python 包 | 官方依赖中仍会安装；批量脚本无渲染时不导入渲染模块 | 必需 | 不要求额外启动 Blender GUI；需满足系统动态库要求 |
| ImportLDraw、LDraw 零件库、背景 EXR | 批量脚本无 `--render` 时可不准备 | 必需 | 官方 CLI 顶层导入渲染，运行中始终生成 PNG |
| NVIDIA GPU 与兼容驱动 | 选 `device=cuda` 时必需；CPU 可做慢速诊断 | 当前 Linux 官方渲染路径也要求可用 CUDA 设备 | `--device cpu` 只改变语言模型设备，不会把渲染改为 CPU |
| StableText2Brick 数据集 | 单条推理不需要 | 单条推理不需要 | 后续测试集评测另行准备 |
| FlashTex、纹理子模块、微调依赖 | 不需要 | 不需要 | 不运行纹理、mesh-to-brick、微调或 CLIP/DINO 评价 |

模型仓库当前包含 LoRA 的 `adapter_config.json` 和 `adapter_model.safetensors`，
其基座指向 `meta-llama/Llama-3.2-1B-Instruct`。适配器文件小，不意味着完整模型只需这些空间；
官方 `AutoModelForCausalLM.from_pretrained()` 会通过 PEFT 集成加载基座与适配器。
参见 [适配器配置](https://huggingface.co/AvaLovelace/BrickGPT/blob/main/adapter_config.json)
和 [对应 Transformers 版本的 PEFT 文档](https://huggingface.co/docs/transformers/v4.51.3/en/peft)。

## 2. 服务器与源码检查：先不下载模型

### 2.1 确认系统、磁盘和 GPU 分配

```bash
uname -m
getconf GNU_LIBC_VERSION
free -h
df -h "$PWD" "$HOME"
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,driver_version --format=csv
```

**预期：** Linux x86_64、glibc 至少 2.28、能看到分配的 RTX 2080 Ti，显存约 11 GB。
先预留至少约 20 GB 磁盘作为环境、模型缓存和资产的操作预算，不是实测占用；
系统 RAM、用户配额和 GPU 空闲量均需确认。锁定的 Torch 与 bpy Linux wheel 均标为
`manylinux_2_28_x86_64`，例如 glibc 2.17 的老系统不能直接按本清单安装。

**失败排查：** 看不到 GPU 时先确认是否已申请到计算节点/容器设备映射，不能在登录节点直接推理。
glibc 太旧交由管理员提供合适节点或获准的容器，不替换系统 libc。
`No space left` 同时检查磁盘和用户配额；不要将大型缓存放在容量很小的 home 分区。

此前提供的服务器信息是驱动 `535.247.01`、`nvidia-smi` 显示 CUDA 12.2，
该信息不是恢复后实测值。锁文件的 Linux Torch 2.7.0 依赖 CUDA 12.6 runtime。
`nvidia-smi` 的 CUDA 字段不是当前 Python wheel 的 CUDA 版本，也不是 `nvcc` 的安装证明。
CUDA 12.x 存在小版本兼容，但较新特性/PTX 可能仍要求新驱动，不能据此保证 535 驱动一定可用。
参见 [NVIDIA 小版本兼容规则](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

通过学校调度系统分配 GPU。若是允许直接选择设备的独占会话，可在确认 UUID 后设置：

```bash
# 仅在未由调度器设置设备可见性时执行；输入已获分配的 2080 Ti UUID。
read -r -p 'Allocated RTX 2080 Ti GPU UUID: ' BRICKGPT_GPU_UUID
export CUDA_VISIBLE_DEVICES="$BRICKGPT_GPU_UUID"
```

**预期：** PyTorch 后续把该设备映射为逻辑 `cuda:0`。不要固定物理 GPU 0，
学校机器还可能有其他 2080 Ti 和 T400 4 GB。Blender 的设备枚举另行检查，不假设其行为与 Torch 相同。

### 2.2 获取课程分支并验证官方文件未改动

若服务器没有仓库，先选择合适目录克隆；已有仓库则直接进入该目录，跳过克隆：

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone --branch feat/generation https://github.com/SunJulesWest/BrickGPT-Course-Project.git
cd BrickGPT-Course-Project
```

```bash
git status --short --branch
git branch --show-current
git remote -v
git rev-parse HEAD
git rev-parse 'upstream-baseline-9c53c7c^{}'
git diff --exit-code upstream-baseline-9c53c7c -- src/brickgpt pyproject.toml uv.lock .gitmodules
```

**预期：** 当前为 `feat/generation`，基准标签解引用得到上述完整官方 SHA；
最后的 diff 无输出，说明官方推理代码及依赖配置仍与基准一致。
克隆课程仓库默认只有 `origin`；若没有官方远程，再运行：

```bash
git remote add upstream https://github.com/AvaLovelace1/BrickGPT.git
```

**失败排查：** 分支不符、存在本地改动或标签不符时先核对，不使用 reset、强制拉取或覆盖队友文件。
旧的干净服务器克隆可运行 `git fetch origin` 和 `git merge --ff-only origin/feat/generation` 更新。
Linux 不要照搬 Windows 仓库中指向 `127.0.0.1:7890` 的本机代理配置；网络按学校节点实际设置。

## 3. Python 环境安装与导入检查

### 3.1 安装 uv，按锁文件建立环境

先运行 `uv --version`、`python3.11 --version`。已有可用 uv 时跳过安装器；否则使用
[uv 官方安装方法](https://docs.astral.sh/uv/getting-started/installation/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

```bash
uv sync --locked --python 3.11 --no-dev
.venv/bin/python --version
git diff --exit-code -- pyproject.toml uv.lock
```

**预期：** 创建项目 `.venv`，Python 3.11，安装锁定依赖，不修改锁文件。
缺少 Python 3.11 时 uv 可能下载解释器，也可使用学校现有解释器路径代替 `3.11`。
本步骤会下载大型 Python 包，只有服务器恢复且确认安装条件后才执行。
不运行 `uv sync --extra finetuning`，也不安装 FlashAttention 或量化扩展。

本仓库 Python 3.11 / Linux 的关键锁定版本：

| 组件 | 版本/限制 | 注意事项 |
| --- | --- | --- |
| Torch | 2.7.0 | Linux 锁定依赖含 CUDA runtime 12.6.77 |
| Transformers / PEFT | 4.51.3 / 0.15.2 | 直接升级可能改变缓存或适配器兼容行为 |
| bpy | 4.4.0 | 3.11 对应 Linux wheel；导入可能依赖系统图形库 |
| NumPy | 1.26.4；项目要求 `<2` | 不随意升级到 NumPy 2 |
| gurobipy | 12.0.2 | Python 包安装成功不代表完整许可证可用 |
| NetworkX | 3.4.2 | 当前由依赖树带入，用于连通性分析 |

**失败排查：** `--locked` 报锁文件过期时保存 uv 版本和错误，不直接重建锁文件；
先检查源码/解释器是否正确，必要时另行评估 uv 版本。
找不到 bpy wheel 先查 Python、架构和 glibc，不在默认路径中从源码编译 Blender。
不单独 `pip install -U torch transformers` 破坏已锁环境。
`uv run` 会自动同步，完成安装后的验收使用 `uv run --no-sync` 或 `.venv/bin/python`，
避免不知情的重新解析/安装。参见 [uv 同步行为](https://docs.astral.sh/uv/concepts/projects/sync/)。

### 3.2 记录版本并检查基础导入

```bash
.venv/bin/python - <<'PY'
from importlib.metadata import version
import torch
import bpy
import gurobipy
import networkx
from brickgpt.models import BrickGPTConfig
for name in ('torch', 'transformers', 'peft', 'bpy', 'numpy', 'gurobipy', 'networkx'):
    print(name, version(name))
print('torch CUDA runtime:', torch.version.cuda)
print('bpy:', bpy.app.version_string)
print('official initial temperature:', BrickGPTConfig().temperature)
PY
```

**预期：** 导入成功，不下载权重，不实例化 `BrickGPT`，默认初始温度 0.6。
`libGL.so.1`、`libXrender` 或其他 `.so` 缺失时，让管理员根据具体报错补系统库；
无桌面环境不自动意味着必须安装完整桌面或使用 Xvfb。
`ModuleNotFoundError: brickgpt` 检查是否从项目 `.venv` 执行且 editable 项目安装成功。

## 4. Hugging Face 访问与模型缓存

在浏览器使用自己的 HF 账号申请并接受
[Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 的访问条款，
确认审批状态，并创建具备相应读取权限的 token。
在服务器中交互输入，不把 token 写进配置、命令历史或 Git：

```bash
read -r -s -p 'HF read token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
```

如果默认缓存分区不足，先输入一个**自己有写权限、容量足够且不位于 Git 仓库内**的缓存目录：

```bash
read -r -p 'Writable Hugging Face cache directory: ' BRICKGPT_HF_CACHE
mkdir -p "$BRICKGPT_HF_CACHE"
export HF_HOME="$BRICKGPT_HF_CACHE"
df -h "$HF_HOME"
```

鉴权检查只请求 API 和权重文件 HEAD 元数据，不下载权重：

```bash
.venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
token = os.environ['HF_TOKEN']
api = HfApi(token=token)
print('HF account:', api.whoami()['name'])
for repo in ('AvaLovelace/BrickGPT', 'meta-llama/Llama-3.2-1B-Instruct'):
    info = api.model_info(repo)
    weights = [f.rfilename for f in info.siblings if f.rfilename.endswith('.safetensors')]
    if not weights:
        raise RuntimeError(f'No safetensors file listed for {repo}; inspect model layout.')
    metadata = get_hf_file_metadata(hf_hub_url(repo, weights[0], revision=info.sha), token=token)
    print('read access:', repo, 'revision:', info.sha, 'file:', weights[0], 'bytes:', metadata.size)
PY
```

**预期：** 两个仓库均可鉴权读取，记录各自 SHA；仅看见公开模型卡不等于具有权重读取权限。
HEAD 成功仍不保证大文件 CDN 下载能完成，首次真实加载时还需核对下载日志、剩余空间与实际解析的版本。
上述元数据接口见 [HF Hub 文档](https://huggingface.co/docs/huggingface_hub/v0.30.2/package_reference/file_download)。

**失败排查：** 401 查 token 是否生效；403 查基座授权及细粒度 token 权限；
超时/证书错误查服务器网络、代理和证书，不能把网络错误直接认定为授权失败。
离线节点需先由获授权账号在可联网环境准备完整基座、适配器与 tokenizer 缓存，
再按学校流程传入。只拷适配器目录不够。
官方配置没有直接暴露所有 HF revision/cache 参数，因此不伪造 `--revision` 等 CLI 开关；
要冻结 HF 文件版本需另行准备快照/离线缓存并记录基座和适配器版本，本轮不改加载器。

## 5. Gurobi 许可证：完整基线的检查关卡

向学校确认可用许可证类型、服务器绑定/容器要求、有效期及对 Gurobi 12 的支持。
如需申请，参见 [Gurobi 学术许可](https://www.gurobi.com/academia/academic-program-and-licenses/)。
Named-user 可能需要在学校网络上运行许可页面给出的 `grbgetkey` 命令；
`uv sync` 安装的 `gurobipy` 不保证附带这个工具，需要时按官方说明获取，不能用装包代替申请许可。
如果使用 WLS，网络和服务凭证也必须在目标节点可用。

对于已获准放在 home 下的文件许可证，可设置：

```bash
export GRB_LICENSE_FILE="$HOME/gurobi.lic"
test -r "$GRB_LICENSE_FILE"
.venv/bin/python - <<'PY'
import gurobipy as gp
from gurobipy import GRB
with gp.Model('license_preflight') as model:
    x = model.addVar(lb=0)
    model.addConstr(x >= 1)
    model.setObjective(x)
    model.optimize()
    assert model.Status == GRB.OPTIMAL
    print('Gurobi small problem OK:', gp.gurobi.version())
PY
```

**预期：** 日志显示正确学术/学校许可并求解成功。
这是 CPU 上的小问题，**即使通过也不能证明已获得无限规模许可**；
若日志为 `Restricted license`，仍需确认完整许可是否被正确加载。
官方稳定性问题可能超过受限许可规模，参见
[Gurobi 规模限制与许可定位](https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error)。

**失败排查：** Error 10009 查过期、文件路径、主机绑定、WLS 网络或许可版本；
`Model too large` 查是否误用受限许可。官方直接创建默认 Gurobi 环境，许可必须对该默认环境生效。
不输出或提交许可文件内容。
拿不到许可时可在后续命令改用 `--use_gurobi False` 做连通性诊断，保存为独立配置和日志，
标注“连通性替代”，完整物理基线这一项仍未通过。不要修改成“已验证物理稳定”。

## 6. PNG 渲染资产：ImportLDraw、LDraw 和 EXR

### 6.1 只初始化基线需要的子模块

```bash
git submodule update --init -- ImportLDraw
git -C ImportLDraw rev-parse HEAD
git submodule status ImportLDraw
```

**预期：** 提交为 `026a8c0d5d8c1af17bde3e6e960ff8c0308db38f`。
使用 `.gitmodules` 中 AvaLovelace1 的固定子模块，不换成上游仓库最新版，
不执行 `git submodule update --remote`，本阶段不初始化纹理相关子模块。
失败先检查 GitHub 网络及子模块本地改动，不覆盖子模块已有工作。

### 6.2 安装或指定 LDraw 零件库

已有零件库时跳过下载，只设置其根目录并执行文件检查。
新安装时可采用以下用户级目录；下载的是零件资源，不是 Python 模型：

```bash
export BRICKGPT_ASSET_ROOT="$HOME/.local/share/brickgpt"
mkdir -p "$BRICKGPT_ASSET_ROOT"
curl -fL https://library.ldraw.org/library/updates/complete.zip -o "$BRICKGPT_ASSET_ROOT/complete.zip"
unzip -tq "$BRICKGPT_ASSET_ROOT/complete.zip"
unzip -nq "$BRICKGPT_ASSET_ROOT/complete.zip" -d "$BRICKGPT_ASSET_ROOT"
export LDRAW_LIBRARY_PATH="$BRICKGPT_ASSET_ROOT/ldraw"
test -d "$LDRAW_LIBRARY_PATH/parts"
test -d "$LDRAW_LIBRARY_PATH/p"
test -s "$LDRAW_LIBRARY_PATH/parts/3005.dat"
sha256sum "$BRICKGPT_ASSET_ROOT/complete.zip"
```

**预期：** 变量指向包含 `parts/` 和 `p/` 的 `ldraw` 根目录，不是 `parts/` 本身。
库会更新，保留下载日期和校验和。路径存在但层级错误时，渲染器可能找不到零件；
环境变量路径不存在时源码还会回落到 `~/ldraw`，所以必须先检查，不依赖静默回落。
下载失败/压缩包损坏时不要继续解压；解压使用 `-n`，已有文件不会被覆盖。

### 6.3 准备准确命名的背景 EXR

从 README 指定的
[背景资源链接](https://drive.google.com/file/d/1Yux0sEqWVpXGMT9Z5J094ISfvxhH-_5K/view)
下载资源，按学校方式传到服务器。该链接可能需要浏览器人工操作；不要将 Google Drive HTML 页面保存为 EXR。
资源必须命名为 `ImportLDraw/loadldraw/background.exr`，固定子模块源码按这个文件名加载。

```bash
read -r -p 'Downloaded EXR file path on this server: ' BRICKGPT_EXR_SOURCE
cp -n -- "$BRICKGPT_EXR_SOURCE" ImportLDraw/loadldraw/background.exr
.venv/bin/python - <<'PY'
from pathlib import Path
p = Path('ImportLDraw/loadldraw/background.exr')
with p.open('rb') as stream:
    assert stream.read(4) == bytes.fromhex('762f3101'), 'Not an OpenEXR file'
print('EXR header OK:', p.stat().st_size)
PY
sha256sum ImportLDraw/loadldraw/background.exr
```

**预期：** 文件非空且 EXR 文件头正确。头部检查不保证解码成功，渲染仍需实测。
已存在的 EXR 不覆盖；校验后保留其来源和 hash。
大型资产和 `background.exr` 不提交到 Git；子模块可能因此显示未跟踪文件，
应在验收记录中说明，不把这种状态误认为官方代码改动。

## 7. RTX 2080 Ti 11 GB 兼容性与显存关卡

RTX 2080 Ti 的计算能力为 7.5，见 [NVIDIA 设备表](https://developer.nvidia.com/cuda/gpus)。
1B 级模型并不等于只需要 1 GB 显存。官方加载器没有显式设置 dtype、量化、device_map 或 CPU offload，
不能假设自动使用低精度或多卡，也不能给它传不存在的 `--torch_dtype` 参数。
按 FP32 估算，仅约 1B 级基座参数就占约 4–5 GB，另有 KV 缓存及拒绝采样中的缓存副本；
PNG 渲染还会与留在显存中的语言模型竞争资源。**11 GB 是否足够须实测，当前没有峰值数据。**

恢复后先做设备和一个基本算子检查，不加载模型：

```bash
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print('device:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
print('wheel architectures:', torch.cuda.get_arch_list())
print('free/total bytes:', torch.cuda.mem_get_info())
assert torch.cuda.get_device_capability(0) == (7, 5), 'Check GPU allocation'
value = torch.ones(1, device='cuda') + 1
torch.cuda.synchronize()
print('basic CUDA result:', value.item())
PY
```

**预期：** 识别到已分配的 2080 Ti，基本算子返回 2。
`no kernel image` 查 wheel 是否支持该架构；`unsupported PTX` / `requires newer driver`
记录驱动、Torch/CUDA 版本并请管理员判断驱动升级或另建兼容环境，不能直接改锁文件。
2080 Ti 不作为原生 BF16 目标，不启用要求更新架构的 FlashAttention 路径；
FP16/量化若后续确有需要，作为另一个有记录的配置变更评估，本轮不实施。

单独检查 Blender/Cycles 的设备，不加载语言模型或运行渲染：

```bash
.venv/bin/python - <<'PY'
import bpy
import ImportLDraw
from brickgpt.render_bricks import render_bricks
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
print('ImportLDraw:', ImportLDraw.__file__)
print('Cycles devices:', [(d.name, d.type) for d in prefs.devices])
PY
```

**预期：** ImportLDraw 导入成功，Cycles 枚举到 2080 Ti。
官方渲染代码会启用名字以 NVIDIA 开头的设备，不保证只用 Torch 选中的那张卡，
也不把多卡显存合并；若 T400 或别人的 GPU 同时可见，先由管理员/调度系统确认隔离方式，
不要直接在未隔离的多用户节点启动官方渲染。`--device cuda` 不能控制 Cycles 设备选择。

显存不足的排查顺序：确认空闲卡与其他进程 → 使用简短单提示词、串行执行 →
检查加载和渲染哪个阶段 OOM。不要终止别人的进程。
诊断时可另设 `--max_bricks 128` 限制结构长度，但该结果必须标注截断配置，不能冒充默认基线。
回退上限控制重复生成开销，不是直接的显存预算开关。
即使模型用 CPU，PNG 的 CUDA 问题也不会自动解决。

## 8. 最小验收 A：单条官方推理与三种输出

先通过 2–7 的前置检查，尤其是物理许可和渲染隔离。
在仓库内已忽略的目录下创建独立验收目录，后续命令沿用同一 shell 的变量：

```bash
mkdir -p experiments/runs
export BRICKGPT_ACCEPTANCE_DIR
BRICKGPT_ACCEPTANCE_DIR=$(mktemp -d "$PWD/experiments/runs/acceptance-XXXXXX")
mkdir -p "$BRICKGPT_ACCEPTANCE_DIR/official"
set -o pipefail
```

下面通过标准输入向官方交互 CLI 提供提示词、输出文件名、种子和最后的退出空行。
`EOF` 不加引号，目的是展开该输出目录；token 仅经环境变量提供。
**这是首次会加载并可能下载真实权重的步骤，当前未执行。**

```bash
uv run --no-sync infer --device cuda --temperature 0.6 --use_gurobi True <<EOF 2>&1 | tee "$BRICKGPT_ACCEPTANCE_DIR/official/infer.log"
A basic chair with four legs.
$BRICKGPT_ACCEPTANCE_DIR/official/chair.png
42

EOF
```

**预期：** 正常退出，打印生成耗时、拒绝/回退信息，并产生非空 `chair.txt`、`chair.ldr`、`chair.png`。
保持官方默认 world_dim、logit masking、拒绝与回退配置；不要把裸基座模型替换为 BrickGPT checkpoint。
本条耗时包含首次下载/加载的等待与渲染进程流程，不能直接和批量记录中的 `generation_seconds` 比较。
另一终端观察 `nvidia-smi` 的显存占用，记录峰值附近读数及是否存在其他占用；它不是精确的 CUDA 峰值仪器。

验证结果格式和三件输出：

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import struct
from brickgpt.data import BrickStructure
base = Path(os.environ['BRICKGPT_ACCEPTANCE_DIR']) / 'official'
for suffix in ('txt', 'ldr', 'png'):
    p = base / f'chair.{suffix}'
    assert p.is_file() and p.stat().st_size > 0, p
structure = BrickStructure.from_txt((base / 'chair.txt').read_text())
assert len(structure) > 0, 'Empty structure'
assert not structure.has_out_of_bounds_bricks(), 'Out of bounds'
assert not structure.has_collisions(), 'Colliding bricks'
assert any(line.startswith('1 ') for line in (base / 'chair.ldr').read_text().splitlines())
header = (base / 'chair.png').read_bytes()[:24]
assert header[:8] == b'\x89PNG\r\n\x1a\n'
assert struct.unpack('>II', header[16:24]) == (512, 512)
print('bricks:', len(structure), 'three output files OK')
stable = structure.is_stable()
print('physics check:', stable)
assert stable, 'Physical stability check did not pass; retain this failure case'
PY
```

**预期：** 文件格式检查通过，并能通过 Gurobi 执行独立稳定性判断；
若打印 False，不得把“生成成功”表述成“稳定性通过”，应记录为失败案例并核查回退是否耗尽。
人工打开 PNG 检查不是空白、缺件、严重裁切或只显示地面；文件头正确不保证视觉正确，也不证明文本匹配。
若使用的是 `use_gurobi=False` 诊断，最后一步只能另行检查 `is_connected()`，不得执行或声称通过物理验收。

**失败排查：** 只有 TXT/LDraw 没有 PNG，先查看渲染日志，不立即重新生成结构。
模型已返回后仍驻留显存，若渲染 OOM，可待该进程退出后在新进程单独渲染已保存的 LDraw：

```bash
uv run --no-sync render_bricks --in_file "$BRICKGPT_ACCEPTANCE_DIR/official/chair.ldr" --out_file "$BRICKGPT_ACCEPTANCE_DIR/official/chair.png"
```

这属于拆分渲染补救，需保留原失败日志，不能记为原始单进程 CLI 三件套一次通过。
如果错误指向 `background.exr`、零件路径或 `bpy` API，分别回到 6.3、6.2、3.2 和固定子模块版本排查。

## 9. 最小验收 B：通过官方单条后再验证批量脚本

先运行指定的无模型测试和一次 mock 调度，不运行整个官方推理测试目录：

```bash
.venv/bin/python -m unittest discover -s tests -p test_run_experiments.py -v
.venv/bin/python run_experiments.py --prompts examples/experiment_prompts.jsonl --temperatures 0.6 --seeds 42 --mock --output-dir "$BRICKGPT_ACCEPTANCE_DIR/mock"
```

**预期：** 14 项测试通过；mock 为 2 次成功，明确标记 `backend=mock`，不当作基线结果。
失败先定位框架/文件权限问题，不用真实模型掩盖 mock 测试错误。

从现有示例取同一条椅子提示词，写入本次验收配置：

```bash
head -n 1 examples/experiment_prompts.jsonl > "$BRICKGPT_ACCEPTANCE_DIR/prompts.jsonl"
cat > "$BRICKGPT_ACCEPTANCE_DIR/config.json" <<'JSON'
{
  "model_name_or_path": "AvaLovelace/BrickGPT",
  "device": "cuda",
  "use_gurobi": true,
  "max_brick_rejections": 500,
  "max_regenerations": 100,
  "use_logit_masking": true,
  "temperature_increase": 0.01,
  "max_temperature": 2.0
}
JSON
```

先验收单条真实批处理，无 PNG，避免同时引入渲染问题：

```bash
.venv/bin/python run_experiments.py --prompts "$BRICKGPT_ACCEPTANCE_DIR/prompts.jsonl" --temperatures 0.6 --seeds 42 --config "$BRICKGPT_ACCEPTANCE_DIR/config.json" --output-dir "$BRICKGPT_ACCEPTANCE_DIR/batch" 2>&1 | tee "$BRICKGPT_ACCEPTANCE_DIR/batch.log"
```

**预期：** 单行汇总显示 planned=1、pending=0、succeeded=1，generation_counts.succeeded=1。
生成 TXT、LDraw、bricks.json，PNG 路径为 null。真实 PNG 调度验收在显存/隔离条件满足时，
以相同参数追加 `--render` 再运行一次，保存为另一个 UUID 批次，不覆盖之前结果。

Gurobi 等组件可能向标准输出写日志，所以完整输出保存在 `batch.log`，不能把整个日志当作单个 JSON。
检查事件去重及后续评价所需接口，直接读取这次单条批处理的结构化文件：

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
from run_experiments import load_latest_records
root = Path(os.environ['BRICKGPT_ACCEPTANCE_DIR'])
journals = list((root / 'batch').glob('*/records.jsonl'))
assert len(journals) == 1, 'Use a fresh acceptance directory with exactly one batch'
journal = journals[0]
manifest = json.loads((journal.parent / 'manifest.json').read_text())
assert len(manifest['jobs']) == 1
rows = load_latest_records(journal)
assert len(rows) == 1
row = rows[0]
assert row['backend'] == 'official'
assert row['status'] == 'succeeded'
assert row['generation_status'] == 'succeeded'
assert row['initial_temperature'] == 0.6 and row['seed'] == 42
assert row['sampling_strategy']['temperature_increase'] == 0.01
assert row['sampling_strategy']['actual_temperature_trace'] is None
for kind in ('txt', 'ldr', 'bricks_json'):
    assert (journal.parent / row['artifacts'][kind]).is_file()
assert row['artifacts']['png'] is None
print('batch interface OK, evaluation join key:', row['run_id'])
PY
```

**排查与边界：** 非零退出时查看 JSONL 的 stage/error/traceback；不删除失败样本。
`generation_status=succeeded` 而整体 failed、stage=render 表示生成成功、后处理失败，
不能统计为生成失败。按 `run_id` 取最新事件后再与评价表一对一合并，不统计 running 事件行数。
模型初始化失败时整批失败，不反复重试下载；先修复 HF、环境或显存问题，再以新批次验收。
同种子不承诺跨设备/库版本逐字节相同；第一次只确认接口与配置一致，再分析生成差异。
初始 0.6 不是全过程固定温度，升温、上限和未采集实际轨迹的边界均以记录为准。

## 10. 人工确认与验收记录

- [ ] 计算节点、Linux/glibc、RAM、磁盘配额、网络与作业调度方式。
- [ ] 可独占的 2080 Ti UUID/空闲显存，以及 Cycles 不会误用 T400 或他人设备。
- [ ] 当前驱动与锁定 CUDA runtime 的兼容性；是否需要管理员调整环境。
- [ ] HF 基座授权、token 权限、缓存位置；基座/适配器实际 SHA 和合法缓存来源。
- [ ] 适配 Gurobi 12 的完整许可、有效期、主机/容器/WLS 条件，而非仅小问题可求解。
- [ ] 固定 ImportLDraw 提交、LDraw 资源日期/hash、可读取且来源明确的 background.exr。
- [ ] 单条官方命令、三件输出、视觉检查、独立物理检查、耗时及显存观察记录。
- [ ] 单条批处理结果与去重后的唯一 run_id，之后才扩展种子/温度组合。

验收结果保存在本次 `BRICKGPT_ACCEPTANCE_DIR`，建议额外记录 `git rev-parse HEAD`、
`uv --version`、`nvidia-smi` 输出、环境版本、资源 hash 和每一关的通过/失败原因。
这些是待填写的检查项，当前不宣称任何服务器验收已完成。

## 资料与源码定位

- 本仓库：[README](../README.md)、[依赖配置](../pyproject.toml)、[锁文件](../uv.lock)、[子模块配置](../.gitmodules)。
- 官方调用链：[infer.py](../src/brickgpt/infer.py) → [BrickGPT / 配置](../src/brickgpt/models/brickgpt.py) → [LLM 加载器](../src/brickgpt/models/llm.py)。
- 导出与渲染：[BrickStructure](../src/brickgpt/data/brick_structure.py)、[render_bricks.py](../src/brickgpt/render_bricks.py)。
- EXR 固定文件名：[固定 ImportLDraw 源码](https://github.com/AvaLovelace1/ImportLDraw/blob/026a8c0d5d8c1af17bde3e6e960ff8c0308db38f/loadldraw/loadldraw.py)。
- 课程脚本接口、事件读取及测试：[experiments.md](experiments.md)。
